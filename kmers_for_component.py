import time
import sys
import re
import pdb,math
import os
import os.path
import collections
from collections import defaultdict
from weight_updated_graph import weight_updated_graph

def run_cmd(s1):
    print(s1); os.system(s1) 

class Counter():
    '''Used for printing the number of kmers processed'''
    def __init__(self, name, report_length):
        self.name = name
        self.count = 0
        self.report_length = report_length

    def increment(self):
        self.count += 1
        if self.count % self.report_length == 0:
            print "{:s}: {:s}, processed {:d}".format(time.asctime(), self.name, self.count)

reverse_complement = lambda x: ''.join([{'A':'T','C':'G','G':'C','T':'A'}[B] for B in x][::-1])
'''def reverse_complement(bases):
    replacements = [('A', 't'), ('T', 'a'), ('C', 'g'), ('G', 'c')]
    for ch1, ch2 in replacements:
        bases = re.sub(ch1, ch2, bases)
    return bases[::-1].upper()'''


def kmers_for_component(k1mer_dictionary,kmer_directory, reads_files, directory_name, contig_file_extension, get_partition_k1mers, double_stranded = True, paired_end = False, repartition = False,  partition_size = 500, overload = 1.5, K = 24, gpmetis_path = 'gpmetis', penalty = 5, only_reads = False):
    """This fuction runs gpmetis on the components above a threshold size.  It then creates a dictionary called 
    k1mers2component {k1mer : component}.  It then sends any reads that share a kmer with a component to that component.
    It then cycles through the k1mer file and outputs the k1mers along with their weights to a file for each component.
    It then creates a kmers2component dictionary.  It then outputs a kmers file for each component. 
    Inputs: 
    k1mer_dictionary
    kmer_directory: used instead of k1mer_dictionary, currently unused
    reads_files: where reads are
    directory_name: where files should be stored
    contig_file_extension: the contig files are in this extension
    """
    
    if os.path.exists(directory_name+"/before_sp_log.txt"):
        f_log = open(directory_name+"/before_sp_log.txt", 'a')
    else:
        f_log = open(directory_name+"/before_sp_log.txt", 'w')
    
    def write_log(s):
	f_log.write(s + "\n"); print(s)
    
    # Counts number of components above size threshold
    Num_Components = 0
    i = 1
    while os.path.exists(directory_name+"/component" + str(i) + "contigs.txt"):
        i += 1
        Num_Components += 1  

    # Counts number of components below size threshold
    Num_Remaining_Components = 0
    i = 1
    while os.path.exists(directory_name+"/remaining_contigs" + str(i) + ".txt"):
        i += 1
        Num_Remaining_Components += 1   
    
    # def find_comps(comp_dict, tries):
    #     #Only the exact 
    #     (comp,hits) = max(comp_dict.iteritems(), key=operator.itemgetter(1))
    #     if hits>=tries:
    #         return set(comp)
    #     else:
    #         return set()

    def get_rmers(read,R):
        i = 0; k1mers = []
        while i < len(read) - R:
            k1mers.append(read[i:i+R])
            i+=R
        k1mers.append(read[-R:])
        return k1mers

    def get_comps(read,k1mers2component):
        k1mers = get_rmers(read,K+1);
        comps = [k1mers2component.get(k1mer, [-1,-1]) for k1mer in k1mers]
        comp_list = [set(comp[0]) for comp in comps if comp[0] != -1]
        if comp_list:
                assigned_comp = set.union(*comp_list)
        else:
                assigned_comp = set()
	return assigned_comp

    def get_comps_paired(read1,read2,k1mers2component):
    	return set.union(get_comps(read1,k1mers2component),get_comps(read2,k1mers2component))

    if get_partition_k1mers:
        ufactor = int(1000.0*overload - 1000.0)
        components_broken = {}
        temp_string = ""
        
        # Runs gpmetis
        for i in range(Num_Components):
            with open(directory_name+"/component" + str(i+1) + contig_file_extension , 'r') as f:
                lines = f.readlines()
                num_contigs = len(lines)
                Partitions = min(int(math.ceil(float(num_contigs)/float(partition_size))),100)
                components_broken[i] = Partitions          
                temp_string += "Component " + str(i) + ": " + str(Partitions) + " partitions, "  
                if len(lines) >= 2:
                    run_cmd(gpmetis_path + " -ufactor="+str(ufactor) + " "+directory_name+"/component" + str(i+1) + ".txt" + " " +str(Partitions))
                    if repartition:
                        #Create GPMETIS file for repartition
                        partition_file = "/component" + str(i+1) + ".txt" + ".part." + str(components_broken[i])
                        og_graph_file = "/component" + str(i+1) + ".txt"
                        new_graph_file = "/component" + str(i+1) + "r2.txt"
                        contig_file = "/component" + str(i+1) + contig_file_extension;
                        new_contig_file = contig_file #Currently unused option
                        randomize = False
                        write_log(str(time.asctime()) + ": " + "Creating graph for repartition ")
                        weight_updated_graph(directory_name, partition_file, og_graph_file, new_graph_file, contig_file, new_contig_file, penalty, randomize)
                        write_log(str(time.asctime()) + ": " + "Created graph for repartition ")
                        #Rerun GPMETIS file for repartition
                        run_cmd(gpmetis_path + " -ufactor="+str(ufactor) + " "+directory_name+"/component" + str(i+1)  + "r2.txt " +str(Partitions))
                    
        
        write_log(str(time.asctime()) + ": " + "gpmetis for partitioning is complete \n " + temp_string)
        
        new_components = {}
        k1mers2component = {}
        kmers2component = {}


        
        # Builds k1mer2component dictionary
        for i in components_broken:
            with open(directory_name+"/component" + str(i+1) + contig_file_extension, 'r') as f_contigs:
                contig_lines = f_contigs.readlines()
                with open(directory_name+"/component" + str(i+1)  + ".txt.part." + str(components_broken[i]) , 'r') as f_component:
                    j = 0
                    for line in f_component:
                        tokens = line.split()
                        comp = 'c' + str(i+1) + "_" + tokens[0]
                        contig = contig_lines[j].split()[0]
                        if comp not in new_components:
                            new_components[comp] = [contig]
                        else:
                            new_components[comp].append(contig)
                        for each in range(len(contig)-(K+1) + 1):
                            k1mer = contig[each:each+(K+1)]

                            if k1mer not in k1mers2component:
                                k1mers2component[k1mer] = [set([comp]), k1mer_dictionary.get(k1mer,0)]
                            else:
                                k1mers2component[k1mer][0].add(comp)
                        j += 1
                if repartition:
                    with open(directory_name+"/component" + str(i+1)  + "r2.txt.part." + str(components_broken[i]) , 'r') as f_component:
                        j = 0
                        for line in f_component:
                            tokens = line.split()
                            comp = 'r2_c' + str(i+1) + "_" + tokens[0]
                            contig = contig_lines[j].split()[0]
                            if comp not in new_components:
                                new_components[comp] = [contig]
                            else:
                                new_components[comp].append(contig)
                            for each in range(len(contig)-(K+1) + 1):
                                k1mer = contig[each:each+(K+1)]
                                if k1mer not in k1mers2component:
                                    k1mers2component[k1mer] = [set([comp]), k1mer_dictionary.get(k1mer,0)]
                                else:
                                    k1mers2component[k1mer][0].add(comp)
                            j += 1

                        
                         
        # Adds remaining components to k1mers2component
        if 1:
            for i in range(Num_Remaining_Components):
                with open(directory_name+"/remaining_contigs"+str(i+1)+".txt", 'r') as remaining_contigs_file:
                    if 1:
                        lines = remaining_contigs_file.readlines()
                        comp = "cremaining"+str(i+1)
                        j = 0
                        for line in lines:
                            tokens = line.split()
                            contig = tokens[0]
                            if comp not in new_components:
                                new_components[comp] = [contig]
                            else:
                                new_components[comp].append(contig)
                            for each in range(len(contig)-(K+1) + 1):
                                k1mer = contig[each:each+(K+1)]
                                if k1mer not in k1mers2component:
                                    k1mers2component[k1mer] = [set([comp]), k1mer_dictionary.get(k1mer,0)]
                                else:
                                    k1mers2component[k1mer][0].add(comp)
                            j += 1                             

        write_log(str(time.asctime()) + ": " + "k1mers2component dictionary created ")
        
        '''no_kmers_in_comp = {}
        for comp in new_components:
            temp = 0
            for contig in new_components[comp]:
                temp += len(contig)
            no_kmers_in_comp[comp] = temp'''
        

                            

        '''iter_tag = "_c"
        if second_iteration:
            iter_tag = "_r2_c"'''
        read_line =''
        
        # Assigns reads to components in the non paired end case
        if paired_end == False:  
            read_part_seq = {}   
            for comp in new_components:
                read_part_seq[comp] = []; #open(directory_name+"/reads"+iter_tag+str(comp)+".fasta", 'w')
            with open(reads_files[0]) as readfile:
                for line in readfile:
                    if line.split()[0][0] == ">":
                        read_line = line
                    else:
                        read = line.split()[0]
                        if read.strip('ACTG'): continue #Contains characters other than ACTG
                        assigned_comp = get_comps(read,k1mers2component)
                        for each_comp in assigned_comp:
                            read_part_seq[each_comp].append(read_line)
                            read_part_seq[each_comp].append(line)

                            
                        
                        if double_stranded:
                            rc_read = reverse_complement(read); 
                            assigned_comp = get_comps(rc_read,k1mers2component)
                            for each_comp in assigned_comp:
                                reversed_read_name=read_line.split()[0]+'_reversed'+'\t' +'\t'.join(read_line.split()[1:])
                                read_part_seq[each_comp].append(reversed_read_name+'\n')
                                read_part_seq[each_comp].append(rc_read+'\n')                            
            for comp in new_components:
                read_part_file = open(directory_name+"/reads"+str(comp)+".fasta", 'w')
                read_part_file.write("".join(read_part_seq[comp]))
                read_part_file.close()

        # Assigns reads to components in the paired end case
        elif paired_end == True:
            read1_part_seq = {}
            read2_part_seq = {}
            for comp in new_components:
                read1_part_seq[comp] = []; # open(directory_name+"/reads"+iter_tag+str(comp)+"_1.fasta", 'w')
                read2_part_seq[comp] = []; #open(directory_name+"/reads"+iter_tag+str(comp)+"_2.fasta", 'w')
            comp2reads = {}

            comp2reads_reversed = {}
            readfile1 = open(reads_files[0], 'r')
            readfile2 = open(reads_files[1], 'r')
            read_line1 = ''; read_line2 = ''
            with open(reads_files[0]) as readfile1, open(reads_files[1]) as readfile2:
                for line1,line2 in zip(readfile1,readfile2):
                    if line1.split()[0][0] == ">":
                        assert line2.split()[0][0] == ">"
                        read_line1 = line1
                        read_line2 = line2
                    else:
                        assert line2.split()[0][0] != ">"
                        read1 = line1.split()[0]
                        read2 = line2.split()[0]
                        if read1.strip('ACTG') or read2.strip('ACTG'): continue #Dont write 'N' reads
                        read1_reversed = reverse_complement(read1)
                        read2_reversed = reverse_complement(read2)
                        
                        #First process (read1, read2_reversed)
                        assigned_comp = get_comps_paired(read1,read2_reversed,k1mers2component)

                        for each_comp in assigned_comp:
                            read1_part_seq[each_comp].append(read_line1)
                            read1_part_seq[each_comp].append(line1)
                            read2_part_seq[each_comp].append(read_line2)
                            read2_part_seq[each_comp].append(line2)

                        if double_stranded:
                            #Now process (read1_reversed, read2)
                            assigned_comp = get_comps_paired(read1_reversed, read2, k1mers2component)
                            for each_comp in assigned_comp:
                                reversed_read1_name=read_line1.split()[0]+'_reversed'+'\t'+'\t'.join(read_line1.split()[1:])
                                reversed_read2_name=read_line2.split()[0]+'_reversed'+'\t'+'\t'.join(read_line2.split()[1:])
                                read1_part_seq[each_comp].append(reversed_read1_name+'\n')
                                read1_part_seq[each_comp].append(read1_reversed+'\n')
                                read2_part_seq[each_comp].append(reversed_read2_name+'\n')
                                read2_part_seq[each_comp].append(read2_reversed+'\n')
            for comp in new_components:
                read1_part_file = open(directory_name+"/reads"+str(comp)+"_1.fasta", 'w')
                read2_part_file = open(directory_name+"/reads"+str(comp)+"_2.fasta", 'w')
                read1_part_file.write("".join(read1_part_seq[comp]))
                read2_part_file.write("".join(read2_part_seq[comp]))
                read1_part_file.close()
                read2_part_file.close()
        
        write_log(str(time.asctime()) + ": " + "reads partititoned ")        
        
        '''c2 = Counter("k1mer Streaming", 10**6)

        # Cycles through k1mer file and adds k1mer weights to k1mers2component dictionary


        with open(kmer_directory + "/k1mer.dict" , 'r') as f:
            for line in f:
                if len(line) == 0: continue
                c2.increment()
                k1mer, weight = line.split()
                k1mer = k1mer.upper()
                weight = (float(weight))
                if k1mer in k1mers2component:
                    k1mers2component[k1mer][1] += weight
                if double_stranded:
                    r_k1mer = reverse_complement(k1mer)
                    if r_k1mer in k1mers2component:
                        k1mers2component[r_k1mer][1] += weight

        write_log(str(time.asctime()) + ": " + "k1mers weights stored in dictionary")'''

        '''write_log(str(time.asctime()) + ": " + "Computing kmer weights.")
        #Update the kmer dictionary weights
        kmer_dictionary = collections.Counter()

        for kp1mer in k1mer_dictionary:
                k1, k2 = kp1mer[:K], kp1mer[-K:]
                kmer_dictionary[k1] += k1mer_dictionary[kp1mer]
                kmer_dictionary[k2] += k1mer_dictionary[kp1mer]                
        write_log(str(time.asctime()) + ": " + "Computed kmer weights.")'''
        if not only_reads: #If only_reads, no need to write k1mers
            write_log(str(time.asctime()) + ": Writing k1mers to file")
            # Writes out k1mers with weights for each partition
            for comp in new_components:
                with open(directory_name+"/component" + comp + "k1mers_allowed.dict" , 'w') as k1mer_file:
                    k1mer_file_data = []
                    for contig in new_components[comp]:
                        for i in range(len(contig)-(K+1) + 1):
                            k1mer = contig[i:i+(K+1)]; #rc = reverse_complement(k1mer); rw = k1mers2component.get(rc,[0,0])
                            k1mer_file_data.append(k1mer + "\t" + str(k1mers2component[k1mer][1]) + "\n")
                            #k1mer_file.write(k1mer + "\t" + str(k1mers2component[k1mer][1] + rw[1]) + "\n")
                        '''for i in range(len(contig)-(K) + 1):
                            kmer = contig[i:i+(K)]
                            kmer_file.write(kmer + "\t" + str(kmer_dictionary[kmer]) + "\n")'''
                    k1mer_file.writelines(k1mer_file_data)
            write_log(str(time.asctime()) + ": " + "k1mers written to file ")
                        
        #del(k1mers2component)
        
        '''kmers2component = {}
        
        # builds kmers2components dictionary that will store kmer weights
        for i in components_broken:
            with open(directory_name+"/component" + str(i+1) + contig_file_extension, 'r') as f_contigs:
                with open(directory_name+"/component" + str(i+1) + graph_file_extension + ".part." + str(components_broken[i]) , 'r') as f_component:
                    contig_lines = f_contigs.readlines()
                    j = 0
                    for line in f_component:
                        tokens = line.split()
                        comp = str(i+1) + "_" + tokens[0]
                        contig = contig_lines[j].split()[0]
                        for each in range(len(contig)-(K) + 1):
                            kmers2component[contig[each:each+(K)]] = 0.0
                        j += 1
                        
        # adds kmers from remaining components if not second iterarion
        if not second_iteration:
            for i in range(Num_Remaining_Components):
                with open(directory_name+"/remaining_contigs"+str(i+1)+".txt", 'r') as remaining_contigs_file:
                    if 1:
                        lines = remaining_contigs_file.readlines()
                        comp = "remaining"+str(i+1)

                        j = 0
                        for line in lines:
                            tokens = line.split()
                            contig = tokens[0]
                            for each in range(len(contig)-(K) + 1):
                                kmers2component[contig[each:each+(K)]] = 0.0
                            j += 1                            

        write_log(str(time.asctime()) + ": " + "kmers2component dictionary created ")
                            
        c2 = Counter("kmer Streaming", 10**6)

        # gets kmer weights from kmer file and adds them to kmers2component
        with open(kmer_directory + "/kmer.dict" , 'r') as f:
            for line in f:
                if len(line) == 0: continue
                c2.increment()
                kmer, weight = line.split()
                kmer = kmer.upper()
                weight = (float(weight))
                if kmer in kmers2component:
                    kmers2component[kmer] += weight
                if double_stranded:
                    r_kmer = reverse_complement(kmer)
                    if r_kmer in kmers2component:
                        kmers2component[r_kmer] += weight

        write_log(str(time.asctime()) + ": " + "kmers weights stored in dictionary")'''



                        
        '''# Writes out kmers and weights for each partition          
        for comp in new_components:
            with open(directory_name+"/component" + comp  + "kmers_allowed.dict" , 'w') as new_file:
                for contig in new_components[comp]:
                    for i in range(len(contig)-(K) + 1):
                        kmer = contig[i:i+(K)]
                        new_file.write(kmer + "\t" + str(kmer_dictionary[kmer]) + "\n")'''

        write_log(str(time.asctime()) + ": " + "kmers written to file " + "\n")
        newcomps = [c for c in new_components]                
        return [components_broken, newcomps]

