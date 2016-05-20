import time
import sys
import re
import pdb,math
import os
import os.path
import numpy as np
import tester
from filter_trans import filter_trans
import test_suite
import subprocess
import run_parallel_cmds

from kmers_for_component import kmers_for_component
from process_concatenated_fasta import process_concatenated_fasta
from extension_correction import  extension_correction

#Set Paths
#test
shannon_dir = os.path.dirname(os.path.abspath(sys.argv[0])) + '/' 
gpmetis_path = 'gpmetis'
jellyfish_path = '/opt/quorum/bin/jellyfish'#'jellyfish'
gnu_parallel_path = 'parallel'
quorum_path = '/opt/quorum/bin/quorum'#'quorum'
python_path = 'python'


#For version
version = '0.0.1'
	
# For jellyfish
double_stranded = True
run_jellyfish = True
paired_end = False # Automatically set if command line is used
jellyfish_kmer_cutoff = 1
find_reps = True


# General, Can be read in from terminal
reads_files = ['~/Full_Assembler/SPombe_algo_input/reads.fasta'] # ['./S15_SE_algo_input/reads.fasta']
sample_name = "SPombe_test"

# For extension correction
run_extension_corr =True
hyp_min_weight = 3
hyp_min_length = 75
partition_size = 500
use_partitioning = True

# For kmers_for_component
use_second_iteration = True
get_partition_kmers = True
overload = 2
K = 24
penalty = 5
nJobs = 1

# General runtime options
run_parallel = False
compare_ans = False
only_reads = False #Use only the reads from partitioning, not the kmers

# Everything beyond this point does not need to be set
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 


def run_cmd(s1):
		#print(s1); 
		os.system(s1) # + '>> temp_out.txt')


def test_install():
	exit_now = False; 
	print('Checking the various dependencies')
	print('--------------------------------------------')
	if test_suite.which(jellyfish_path):
		print('Using jellyfish in ' + test_suite.which(jellyfish_path))
		a=subprocess.check_output([jellyfish_path,'--version'])
		if len(a) < 11:
			print('Unable to automatically determine jellyfish version. Ensure that it is version 2.0.0 or greater')
		else:
			if a[10] != '2':
				print('Jellyfish version does not seem to be greater than 2.0.0. Please ensure that it is version 2.0.0 or greater, continuing run...')

	else:
		print('ERROR: Jellyfish not found. Set variable jellyfish_path correctly'); exit_now = True
	if test_suite.which(gpmetis_path):
		print('Using GPMETIS in ' + test_suite.which(gpmetis_path))
	else:
		print('ERROR: GPMETIS not found in path. Set variable gpmetis_path correctly'); exit_now = True
	try:
		import cvxopt
	except ImportError, e:
		print('ERROR: CVXOPT not installed into Python. Please see online manual for instructions.'); exit_now = True
	return exit_now

def test_install_quorum():
	if test_suite.which(quorum_path):
		print('Using Quorum in ') + test_suite.which(quorum_path)
	else:
		print('ERROR: Quorum not found in path. Set variable quorum_path correctly'); 
		sys.exit()


def test_install_gnu_parallel():
	if test_suite.which(gnu_parallel_path):
		print('Using GNU Parallel in ') + test_suite.which(gnu_parallel_path)
	else:
		print('ERROR: GNU Parallel not found in path. If you need to run multi-threaded, GNU Parallel is needed. Set variable gnu_parallel_path correctly'); exit_now = True



def print_message():
	print('--------------------------------------------')
	print('Shannon: RNA Seq de novo Assembly')
	print('Version: ' + version)
	print('--------------------------------------------')


print_message()
exit_now = test_install()
# Read input from terminal
n_inp = sys.argv[1:]
if '--help' in n_inp:
	with open('manual.md','r') as fin:
		print fin.read()
	exit_now = True
	sys.exit()

if '--version' in n_inp:
	print_message()
	exit_now = True
	sys.exit()

if '--compare' in n_inp:
	ind1 = n_inp.index('--compare')
	compare_ans = True
	ref_file = n_inp[ind1+1]
	ref_file = os.path.abspath(ref_file)
	n_inp = n_inp[:ind1]+n_inp[ind1+2:]    

if '--strand_specific' in n_inp:
	ind1 = n_inp.index('--strand_specific')
	double_stranded = False
	n_inp = n_inp[:ind1]+n_inp[ind1+1:]    
	print('OPTIONS --strand_specific: Single-stranded mode enabled')

if '--fastq' in n_inp:
	ind1 = n_inp.index('--fastq')
	run_quorum = False
	n_inp = n_inp[:ind1]+n_inp[ind1+1:]    
	print('OPTIONS --strand_specific: Single-stranded mode enabled')


if '--ss' in n_inp:
	ind1 = n_inp.index('--ss')
	double_stranded = False
	n_inp = n_inp[:ind1]+n_inp[ind1+1:]    
	print('OPTIONS --ss: Single-stranded mode enabled')


if '-s' in n_inp:
	ind1 = n_inp.index('-s')
	double_stranded = False
	n_inp = n_inp[:ind1]+n_inp[ind1+1:]    
	print('OPTIONS -s: Single-stranded mode enabled')


if '-p' in n_inp:
	ind1 = n_inp.index('-p')
	nJobs = int(n_inp[ind1+1])
	n_inp = n_inp[:ind1]+n_inp[ind1+2:] 
	run_parallel = True
	print('OPTIONS -p: Running parallel with ' + str(nJobs) + ' jobs.')

if '--partition' in n_inp:
	ind1 = n_inp.index('--partition')
	partition_size = int(n_inp[ind1+1])
	n_inp = n_inp[:ind1]+n_inp[ind1+2:] 
	print('OPTIONS --partition: Partition size set to ' + str(partition_size))	
	
if '--only_reads' in n_inp:
	ind1 = n_inp.index('--only_reads')
	n_inp = n_inp[:ind1]+n_inp[ind1+1:] 
	print('OPTIONS --only_reads: Reads are partitioned but kmers are recomputed.')	
	only_reads = True
	
if '-K' in n_inp:
	ind1 = n_inp.index('-K')
	K = int(n_inp[ind1+1])
	n_inp = n_inp[:ind1]+n_inp[ind1+2:]
	print('OPTIONS -K: Kmer size set to ' + str(K))

if '--kmer_hard_cutoff' in n_inp:
	ind1 = n_inp.index('--kmer_hard_cutoff')
	jellyfish_kmer_cutoff = int(n_inp[ind1+1])
	n_inp = n_inp[:ind1]+n_inp[ind1+2:]
	print('OPTIONS --kmer_hard_cutoff: Kmer hard cutoff set to ' + str(jellyfish_kmer_cutoff))

if '--kmer_soft_cutoff' in n_inp:
	ind1 = n_inp.index('--kmer_soft_cutoff')
	hyp_min_weight = int(n_inp[ind1+1])
	n_inp = n_inp[:ind1]+n_inp[ind1+2:]
	print('OPTIONS --kmer_soft_cutoff: Kmer soft cutoff set to ' + str(hyp_min_weight))


if '-o' in n_inp:
	ind1 = n_inp.index('-o')
	comp_directory_name = n_inp[ind1+1]
	comp_directory_name=os.path.abspath(comp_directory_name)
	n_inp = n_inp[:ind1]+n_inp[ind1+2:]
	if os.path.isdir(comp_directory_name) and os.listdir(comp_directory_name):
		print('ERROR: Output directory specified with -o needs to be an empty or non-existent directory')
		exit_now = True
else:
	print('ERROR: Output directory needed. Use -o flag, which is mandatory.')
	exit_now = True



reads_files = []

if '--left' in n_inp and '--right' in n_inp:
	ind1 = n_inp.index('--left')
	reads_files.append(os.path.abspath(n_inp[ind1+1]))
	n_inp = n_inp[:ind1]+n_inp[ind1+2:]

	ind2 = n_inp.index('--right')
	reads_files.append(os.path.abspath(n_inp[ind2+1]))
	n_inp = n_inp[:ind2]+n_inp[ind2+2:]

elif '--single' in n_inp:
	ind1 = n_inp.index('--single')
	reads_files.append(os.path.abspath(n_inp[ind1+1]))
	n_inp = n_inp[:ind1]+n_inp[ind1+2:]

else:
	print("ERROR: Need to specify single-ended reads with --single or specify both --left and --right for paired-ended reads.")
# if len(n_inp)>1:
#     comp_directory_name = n_inp[1]
#     reads_files = [n_inp[2]]
#     if len(n_inp)>3:
#         reads_files.append(n_inp[3])
# else:

if n_inp:
	print('OPTIONS WARNING: Following options not parsed: ' + " ".join(n_inp))


	''''with open('manual.md','r') as fin:
		print fin.read()'''

if exit_now:
	print('Try running python shannon.py --help for a short manual')   
	sys.exit()
else:
        print('--------------------------------------------')
	print "{:s}: Starting Shannon run..".format(time.asctime())

if len(reads_files) == 1:
	paired_end = False
elif len(reads_files) == 2:
	paired_end = True

run_quorum = False
if reads_files[0][-1] == 'q': #Fastq mode
	run_quorum = True
	test_install_quorum()

if run_parallel:
	test_install_gnu_parallel()

	
paired_end_flag = ""
if paired_end:
	paired_end_flag = " --paired_end "
	

# For extension correction
sample_name = comp_directory_name.split('/')[-1] + "_"
new_directory_name = comp_directory_name
sample_name_input = comp_directory_name + "/" + sample_name
comp_size_threshold = partition_size
run_og_components = False
get_og_comp_kmers = False

# For kmers_for_component
kmer_directory = sample_name_input+"algo_input" # "./S15_SE_algo_input" # K = 24
base_directory_name = comp_directory_name #"./S15_SE_contig_partition"
contig_file_extension = "contigs.txt"

randomize = False 

if use_partitioning == False:
	partition_size = 100000000000000000000000000000000000000000000000000000000000000000
	comp_size_threshold = partition_size

run_cmd('mkdir ' + comp_directory_name)
run_cmd('mkdir ' + sample_name_input+ "algo_input")


#Run Quorum now
if run_quorum:
	print "{:s}: Running Quorum for read error correction with quality scores..".format(time.asctime())
	run_cmd(python_path + ' ' + shannon_dir + 'run_quorum.py ' + quorum_path + ' ' + comp_directory_name + ' ' + '\t'.join(reads_files) + " -t " + str(nJobs))
	if paired_end:
		reads_files = [comp_directory_name + '/corrected_reads_1.fa',comp_directory_name + '/corrected_reads_2.fa']
	else:
		reads_files = [comp_directory_name + '/corrected_reads.fa']

reads_string = ' '.join(reads_files)    

# Runs Jellyfish
if run_jellyfish:
	print "{:s}: Starting Jellyfish to extract Kmers from Reads..".format(time.asctime())
	K_value = K
	run_jfs = ' '
	if double_stranded:
		run_jfs += ' -C '
	#run_cmd('rm '+sample_name_input+'algo_input/jelly*')  #Remove old jellyfish files
	run_cmd(jellyfish_path+' count -m ' + str(K_value+1) + run_jfs+ ' -o ' + sample_name_input+'algo_input/jellyfish_p1_output.jf -s 20000000 -c 4 -t ' + str(nJobs) + " " +reads_string)

	run_cmd(jellyfish_path+' dump -c -t -L ' + str(jellyfish_kmer_cutoff) + ' ' + sample_name_input+'algo_input/jellyfish_p1_output.jf > ' +sample_name_input+'algo_input/k1mer.dict_org')
	if (not run_extension_corr) and double_stranded:
		tester.double_strandify(sample_name_input+'algo_input/k1mer.dict_org', sample_name_input+'algo_input/k1mer.dict')
	if (not run_extension_corr) and (not double_stranded):
		run_cmd('mv ' + sample_name_input+'algo_input/k1mer.dict_org ' + sample_name_input+'algo_input/k1mer.dict')
	print "{:s}: Jellyfish finished..".format(time.asctime())

# Runs error correction for k1mers (Deletes error k1mers) using contig approach
# and determines seperate groups of contigs that share no kmers (components)
if run_extension_corr:
	#run_cmd('rm ' + base_directory_name+"/component*contigs.txt")    

	if double_stranded:
		str_ec = ' -d '
	else: 
		str_ec = ' '
	
	#run_cmd('python ' + shannon_dir + 'extension_correction.py ' + str_ec + sample_name_input+'algo_input/k1mer.dict_org ' +sample_name_input+'algo_input/k1mer.dict ' + str(hyp_min_weight) + ' ' + str(hyp_min_length) + ' ' + comp_directory_name + " " + str(comp_size_threshold))
	str_ec += sample_name_input+'algo_input/k1mer.dict_org ' +sample_name_input+'algo_input/k1mer.dict ' + str(hyp_min_weight) + ' ' + str(hyp_min_length) + ' ' + comp_directory_name + " " + str(comp_size_threshold)
	k1mer_dictionary = extension_correction(str_ec.split())

# Gets kmers from k1mers
'''if run_jellyfish or run_extension_corr:
	run_cmd('python ' + shannon_dir + 'kp1mer_to_kmer.py ' + sample_name_input+'algo_input/k1mer.dict ' + sample_name_input+'algo_input/kmer.dict')'''

# Runs gpmetis to partition components of size above "partition_size" into partitions of size "partition_size"
# Gets k1mers, kmers, and reads for each partition
[components_broken, new_components] = kmers_for_component(k1mer_dictionary,kmer_directory, reads_files, base_directory_name, contig_file_extension, get_partition_kmers, double_stranded, paired_end, use_second_iteration, partition_size, overload, K, gpmetis_path, penalty, only_reads)

del k1mer_dictionary
# This counts remaining and non-remaining partitions for log.
num_remaining = 0
num_non_remaining = 0 
for part in new_components:
	if 'remaining' in part:
		num_remaining += 1
	else:
		num_non_remaining += 1

# This code updates the log
if os.path.exists(comp_directory_name+"/before_sp_log.txt"):
	f_log = open(comp_directory_name+"/before_sp_log.txt", 'a')
else:
	f_log = open(comp_directory_name+"/before_sp_log.txt", 'w')
f_log.write(str(time.asctime()) + ": " +"Number of simple Partitions: " + str(num_remaining)  + "\n")
print(str(time.asctime()) + ": " +"Number of simple Partitions: " + str(num_remaining))
f_log.write(str(time.asctime()) + ": " +"Number of complex Partitions: " + str(num_non_remaining)  + "\n")
print(str(time.asctime()) + ": " +"Number of complex Partitions: " + str(num_non_remaining)  + "\n")
f_log.close()
		
# parameters for main_server call
main_server_parameter_string = ""
main_server_og_parameter_string = ""

# Create directories for each partition where run_MB_SF.py will be run
for comp in new_components:
	dir_base = comp_directory_name + "/" + sample_name + str(comp)
	run_cmd("mkdir " + dir_base + "algo_input")
	run_cmd("mkdir " + dir_base + "algo_output")
	if paired_end:
		run_cmd("mv " + base_directory_name + "/reads" + str(comp) + "_1.fasta " + dir_base + "algo_input/reads_1.fasta")
		run_cmd("mv " + base_directory_name + "/reads" + str(comp) + "_2.fasta " + dir_base + "algo_input/reads_2.fasta")
	else:
		run_cmd("mv " + base_directory_name + "/reads" + str(comp) + ".fasta " + dir_base + "algo_input/reads.fasta")

	#run_cmd("mv " + base_directory_name + "/component" + comp + "kmers_allowed.dict " + dir_base + "algo_input/kmer.dict")
	if not only_reads: #if only_reads, no need to copy k1mers
		run_cmd("mv " + base_directory_name + "/component" + str(comp)  + "k1mers_allowed.dict " + dir_base + "algo_input/k1mer.dict")
	main_server_parameter_string = main_server_parameter_string + dir_base + " " 
	
child_names = [x[0][:-10] for x in os.walk(comp_directory_name) if x[0].endswith('algo_input') and not x[0].endswith('_algo_input') and not x[0].endswith('allalgo_input')]
main_server_parameter_string = ' '.join(child_names)

		
# Run run_MB_SF.py for each partition in parallel
mb_sf_param_string = " "
'''if double_stranded:
	mb_sf_param_string += "  --ds " '''
if only_reads:
	mb_sf_param_string += "  --only_reads "
mb_sf_param_string += "  --nJobs "	+ str(nJobs) + " "

if main_server_parameter_string:
	if run_parallel and nJobs > 1:
		cmds = []
		for param_str in main_server_parameter_string.split():
			cmds.append(python_path + " " + shannon_dir + "run_MB_SF.py " + param_str + " --run_alg " + mb_sf_param_string + " --kmer_size " + str(K)  + " " + paired_end_flag + " --dir_name " + comp_directory_name + " " + param_str + " --shannon_dir " + shannon_dir + " --python_path " + python_path)
		cmds = tuple(cmds)
		run_parallel_cmds.run_cmds(cmds,nJobs)
		#run_cmd(gnu_parallel_path + " -j " + str(nJobs) + " " + python_path + " " + shannon_dir + "run_MB_SF.py {} --run_alg " + mb_sf_param_string + " --kmer_size " + str(K)  + " " + paired_end_flag + " --dir_name " + comp_directory_name + " --shannon_dir " + shannon_dir + " --python_path " + python_path +  " ::: " + main_server_parameter_string)
	else:
		for param_str in main_server_parameter_string.split():
				run_cmd(python_path + " " + shannon_dir + "run_MB_SF.py " + param_str + " --run_alg " + mb_sf_param_string + " --kmer_size " + str(K)  + " " + paired_end_flag + " --dir_name " + comp_directory_name + " " + param_str + " --shannon_dir " + shannon_dir + " --python_path " + python_path)

if os.path.exists(comp_directory_name+"/before_sp_log.txt"):
	f_log = open(comp_directory_name+"/before_sp_log.txt", 'a')
else:
	f_log = open(comp_directory_name+"/before_sp_log.txt", 'w')

# updates log

# locates all reconstructed files

reconstructed_files = comp_directory_name+"/reconstructed_single_contigs.fasta "
for comp in new_components:
	dir_base = comp_directory_name + "/" + sample_name + str(comp)
	dir_out = dir_base + "algo_output"
	reconstructed_files +=  (dir_out + '/' + 'reconstructed.fasta ')
	
# Creates new directory with concatenation of all reconstructed files
dir_base = comp_directory_name + "/" + sample_name + "all"
dir_out = dir_base + "algo_output"
run_cmd("mkdir " + dir_out)
out_file = dir_out + "/" + "all_reconstructed.fasta"
run_cmd("cat " + reconstructed_files + " > " + out_file)
process_concatenated_fasta(out_file, dir_out + "/reconstructed_org.fasta")
f_log.write(str(time.asctime()) + ': All partitions completed.\n')

#run_cmd('cp ' + dir_out + "/reconstructed.fasta " + dir_out + "/reconstructed_org.fasta")

if find_reps:
	f_log.write(str(time.asctime()) + ': Finding representative outputs\n')
	run_cmd('cat ' +  dir_out + "/reconstructed_org.fasta | perl -e 'while (<>) {$h=$_; $s=<>; $seqs{$h}=$s;} foreach $header (sort {length($seqs{$a}) <=> length($seqs{$b})} keys %seqs) {print $header.$seqs{$header}}' > " +  dir_out +  "/reconstructed_sorted.fasta " )
	run_cmd(python_path + ' ' + shannon_dir + 'faster_reps.py -d ' + dir_out + "/reconstructed_sorted.fasta " + dir_out + "/reconstructed.fasta ")
	f_log.write(str(time.asctime()) + ': Representative outputs found.\n')
else:	
	run_cmd('mv '  + dir_out + "/reconstructed_org.fasta " + dir_out + "/reconstructed.fasta ")


# Compares reconstructed file against reference
if compare_ans:
	run_cmd("cp " + ref_file + ' ' +  dir_out + "/reference.fasta")
	run_cmd(python_path + " " + shannon_dir + "run_MB_SF.py " + dir_base + " --compare --shannon_dir " + shannon_dir + " --python_path " + python_path)

num_transcripts = 0
with open(dir_out + "/" + "reconstructed.fasta", 'r') as reconstructed_transcripts:
	num_transcripts = int(len(reconstructed_transcripts.readlines())/2.0)
f_log.write(str(time.asctime()) + ": " +"All partitions completed: " + str(num_transcripts) + " transcripts reconstructed" + "\n")
print(str(time.asctime()) + ": " +"All partitions completed: " + str(num_transcripts) + " transcripts reconstructed" + "\n")
f_log.close()

# Creates final output
run_cmd('mkdir '+ comp_directory_name + '/TEMP')
run_cmd('mv ' + comp_directory_name + '/*_* ' + comp_directory_name + '/TEMP')
run_cmd('mv ' + comp_directory_name + '/TEMP/before_sp_log.txt ' + comp_directory_name + '/log.txt')
run_cmd('mv ' +  comp_directory_name + "/TEMP/" + sample_name + "allalgo_output/reconstructed.fasta " + comp_directory_name + '/shannon.fasta')
run_cmd('more ' +  comp_directory_name + "/TEMP/*output.txt > " +comp_directory_name + '/terminal_output.txt') 

if compare_ans:
   run_cmd('mv ' +   comp_directory_name + "/TEMP/" + sample_name + "allalgo_output/reconstr_log.txt "  + comp_directory_name + '/compare_log.txt') 


print("-------------------------------------------------")
print(str(time.asctime()) + ": Shannon Run Completed")
print("-------------------------------------------------")

 
