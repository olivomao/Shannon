import util

def write_cutoff(cutoff, weights, transcripts, false_positives):
    weights = weights[int(0.1*cutoff*len(weights)):]
    good_names = set([x[0] for x in weights])

    transcripts = [(name, seq) for name, seq in transcripts if name in good_names]
    hits = 0
    for name, seq in transcripts:
        if name in false_positives:
            hits += 1
    print('{} false positives for {}'.format(hits, cutoff))
    util.seqs_to_fasta('drop{}/transcripts.fa'.format(cutoff), transcripts)

transcripts = util.from_fasta('reconstructed.fa')

weights = []
zero = 0
with open('abundances/abundance.tsv') as f:
    f.readline()
    for line in f:
        name, _, _, _, weight = line.split()
        if float(weight) == 0:
            zero += 1
        weights.append((name, float(weight)))
weights.sort(key = lambda x: x[1])

false_positives = set()
with open('false_positives.txt') as f:
    for line in f:
        name, code, length = line.split()
        if int(length) < 200: continue
        if code == '0':
            false_positives.add(name)
print('{} false positives'.format(len(false_positives)))

for cutoff in range(10):
    write_cutoff(cutoff, weights, transcripts, false_positives)
