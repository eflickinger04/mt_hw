To train the implemented IBM model on 10,000 sentences and report alignment error rate:

python3 align-ibm -n 10000 | python score-alignments 

To train the implemented IBM2 model on 10,000 sentences and report alignment error rate:

python3 align-ibm2 -n 10000 | python score-alignments 

If you want to output the alignments to file called 
"dice.a": 
python align > dice .a
python align-ibm > dice .a
python align-ibm2 > dice .a

-- Existing Text --
There are three python programs here (`-h` for usage):

- `./align` aligns words.

- `./check-alignments` checks that the entire dataset is aligned, and
  that there are no out-of-bounds alignment points.

- `./score-alignments` computes alignment error rate.

The commands work in a pipeline. For instance:

   > ./align -t 0.9 -n 1000 | ./check | ./grade -n 5

The `data` directory contains a fragment of the Canadian Hansards,
aligned by Ulrich Germann:

- `hansards.e` is the English side.

- `hansards.f` is the French side.

- `hansards.a` is the alignment of the first 37 sentences. The 
  notation i-j means the word as position i of the French is 
  aligned to the word at position j of the English. Notation 
  i?j means they are probably aligned. Positions are 0-indexed.
