#!/bin/bash
i=20

let "subj = $i % 20"
# subj=$i%20
let "start=$i/20"
echo "i: $i, subj: $subj, st: $start"

# for i in {0..1000}
# do
# subj=

# python run.py -s $i --wvec skipgram  --traind channels --avg random --notemp
# python run.py -s 1 --wvec skipgram --whcond noun_train --traind channels --avg random --notemp --isperm --permnum 1 --permst 0 --procnum 1
# done