#!/bin/bash
source activate mne

# train on noun only, test on every
for i in {0..19}
do
python run.py -s $i --wvec skipgram --whcond noun_train --traind channels --avg random # --notemp
python run.py -s $i --wvec skipgram --whcond adj_train --traind channels --avg random # --notemp
done

#train on adj-only, test on every
# for i in {0..19}
# do
# python run.py -s $i --wvec skipgram --whcond adj_train --traind channels --avg random --notemp
# done


# # train on ph/li, test on ph/li and li/ph
# for i in {0..19}
# do
# python run.py -s $i --wvec skipgram  --traind channels --avg random --notemp
# done