# NeuralPhraseComposition
experiment code for paper "Neural representation of words within phrases: Temporal evolution of color-adjectives and object-nouns during simple composition"

## Run decoding
 All decoding scripts are in decoding folder.
 To run the python code:
install the packages from requirements.txt and run the python code:
```
 usage: run.py [-h] [-v] [-s {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}] [--traind {channels}] [--notemp] [--wvec {skipgram}] [-c] [--avg {random}]
              [--whcond {straight,ph/li,adj_train,noun_train,phrasal}] [--bpass] [--isperm] [--permnum PERMNUM] [--permst PERMST] [--procnum PROCNUM] [--tgm]

optional arguments:
  -h, --help            show this help message and exit
  -v, --verbose
  -s {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19}
                        subject number
  --traind {channels}, -t {channels}
                        type of train data
  --notemp              trains on the whole time after stumuli,def notemp=false
  --wvec {skipgram}     word vector type
  -c, --cluster         change paths to run on cluster
  --avg {random}        how to average trials for each examplar
  --whcond {straight,ph/li,adj_train,noun_train,phrasal}
                        condition types for train & test
  --bpass               bandpasses(low) data at 40 hz
  --isperm              do permutation test
  --permnum PERMNUM     number of shuffles for permutation test
  --permst PERMST       number of shuffles for permutation test
  --procnum PROCNUM     number of precesses
  --tgm                 do tgm

 ```


 
 
 
 
 
 
 
 
 
