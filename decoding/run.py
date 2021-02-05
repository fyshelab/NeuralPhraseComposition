#!user/bin/env/ python3
"""
Description: parsing arguments to the functions

Notes: multiprocessing is limited to one node

"""
__author__ = 'Maryam Honari'

import argparse
# import cpickle as pickle
import os
import time
import logging
import multiprocessing as multp
from datetime import datetime
import json
import re

import scipy.io as sio
import numpy as np
# import h5py

## for parallel
from itertools import repeat
from functools import partial
import multiprocessing as mp
from scipy.signal import butter, lfilter

from evaluator import Evaluator
from evaluatorParallel import *

# from preprocessor import Preprocessor
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)  # keep an I on runtimeerrors
# warnings.simplefilter(action='ignore')#, category=ResourceWarning)#, category=RuntimeWarning)
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

inData_opt = ['channels']
wordvec_opt = ['skipgram']
end_opt = ['local', 'cluster']
avg_opt = ['random']
whcond_opt = ['straight',  # train on x, test on x
              'ph/li',  # train on list test on phrase & viceversa
              'adj_train',  # train on adj only test on ph/li
              'noun_train',  # train on noun only test on ph/li
              'phrasal']

# parsing parameters
parser = argparse.ArgumentParser()

parser.add_argument('-v', '--verbose', action='store_false')  # reverse it later
parser.add_argument('-s', help='subject number', type=int, choices=range(0, 20), default=0)
parser.add_argument('--traind', '-t', help='type of train data',
                    default=inData_opt[0], choices=inData_opt)
parser.add_argument('--notemp', help='trains on the whole time after stumuli,def notemp=false',
                    action='store_true')  # by default it trains on temporal dimention
parser.add_argument('--wvec', help='word vector type', default=wordvec_opt[0],
                    choices=wordvec_opt)
parser.add_argument('-c', '--cluster', help='change paths to run on cluster',
                    action='store_true')
parser.add_argument('--avg', help='how to average trials for each examplar',
                    choices=avg_opt, default=avg_opt[0])  # change later
parser.add_argument('--whcond', help='condition types for train & test',
                    choices=whcond_opt, default=whcond_opt[0])

parser.add_argument('--bpass', help='bandpasses(low) data at 40 hz', action='store_true')

parser.add_argument('--isperm', help=' do permutation test', action='store_true')

parser.add_argument('--permnum', help='number of shuffles for permutation test', default=10)
parser.add_argument('--permst', help='number of shuffles for permutation test', default=0)
parser.add_argument('--procnum', help='number of precesses', default=2)  # on cluster 32
parser.add_argument('--tgm', help='do tgm', action='store_true')


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print('[%s]' % self.name, end=' ')
        print('Elapsed: %s' % (time.time() - self.tstart))


def wrapper_2v2_train_test(in_arg):
    # return do_2v2(*in_arg)
    return do_2v2_train_test(*in_arg)


def wrapper_2v2_train_test_noun_adj(in_arg):
    return do_2v2_train_test_noun_adj(*in_arg)


def wrapper_tgm_2v2_train_test_noun_adj(in_arg):
    # return do_2v2(*in_arg)
    return tgm_do_2v2_train_test_noun_adj(*in_arg)


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Global variables
subjects = ['A0003', 'A0005', 'A0007', 'A0008', 'A0014', 'A0051', 'A0054', 'A0056', 'A0057', 'A0068',
            'A0072', 'A0078', 'A0085', 'A0091', 'A0094', 'A0097', 'A0098', 'A0099', 'A0100', 'A0101']


class params():
    pass


eps = np.finfo(float).eps

if __name__ == '__main__':
    args = parser.parse_args()
    args.procnum = int(args.procnum)
    args.permnum = int(args.permnum)
    args.permst = int(args.permst)
    print('parsing...')

    if args.verbose:
        logging.basicConfig(level=logging.NOTSET)  # ~?
    print('inputdata: {}\nwordvec: {}\niscluster: {}\navrage: {}\ntrainTestCond: {}\n'.format(args.traind, args.wvec,
                                                                                              args.cluster, args.avg,
                                                                                              args.whcond))

    my_subj = args.s
    if subjects[my_subj] == 'A0008' and args.traind != 'channels':
        print('no source data for subject A0008')
        exit(0)

    # ## set time
    # def case
    time_start = -0.1  # change def_time_chunk if you're changing this
    time_end = 0.700  # 0.700
    time_step = 0.005  # origin: 0.005
    params.time_window = 0.1
    t_vec = np.arange(time_start, (time_end - params.time_window + eps), time_step)
    t_vec = np.around(t_vec, 3)
    if args.tgm:
        time_start = -0.1  # change def_time_chunk if you're changing this
        time_end = 0.700
        time_step = 0.025  # origin: 0.005
        params.time_window = 0.1
        t_vec = np.arange(time_start, (time_end - params.time_window + eps), time_step)
        t_vec = np.around(t_vec, 3)

    # notemp case
    elif args.notemp:
        time_start = 0 - eps
        time_end = 0.700
        time_step = 0.700  # origin: 0.005
        params.time_window = 0.700
        t_vec = np.arange(time_start, (time_end - params.time_window), time_step)

    params.avg = args.avg

    # ## where to run? local or cluster
    if args.cluster:
        params.data_dir = '/scratch/mahon/composition'
    else:
        params.data_dir = '..'
        params.out_data_dir = '../data/'  # out means outside of project

    # ## averaging configuration
    if args.avg == 'random':
        # number of instances per word, eg. 4 times noun 'bell'
        params.num_per_inst = 4  # NOTE: find out 4 or 5 works bether?
        params.num_labels = 5  # 5 unique labels

    # ##skipgram choosing
    if args.wvec == 'skipgram':
        skipgram = sio.loadmat('{}/wordvecs/skipgram/skipgram_vecs.mat'.format(params.data_dir))
        params.word_dims = range(0, skipgram['vectors'].shape[1])
        params.word_vecs = skipgram['vectors']
        params.dist_metric = 'cosine'
        params.words = skipgram['words']

    #  ## training data  choosing
    logging.info(subjects[my_subj])
    params.subjs = subjects[my_subj]

    adjnoun = sio.loadmat('{}/{}_ASL_NR_epoch_parsed.mat'.format(params.out_data_dir, subjects[my_subj]))

    params.channels = range(0, 208)

    adjnoun['data'] = adjnoun.get('data')[:, params.channels, :]

    # labels are 1-indexed changing them to 0
    adjnoun['labels'] = adjnoun['labels'] - 1

    # ## Frequency filter
    if args.bpass:
        logging.info('bandpassing data ...')
        butter_bandpass_filter(adjnoun.data, 0.1, 40, 1000, 20)

    logging.info('about to call 2v2 on {} | {} {} {} | {} {}\n'.format(params.data_dir,
                                                                           args.traind, args.wvec, args.whcond,
                                                                           args.avg,
                                                                           str(datetime.now())))

    my_out_dir = '{}/results/analysis_{}_{}_{}_tgm_{}/'.format(params.data_dir,
                                                                 args.whcond,
                                                                 args.wvec,
                                                                 args.avg,
                                                                 str(args.tgm))
    params.my_out_dir = my_out_dir
    try:
        if not os.path.exists(my_out_dir + '/' + params.subjs):
            os.makedirs(my_out_dir + '/' + params.subjs)
    except FileExistsError:
        if os.path.exists(my_out_dir + '/' + params.subjs):
            print("File exist error but file is there")
        else:
            print("File exist error and folder is not there", (my_out_dir + '/' + params.subjs))
            exit(0)

    if args.isperm is False:
        evaler = Evaluator(t_vec, subjects[my_subj], adjnoun, params, args)
        all_2v2 = []
        all_2v2_res = []
        if args.whcond == 'phrasal':
            all_2v2, all_2v2_res = evaler.do_2v2_phrasal(range(2, 4))
        if args.whcond == 'straight':
            # do_2v2_onesubj
            # all_2v2 = eval.do_2v2(range(2,4))
            all_2v2 = evaler.do_2v2_train_test(range(2, 4), range(0, 2))
        elif args.whcond == 'adj_train':
            if args.tgm:
                params.note = 'tgm training on adjective testing on phrase/list'
                all_2v2 = evaler.tgm_do_2v2_train_test_noun_adj()
            else:
                params.note = 'training on adjective testing on phrase/list'
                all_2v2 = evaler.do_2v2_train_test_noun_adj()
        elif args.whcond == 'noun_train':
            if args.tgm:
                params.note = 'tgm training on noun testing on phrase/list'
                all_2v2 = evaler.tgm_do_2v2_train_test_noun_adj()
            else:
                params.note = 'training on noun testing on phrase/list'
                all_2v2 = evaler.do_2v2_train_test_noun_adj()
            pass

        sio.savemat('{}/{}_test_picture_comp_run_classify_{}_on_{}_with_{}_averaging.mat'.format(
            my_out_dir, subjects[my_subj], args.whcond, args.traind, args.avg),
            {'all_2v2': all_2v2, 'subjs': subjects, 't_vec': t_vec, 'params': params, 'parser': args,
             'all_2v2_res': all_2v2_res})

    else:  # ## permutation test
        os.environ['MKL_NUM_THREADS'] = '1'
        os.environ['OPENBLAS_NUM_THREADS'] = '1'

        if args.whcond == 'straight':
            # TODO: do_2v2_onesubj
            my_out_dir = '{}/results/permtest_on_channels_{}_{}_{}_notemp_{}_tgm_{}/'.format(
                params.data_dir,
                args.whcond,
                args.wvec,
                args.avg,
                str(args.notemp),
                str(args.tgm))  # parser.Results.istemporal)

            try:
                if not os.path.exists(my_out_dir + '/' + params.subjs):
                    os.makedirs(my_out_dir + '/' + params.subjs)
            except FileExistsError:
                if os.path.exists(my_out_dir + '/' + params.subjs):
                    print("File exist error but file is there")
                else:
                    print("File exist error and folder is not there", (my_out_dir + '/' + params.subjs))
                    exit(0)
            print('procnum: ', args.procnum)
            print('permnum: ', args.permnum)
            print('permst: ', args.permst)
            pool = mp.Pool(processes=int(args.procnum))

            manager = mp.Manager()
            adjnoun = manager.dict(adjnoun)
            ns = manager.Namespace()
            ns.t_vec = t_vec
            ns.subj = subjects[my_subj]
            ns.params = params
            ns.args = args
            ns.my_out_dir = my_out_dir

            repnum = np.arange(args.permst, args.permst + args.permnum)
            logging.info('repnum: {}'.format(repnum))

            results = pool.map(wrapper_2v2_train_test,
                               zip(repeat(range(2, 4), args.permnum), repeat(adjnoun), repeat(ns), repnum))
            # results = pool.map(wrapper_2v2, zip(repeat(range(2, 4), args.permnum), repeat(adjnoun), repeat(ns), repnum))

            # FIXME: fix the range later
            print(results)

            # TODO: save data
            sio.savemat('{}/{}_test_picture_comp_run_classify_{}_on_{}_with_{}_averaging.mat'.format(
                my_out_dir, subjects[my_subj], args.whcond, args.traind, args.avg),
                {'res_all_2v2': results, 'subj': subjects[my_subj], 'subjs': subjects, 't_vec': t_vec,
                 'params': params, 'parser': args})
            pass

        if args.whcond == 'adj_train' or args.whcond == 'noun_train':
            # TODO: do_2v2_onesubj
            my_out_dir = '{}/results/permtest_on_channels_{}_{}_{}_notemp_{}_tgm_{}/'.format(params.data_dir,
                                                                                               args.whcond,
                                                                                               args.wvec,
                                                                                               args.avg,
                                                                                               str(args.notemp),
                                                                                               str(
                                                                                                   args.tgm))  # parser.Results.istemporal)

            try:
                if not os.path.exists(my_out_dir + '/' + params.subjs):
                    os.makedirs(my_out_dir + '/' + params.subjs)
            except FileExistsError:
                if os.path.exists(my_out_dir + '/' + params.subjs):
                    print("File exist error but file is there")
                else:
                    print("File exist error and folder is not there", (my_out_dir + '/' + params.subjs))
                    exit(0)
            print('procnum: ', args.procnum)
            print('permnum: ', args.permnum)
            print('permst: ', args.permst)
            pool = mp.Pool(processes=int(args.procnum))

            manager = mp.Manager()
            adjnoun = manager.dict(adjnoun)
            ns = manager.Namespace()
            # ns = argparse.Namespace()
            ns.t_vec = t_vec
            ns.subj = subjects[my_subj]
            ns.params = params
            ns.args = args
            ns.my_out_dir = my_out_dir

            repnum = np.arange(args.permst, args.permst + args.permnum)

            print(repnum)
            if args.tgm:

                results = pool.map(wrapper_tgm_2v2_train_test_noun_adj,
                                   zip(repeat(adjnoun), repeat(ns), repnum))
            else:
                results = pool.map(wrapper_2v2_train_test_noun_adj,
                                   zip(repeat(range(2, 4), args.permnum), repeat(adjnoun), repeat(ns), repnum))

            # ##save data
            sio.savemat('{}/{}_test_picture_comp_run_classify_{}_on_{}_with_{}_averaging.mat'.format(
                my_out_dir, subjects[my_subj], args.whcond, args.traind, args.avg),
                {'res_all_2v2': results, 'subj': subjects[my_subj], 'subjs': subjects, 't_vec': t_vec,
                 'params': params, 'parser': args})
