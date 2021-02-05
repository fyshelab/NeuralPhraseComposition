from __future__ import division
from itertools import combinations

import numpy as np

import time
import os
from datetime import datetime
import joblib
import logging
from scipy.spatial.distance import cdist
import scipy.io as sio
from sklearn.model_selection import KFold, GridSearchCV
from regressor import VectorRegressor

try:
    import cPickle as pickle
except:
    import pickle

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self):
        if self.name:
            print('[%s]' % self.name, end=' ')
        print('Elapsed: %s' % (time.time() - self.tstart))


def do_2v2(taskindList, adjnoun, ns, repnum): # t_vec, subj , adjnoun, params):
    # IMPORTANT: assumed the desired channels have been already selected in adjnoun
    # IMPORTANT: labels are 1-indexed in the data file changing them into 0-indexed
    # IMPORTANT: taskindList is 1-indexed
    # NOTE: repnum indicates which index in the random stream is used to shuffle the labels after averaging

    fileName = '{}/{}/{}_{}_perm_classify_{}_on_{}_with_{}_averaging.mat'.format(
            ns.my_out_dir, ns.params.subjs, repnum, ns.subj, ns.args.whcond, ns.args.traind, ns.args.avg)
    if os.path.exists(fileName):
        loaded = sio.loadmat(fileName)
        all_2v2 = loaded['all_2v2']
    else:
        all_2v2 = np.zeros((ns.t_vec.shape[0], 5, 2))
    for wordind in range(0, 2):  # label index (first word spoken, second word spoken) #FIXME: range(0,2)
        for taskind in taskindList: # taskind is chosen based on .mat file and is 1-indexed
            if taskind ==5 and ns.subj =='A0056': # bad data for this task
                continue
            if taskind > 3 and wordind > 0:
                break
            ### selecting data
            data = data_select(taskind, adjnoun, ns.t_vec, ns.params)

            print('process numnbr: ', repnum)
            ### average data

            a, avrg_alltime_data, avrg_labels = avg_data(wordind, taskind, data, adjnoun, ns.params, repnum)
            for timeind, _ in enumerate(ns.t_vec):  # NOTE: Timeind all
                if all_2v2[timeind, taskind-1, wordind] != 0:
                    continue
                logging.info('{} Task {} word {} t {:.3f}\t'.format(ns.subj, taskind, wordind, ns.t_vec[timeind]))
                avrg_data = avrg_alltime_data[:, :,
                       np.logical_and(np.squeeze(adjnoun['time'] > ns.t_vec[timeind]),
                                      np.squeeze(adjnoun['time'] <= (
                                              ns.t_vec[timeind] + ns.params.time_window)))]

                ### training
                word_vecs = ns.params.word_vecs[avrg_labels.astype(int), :]
                word_vecs = word_vecs[:, ns.params.word_dims]
                tim = Timer()
                tim.__enter__()
                all_ests, all_targs, _, _ = leave_two_out(avrg_data, word_vecs, a, 1)
                tim.__exit__()
                logging.info('mem {} {}'.format(repnum, memory_usage_psutil())) # NOTE: remove

                all_res = make_results(all_ests, all_targs, ns.params.dist_metric)
                all_2v2[timeind, taskind-1, wordind] = 100 * np.mean(all_res)
                logging.critical('2v2 Acc {:.4f}\t\n{}\n\n'.format(100 * np.mean(all_res),
                                                                   str(datetime.now())))
                if timeind % 5 == 0:
                    sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': ns.t_vec, 'params': ns.params,
                         'parser': ns.args, 'repnum': repnum})

        sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': ns.t_vec, 'params': ns.params,
             'parser': ns.args, 'repnum': repnum})


    return all_2v2


def leave_two_out(X, y, a, foldZscore, Xp=None, yp=None, Xl=None, yl=None):
    all_ests = np.zeros((a.shape[0], 2, y.shape[1]))
    all_targs = np.zeros((a.shape[0], 2, y.shape[1]))
    p_all_ests = np.zeros((a.shape[0], 2, y.shape[1]))
    p_all_targs = np.zeros((a.shape[0], 2, y.shape[1]))
    l_all_ests = np.zeros((a.shape[0], 2, y.shape[1]))
    l_all_targs = np.zeros((a.shape[0], 2, y.shape[1]))
    for j in range(a.shape[0]):
        folds = np.zeros(y.shape[0])
        folds[a[j, :]] = 1
        reg = VectorRegressor(fZscore=1, folds=folds)
        ## Train
        reg.fit(X, y)

        ## predict
        all_ests[j, :, :], all_targs[j, :, :] = reg.predict(X, y)
        if Xp is not None:
            _, _, scaler =reg.transform(Xp,yp)
            p_all_ests[j, :, :], p_all_targs[j, :, :] = reg.predict(Xp, yp, scaler)
        if Xl is not None:
            _, _, scaler =reg.transform(Xl,yl)
            l_all_ests[j, :, :], l_all_targs[j, :, :] = reg.predict(Xl, yl, scaler)

    return all_ests, all_targs, p_all_ests, p_all_targs, l_all_ests, l_all_targs


def do_2v2_train_test(trainTasklList, adjnoun, ns, repnum): # testTaskList
    # TODO: when averaging based on utterance, there's different number of instances
    # TODO: between 1-word and 2word conditions
    # IMPORTANT: assumed the desired channels have been already selected in adjnoun
    # IMPORTANT: labels must be 0-indexed
    # IMPORTANT: trainTaskList and testTaskList are 1-indexed
    fileName = '{}/{}/{}_{}_perm_classify_{}_on_{}_with_{}_averaging.mat'.format(
        ns.my_out_dir, ns.params.subjs, repnum, ns.subj, ns.args.whcond, ns.args.traind, ns.args.avg)
    if os.path.exists(fileName):
        loaded = sio.loadmat(fileName)
        all_2v2 = loaded['all_2v2']
        if all_2v2.shape == tuple((ns.t_vec.shape[0], 5, 2)):
            all_2v2_new = np.zeros((ns.t_vec.shape[0], 5, 2, 2))
            all_2v2_new[:, :, :, 0] = all_2v2
            all_2v2 = all_2v2_new
            all_2v2_new = None
    else:
        all_2v2 = np.zeros((ns.t_vec.shape[0], 5, 2, 2)) #Note: time, task, word, train,test

    for wordind in range(0,2):  # label index (first word spoken, second word spoken)

        ### general begin
        for taskind in trainTasklList:  # taskind is chosen based on .mat file and is 1-indexed
            taskind_test = 5 - taskind
        ### general end
            if taskind == 5 and ns.subj == 'A0056':  # bad data for this task
                continue
            if taskind ==5 and wordind == 0: # training on noun only but wordind is 0
                break
            if taskind == 4 and wordind == 1:
                break
            data_train = data_select(taskind, adjnoun, ns.t_vec, ns.params)
            data_test = data_select(taskind_test, adjnoun, ns.t_vec, ns.params )
            ### average data
            # check labels_train va labels_test yeki bashe
            a, avrg_alltime_train, avrg_labels_train = avg_data(wordind, taskind, data_train, adjnoun, ns.params, repnum)
            a, avrg_alltime_test, avrg_labels_test = avg_data(wordind, taskind_test, data_test, adjnoun, ns.params, repnum)

            logging.critical('process numnbr: '.format(repnum))
            for timeind, _ in enumerate(ns.t_vec):
                if all_2v2[timeind, taskind-1, wordind, 0] != 0 and  all_2v2[timeind, taskind_test-1, wordind, 1] != 0:
                    continue
                logging.info('{} trainTask {} testTask {}, word {} t {:.3f}\t'.format(
                    ns.subj, taskind, taskind_test, wordind, ns.t_vec[timeind]))
                ### selecting data

                ### average data
                # check labels_train va labels_test yeki bashe
                avrg_data_train = avrg_alltime_train[:, :,
                                  np.logical_and(np.squeeze(adjnoun['time'] > ns.t_vec[timeind]),
                                                 np.squeeze(adjnoun['time'] <= (
                                                     ns.t_vec[timeind] + ns.params.time_window)))]
                avrg_data_test = avrg_alltime_test[:, :,
                                  np.logical_and(np.squeeze(adjnoun['time'] > ns.t_vec[timeind]),
                                                 np.squeeze(adjnoun['time'] <= (
                                                     ns.t_vec[timeind] + ns.params.time_window)))]
                ### training
                word_vecs = ns.params.word_vecs[avrg_labels_train.astype(int), :]
                word_vecs = word_vecs[:, ns.params.word_dims]

                all_ests, all_targs, all_ests_test, all_targs_test, _, _ = \
                    leave_two_out(avrg_data_train, word_vecs, a, 1, avrg_data_test, word_vecs)

                all_res = make_results(all_ests, all_targs, ns.params.dist_metric)
                all_res_test = make_results(all_ests_test, all_targs_test, ns.params.dist_metric)
                all_2v2[timeind, taskind-1, wordind, 0] = 100 * np.mean(all_res)
                all_2v2[timeind, taskind_test - 1, wordind, 1] = 100 * np.mean(all_res_test)
                logging.critical('2v2 Acc {:.4f}\t 2v2 Acc test {:.4f}\t\n{}\n\n'.format(100 * np.mean(all_res),
                                                                                         100 * np.mean(all_res_test),
                                                                                         str(datetime.now())))
                if timeind % 5 == 0:
                    sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': ns.t_vec, 'params': ns.params,
                         'parser': ns.args, 'repnum': repnum})

        sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': ns.t_vec, 'params': ns.params,
                               'parser': ns.args, 'repnum': repnum})
    return all_2v2

def do_2v2_train_test_noun_adj(trainTasklList, adjnoun, ns, repnum): # testTaskList
    # TODO: when averaging based on utterance, there's different number of instances
    # TODO: between 1-word and 2word conditions
    # IMPORTANT: assumed the desired channels have been already selected in adjnoun
    # IMPORTANT: labels must be 0-indexed
    # IMPORTANT: trainTaskList and testTaskList are 1-indexed
    fileName = '{}/{}/{}_{}_perm_classify_{}_on_{}_with_{}_averaging.mat'.format(
        ns.my_out_dir, ns.params.subjs, repnum, ns.subj, ns.args.whcond, ns.args.traind, ns.args.avg)
    if os.path.exists(fileName):
        loaded = sio.loadmat(fileName)
        all_2v2 = loaded['all_2v2']

    else:
        all_2v2 = np.zeros((ns.t_vec.shape[0], 5, 2)) #Note: time, task, word, train,test

    for wordind in range(0,2):  # label index (first word spoken, second word spoken)
        if ns.args.whcond == 'noun_train':
            taskind = 5
        elif ns.args.whcond == 'adj_train':
            taskind = 4
        else:
            return

        ### noun begin
        taskind_test = [2, 3] #, taskind]  # taskind is chosen based on .mat file and is 1-indexed

        if taskind == 5 and ns.subj == 'A0056':  # bad data for this task
            break
        if taskind ==5 and wordind == 0: # training on noun only but wordind is 0
            continue
        if taskind == 4 and wordind == 1:
            continue
        data_train = data_select(taskind, adjnoun, ns.t_vec, ns.params)
        data_test_0 = data_select(taskind_test[0], adjnoun, ns.t_vec, ns.params)
        data_test_1 = data_select(taskind_test[1], adjnoun, ns.t_vec, ns.params)
        ### average data
        # check labels_train va labels_test yeki bashe
        a, avrg_alltime_train, avrg_labels_train = avg_data(wordind, taskind, data_train, adjnoun, ns.params, repnum)
        a, avrg_alltime_test_0, avrg_labels_test_0 = avg_data(wordind, taskind_test[0], data_test_0, adjnoun, ns.params, repnum)
        a, avrg_alltime_test_1, avrg_labels_test_1 = avg_data(wordind, taskind_test[1], data_test_1, adjnoun, ns.params, repnum)

        logging.critical('process numnbr: {}'.format(repnum))
        for timeind, _ in enumerate(ns.t_vec):
            if all_2v2[timeind, taskind-1, wordind] != 0 and  all_2v2[timeind, taskind_test[0] - 1, wordind] != 0 and all_2v2[timeind, taskind_test[1] - 1, wordind]:
                continue
            logging.info('num {}: {} trainTask {} testTask {}, word {} t {:.3f}\t'.format(repnum,
                ns.subj, taskind, taskind_test, wordind, ns.t_vec[timeind]))
            ### selecting data

            ### average data
            # check labels_train va labels_test yeki bashe
            avrg_data_train = avrg_alltime_train[:, :,
                              np.logical_and(np.squeeze(adjnoun['time'] > ns.t_vec[timeind]),
                                             np.squeeze(adjnoun['time']<= (
                                                 ns.t_vec[timeind] + ns.params.time_window)))]
            avrg_data_test_0 = avrg_alltime_test_0[:, :,
                              np.logical_and(np.squeeze(adjnoun['time'] > ns.t_vec[timeind]),
                                             np.squeeze(adjnoun['time']<= (
                                                 ns.t_vec[timeind] + ns.params.time_window)))]
            avrg_data_test_1 = avrg_alltime_test_1[:, :,
                               np.logical_and(np.squeeze(adjnoun['time'] > ns.t_vec[timeind]),
                                              np.squeeze(adjnoun['time'] <= (
                                                      ns.t_vec[timeind] + ns.params.time_window)))]
            ### training
            word_vecs = ns.params.word_vecs[avrg_labels_train.astype(int), :]
            word_vecs = word_vecs[:, ns.params.word_dims]

            all_ests, all_targs, all_ests_test_0, all_targs_test_0, all_ests_test_1, all_targs_test_1  = \
                leave_two_out(avrg_data_train, word_vecs, a, 1, avrg_data_test_0, word_vecs, avrg_data_test_1, word_vecs)

            all_res = make_results(all_ests, all_targs, ns.params.dist_metric)
            all_res_test_0 = make_results(all_ests_test_0, all_targs_test_0, ns.params.dist_metric)
            all_res_test_1 = make_results(all_ests_test_1, all_targs_test_1, ns.params.dist_metric)
            all_2v2[timeind, taskind-1, wordind] = 100 * np.mean(all_res)
            all_2v2[timeind, taskind_test[0] - 1, wordind] = 100 * np.mean(all_res_test_0)
            all_2v2[timeind, taskind_test[1] - 1, wordind] = 100 * np.mean(all_res_test_1)
            logging.critical('2v2 Acc {:.4f}\t 2v2 Acc test {:.4f}\t 2v2 Acc test {:.4f}\t\n{}\n\n'.format(
                                                                                     100 * np.mean(all_res),
                                                                                     100 * np.mean(all_res_test_0),
                                                                                     100 * np.mean(all_res_test_1),
                                                                                     str(datetime.now())))
            if timeind % 5 == 0:
                sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': ns.t_vec, 'params': ns.params,
                     'parser': ns.args, 'repnum': repnum})

    sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': ns.t_vec, 'params': ns.params,
                               'parser': ns.args, 'repnum': repnum})
    return all_2v2


def tgm_do_2v2_train_test_noun_adj(adjnoun, ns, repnum):  # testTaskList
        # TODO: when averaging based on utterance, there's different number of instances
        # TODO: between 1-word and 2word conditions
        # IMPORTANT: assumed the desired channels have been already selected in adjnoun
        # IMPORTANT: labels must be 0-indexed
        # IMPORTANT: trainTaskList and testTaskList are 1-indexed
        weightfolder = '{}/weights/{}/{}'.format(ns.my_out_dir,  ns.params.subjs, repnum)
        if not os.path.exists(weightfolder):
            os.makedirs(weightfolder)

        fileName = '{}/{}/{}_tgm_{}_classify_{}_on_{}_with_{}_averaging.mat'.format(
            ns.my_out_dir, ns.params.subjs, repnum, ns.subj, ns.args.whcond, ns.args.traind, ns.args.avg)
        if os.path.exists(fileName):
            loaded = sio.loadmat(fileName)
            all_2v2 = loaded['all_2v2']

        else:
            all_2v2 = np.zeros((ns.t_vec.shape[0], ns.t_vec.shape[0], 5, 5, 2))  # Note: time, task, word, train,test

        if ns.args.whcond == 'noun_train':
            all_tasks = [5,3,2]
            wordind = 1
            if ns.subj == 'A0056':
                return
        elif ns.args.whcond == 'adj_train':
            all_tasks = [4,3,2]
            wordind = 0
        else:
            return

        for taskind in all_tasks:  # label index (first word spoken, second word spoken)
            taskind_test = all_tasks.copy()
            taskind_test.remove(taskind)
            #if taskind <4:
            #    taskind_test.remove(5-taskind)

            data_train = data_select(taskind, adjnoun, ns.t_vec, ns.params)
            data_test_0 = data_select(taskind_test[0], adjnoun, ns.t_vec, ns.params)
            data_test_1 = data_select(taskind_test[1], adjnoun, ns.t_vec, ns.params)

            ### average data
            # check labels_train va labels_test yeki bashe
            a, avrg_alltime_train, avrg_labels_train = avg_data(wordind, taskind, data_train, adjnoun, ns.params, repnum)
            a, avrg_alltime_test_0, avrg_labels_test_0 = avg_data(wordind, taskind_test[0], data_test_0, adjnoun, ns.params, repnum)
            a, avrg_alltime_test_1, avrg_labels_test_1 = avg_data(wordind, taskind_test[1], data_test_1, adjnoun, ns.params, repnum)

            for traintimeind, _ in enumerate(ns.t_vec):
                logging.info('{} trainTask {} testTask {}, word {} t {:.3f}\t'.format(ns.subj, taskind,
                                                                                      taskind_test, wordind,
                                                                                      ns.t_vec[traintimeind]))
                train_range = range(np.argmax(adjnoun['time']> ns.t_vec[traintimeind]), np.argmax(adjnoun['time'] > ns.t_vec[traintimeind])+99)
                avg_cond_r_time_r = avrg_alltime_train[:, :, train_range]

                for testtimeind, _ in enumerate(ns.t_vec):
                    if all_2v2[traintimeind, testtimeind, taskind - 1, taskind - 1, wordind] != 0 and \
                        all_2v2[traintimeind, testtimeind, taskind - 1, taskind_test[0] - 1, wordind] != 0 and \
                            all_2v2[traintimeind, testtimeind, taskind - 1, taskind_test[1] - 1, wordind] != 0:
                        continue

                    test_range = range(np.argmax(adjnoun['time'] > ns.t_vec[testtimeind]),
                                        np.argmax(adjnoun['time'] > ns.t_vec[testtimeind]) + 99)

                    avg_cond_r_time_t = avrg_alltime_train[:, :, test_range]
                    avg_cond_t_time_t_0 = avrg_alltime_test_0[:, :, test_range]
                    avg_cond_t_time_t_1 = avrg_alltime_test_1[:, :, test_range]


                    ### training
                    word_vecs =  ns.params.word_vecs[avrg_labels_train.astype(int), :]
                    word_vecs = word_vecs[:, ns.params.word_dims]

                    all_ests = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    all_targs = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    p_all_ests = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    p_all_targs = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    l_all_ests = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    l_all_targs = np.zeros((a.shape[0], 2, word_vecs.shape[1]))

                    for j in range(a.shape[0]):
                        folds = np.zeros(word_vecs.shape[0])
                        folds[a[j, :]] = 1
                        modelfile = '{}/wordind_{}_ttaskind_{}_ttimeind_{}_a_{}_repnum_{}.joblib'.format(
                            weightfolder, wordind, taskind, traintimeind,j, repnum)
                        if os.path.exists(modelfile):
                            reg = joblib.load(modelfile)
                        else:
                            reg = VectorRegressor(fZscore=1, folds=folds)
                            ## Train
                            reg.fit(avg_cond_r_time_r, word_vecs)
                            joblib.dump(reg, modelfile)

                        ## predict
                        _, _, scaler = reg.transform(avg_cond_r_time_t, word_vecs)
                        all_ests[j, :, :], all_targs[j, :, :] = reg.predict(avg_cond_r_time_t, word_vecs, scaler)
                        _, _, scaler = reg.transform(avg_cond_t_time_t_0, word_vecs)
                        p_all_ests[j, :, :], p_all_targs[j, :, :] = reg.predict(avg_cond_t_time_t_0, word_vecs, scaler)
                        _, _, scaler = reg.transform(avg_cond_t_time_t_1, word_vecs)
                        l_all_ests[j, :, :], l_all_targs[j, :, :] = reg.predict(avg_cond_t_time_t_1, word_vecs, scaler)


                    all_res = make_results(all_ests, all_targs, ns.params.dist_metric)
                    all_res_test_0 = make_results(p_all_ests, p_all_targs, ns.params.dist_metric)
                    all_res_test_1 = make_results(l_all_ests, l_all_targs, ns.params.dist_metric)
                    all_2v2[traintimeind, testtimeind, taskind - 1, taskind - 1, wordind] = 100 * np.mean(all_res)
                    all_2v2[traintimeind, testtimeind, taskind - 1, taskind_test[0] - 1, wordind] = 100 * np.mean(all_res_test_0)
                    all_2v2[traintimeind, testtimeind, taskind - 1, taskind_test[1] - 1, wordind] = 100 * np.mean(all_res_test_1)


                    logging.critical('2v2 Acc {:.4f}\t 2v2 Acc test {:.4f}\t 2v2 Acc test {:.4f}\t\n | traintime {} ,  testtime {}\n\n'.format(
                        100 * np.mean(all_res),
                        100 * np.mean(all_res_test_0),
                        100 * np.mean(all_res_test_1),
                        ns.t_vec[traintimeind],
                        ns.t_vec[testtimeind]))

                if traintimeind % 5 == 0:
                    sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': ns.t_vec, 'params': ns.params,
                                           'parser':ns.args, 'repnum': repnum})


        sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': ns.t_vec, 'params': ns.params,
                               'parser': ns.args, 'repnum': repnum})
        return all_2v2


def memory_usage_psutil():
    # return the memory usage in MB
    import psutil
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem

def data_select(taskind, adjnoun, t_vec, params):
    data = adjnoun['data'][np.squeeze(adjnoun['task'] == taskind), :, :]
    data = data * 10 ** 12
    logging.info('\n\t#trials: {}\n'.format(data.shape[0]))
    return data


def avg_data(wordind, taskind,data, adjnoun, params, repnum):
    ### averaging trials together
    # forall task == cur_task choose 1st / 2nd word cods
    labels = adjnoun['labels'][wordind, np.squeeze(adjnoun['task'] == taskind)]
    if labels[0] == 255:  # when we have only 1 word and it might be stored in 0 or 1 index
        labels = adjnoun['labels'][:, np.squeeze(adjnoun['task'] == taskind)]
        labels +=1
        labels = np.sum(labels, axis=0)
        labels -=1

    # for every unique word have exactly 4(num_per_inst) instance
    total_num_inst = np.unique(labels).size * params.num_per_inst

    # average trials together so that we have say 20 ecpochs in total (total_num_inst) 5 * 4 = 20
    avrg_data = np.zeros((total_num_inst, data.shape[1], data.shape[2]))
    avrg_labels = np.zeros(total_num_inst)
    if params.avg == 'random':
        # TODO: needs random, cv and the averaging
        np.random.seed(9876)
        avrg_counter = 0
        labs = np.unique(labels)
        for i in range(0, labs.size):
            cur_data = data[np.squeeze(labels == labs[i]), :, :]
            f = KFold(params.num_per_inst, True, np.random.randint(1000))
            for _, fold_ind in f.split(cur_data):
                avrg_data[avrg_counter, :, :] = np.mean(cur_data[fold_ind, :, :], 0)
                avrg_labels[avrg_counter] = labs[i]
                avrg_counter = avrg_counter + 1
        pass
    else:  # TODO: rethink this else
        avrg_counter = 0
        labs = np.unique(labels)
        all_labels = adjnoun['labels'][:, np.squeeze(adjnoun['task'] == taskind)]
        for i in range(0, labs.size):
            cur_data = data[np.squeeze(labels == labs[i]), :, :]
            cur_labels = all_labels[:, np.squeeze(all_labels[wordind, :] == labs[i])]
            uniq_utter = np.unique(cur_labels[1 - wordind, :])  # labesls of the other word
            for j in range(0, uniq_utter.size):
                # % fprintf('%d %d\n', labs(i), uniq_utter(j));
                selected_trials = cur_labels[1 - wordind, :] == uniq_utter[j]
                avrg_data[avrg_counter, :, :] = np.mean(cur_data[selected_trials, :, :], 0)
                avrg_labels[avrg_counter] = labs[i]
                avrg_counter += 1

    ### choosing pairs of distinct labels
    # a is the 2 vs 2 pairs to test on
    a = np.array(list(combinations(range(total_num_inst), 2)))
    keep_vec = np.full(len(a), True)
    # Keep only those 2v2 pairs that have different labels
    for i in range(len(a)):
        if avrg_labels[a[i][0]] == avrg_labels[a[i][1]]:
            keep_vec[i] = False

    a = a[keep_vec, :]

    #NOTE: do shuffling
    np.random.seed(1234)
    for i in range(repnum):
        np.random.shuffle(avrg_labels)
    logging.info('{} : before {}'.format(repnum, avrg_labels))
    np.random.shuffle(avrg_labels)
    logging.info('{} : after {}'.format(repnum, avrg_labels))

    return a, avrg_data, avrg_labels


def make_results(ests, targs, dist_metric):
    # word_model by subjects by combo_method by permutation
    results = np.zeros(ests.shape[0])
    for p in range(ests.shape[0]):
        e = np.squeeze(ests[p,:,:])
        t = np.squeeze(targs[p,:,:])
        if e.shape[0] != t.shape[0] or e.shape[1] != t.shape[1]:
            # print('w {} s %i c %i p %i\n', w, s, c, p);
            print('e {} \n'.format(e.shape))
            print('t %i %i \n'.format(e.shape))

        d = cdist(e, t, dist_metric)
        if d[0, 0] + d[1, 1] < d[0, 1] + d[1, 0]:
            results[p] = 1
        elif d[0, 0] + d[1, 1] == d[0, 1] + d[1, 0]:
            # This happens when the target vectors are the same
            # for both(same adj or same noun and using just
            # one or the other vector)
            results[p] = 0.5

    return results
