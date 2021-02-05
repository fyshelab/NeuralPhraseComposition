from __future__ import division
from itertools import combinations
import numpy as np
import time
import os
import datetime

import logging
from scipy.spatial.distance import cdist
import scipy.io as sio
from sklearn.model_selection import KFold, GridSearchCV
from regressor import VectorRegressor


class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self):
        if self.name:
            print('[%s]' % self.name, end=' ')
        print('Elapsed: %s' % (time.time() - self.tstart))


class Evaluator:
    def __init__(self, t_vec, subj, adjnoun, params, args):
        self.t_vec = t_vec
        self.subj = subj
        self.adjnoun = adjnoun
        self.params = params
        self.args = args

    def leave_two_out(self, X, y, a, foldZscore, Xp=None, yp=None, Xl=None, yl=None):
        from scipy.stats import zscore
        from numpy.matlib import repmat

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
            # print(reg.clf.alpha_)
            ## predict
            all_ests[j, :, :], all_targs[j, :, :] = reg.predict(X, y)
            if Xp is not None:
                _, _, scaler = reg.transform(Xp, yp)
                p_all_ests[j, :, :], p_all_targs[j, :, :] = reg.predict(Xp, yp, scaler)
            if Xl is not None:
                _, _, scaler = reg.transform(Xl, yl)
                l_all_ests[j, :, :], l_all_targs[j, :, :] = reg.predict(Xl, yl, scaler)

        return all_ests, all_targs, p_all_ests, p_all_targs, l_all_ests, l_all_targs

    def do_2v2(self, taskindList):
        # IMPORTANT: assumed the desired channels have been already selected in adjnoun
        # IMPORTANT: labels must be 0-indexed
        # IMPORTANT: taskindList is 1-indexed

        all_2v2 = np.zeros((self.t_vec.shape[0], 5, 2))

        for timeind, _ in enumerate(self.t_vec):

            for taskind in taskindList:  # taskind is chosen based on .mat file and is 1-indexed
                if taskind == 5 and self.subj == 'A0056':  # bad data for this task
                    continue
                for wordind in range(0, 2):  # label index (first word spoken, second word spoken)
                    if taskind > 3 and wordind > 0:
                        break
                    logging.info(
                        '{} Task {} word {} t {:.3f}\t'.format(self.subj, taskind, wordind, self.t_vec[timeind]))

                    ### selecting data
                    data = self.data_select(taskind, timeind)

                    ### average data
                    a, avrg_data, avrg_labels = self.avg_data(wordind, taskind, data)

                    ### training
                    word_vecs = self.params.word_vecs[avrg_labels.astype(int), :]
                    word_vecs = word_vecs[:, self.params.word_dims]
                    all_ests, all_targs, _, _, _, _ = self.leave_two_out(avrg_data, word_vecs, a, 1)
                    all_res = self.make_results(all_ests, all_targs, self.params.dist_metric)
                    all_2v2[timeind, taskind - 1, wordind] = 100 * np.mean(all_res)
                    logging.critical('2v2 Acc {:.4f}\t\n{}\n\n'.format(100 * np.mean(all_res),
                                                                       str(datetime.datetime.now())))
        return all_2v2

    def do_2v2_phrasal(self, taskindList):
        # IMPORTANT: assumed the desired channels have been already selected in adjnoun
        # IMPORTANT: labels must be 0-indexed
        # IMPORTANT: taskindList is 1-indexed

        all_2v2 = np.zeros((self.t_vec.shape[0], 5))
        all_2v2_res = np.zeros((self.t_vec.shape[0], 5, 300))  # [0]*5 # np.zeros((self.t_vec.shape[0], 5))

        for taskind in taskindList:  # taskind is chosen based on .mat file and is 1-indexed
            timepass_res = []
            for timeind, _ in enumerate(self.t_vec):
                if taskind > 3:  # this function only works for two word utterance
                    break
                logging.info(
                    '{} Task {} t {:.3f}\t'.format(self.subj, taskind, self.t_vec[timeind]))

                ### selecting data
                data = self.data_select(taskind, timeind)

                ### average data
                a, avrg_data, word_vecs = self.avg_data(0, taskind, data, self.params)

                ### training
                all_ests, all_targs, _, _, _, _ = self.leave_two_out(avrg_data, word_vecs, a, 1)
                all_res = self.make_results(all_ests, all_targs, self.params.dist_metric)
                all_2v2[timeind, taskind - 1] = 100 * np.mean(all_res)
                all_2v2_res[timeind, taskind, :] = all_res
                logging.critical('2v2 Acc {:.4f}\t\n{}\n\n'.format(100 * np.mean(all_res),
                                                                   str(datetime.datetime.now())))

        return all_2v2, all_2v2_res

    def do_2v2_train_test(self, trainTasklList, wordList):  # testTaskList
        # IMPORTANT: assumed the desired channels have been already selected in adjnoun
        # IMPORTANT: labels must be 0-indexed
        # IMPORTANT: trainTaskList and testTaskList are 1-indexed
        all_2v2 = np.zeros((self.t_vec.shape[0], 5, 2, 2))  # Note: time, task, word, train,test

        for timeind, _ in enumerate(self.t_vec):
            for taskind in trainTasklList:  # taskind is chosen based on .mat file and is 1-indexed
                if taskind == 5 and self.subj == 'A0056':  # bad data for this task
                    continue
                for wordind in wordList:  # label index (first word spoken, second word spoken)
                    if taskind > 3 and wordind > 0:
                        break
                    taskind_test = 5 - taskind
                    logging.critical(
                        'train:{}, test:{}, time:{}, wordind:{}'.format(taskind, taskind_test, self.t_vec[timeind],
                                                                        wordind))
                    # logging.info('{} trainTask {} word {} t {:.3f}\t'.format(self.subj, taskind, wordind, self.t_vec[timeind]))

                    ### selecting data
                    data_train = self.data_select(taskind, timeind)
                    data_test = self.data_select(taskind_test, timeind)

                    ### average data
                    # check labels_train va labels_test yeki bashe
                    a, avrg_data_train, avrg_labels_train = self.avg_data(wordind, taskind, data_train)
                    a, avrg_data_test, avrg_labels_test = self.avg_data(wordind, taskind_test, data_test)
                    ### training
                    word_vecs = self.params.word_vecs[avrg_labels_train.astype(int), :]
                    word_vecs = word_vecs[:, self.params.word_dims]

                    all_ests, all_targs, all_ests_test, all_targs_test, _, _ = \
                        self.leave_two_out(avrg_data_train, word_vecs, a, 1, Xp=avrg_data_test, yp=word_vecs)

                    all_res = self.make_results(all_ests, all_targs, self.params.dist_metric)
                    all_res_test = self.make_results(all_ests_test, all_targs_test, self.params.dist_metric)
                    all_2v2[timeind, taskind - 1, wordind, 0] = 100 * np.mean(all_res)
                    all_2v2[timeind, taskind_test - 1, wordind, 1] = 100 * np.mean(all_res_test)
                    logging.critical('2v2 Acc {:.4f}\t 2v2 Acc test {:.4f}\t\n{}\n\n'.format(100 * np.mean(all_res),
                                                                                             100 * np.mean(
                                                                                                 all_res_test),
                                                                                             str(
                                                                                                 datetime.datetime.now())))
        return all_2v2

    def do_2v2_train_test_noun_adj(self):  # testTaskList
        # TODO: when averaging based on utterance, there's different number of instances
        # TODO: between 1-word and 2word conditions
        # IMPORTANT: assumed the desired channels have been already selected in adjnoun
        # IMPORTANT: labels must be 0-indexed
        # IMPORTANT: trainTaskList and testTaskList are 1-indexed
        all_2v2 = np.zeros((self.t_vec.shape[0], 5, 2))  # Note: time, task, word, train,test

        for wordind in range(0, 2):  # label index (first word spoken, second word spoken)
            if self.args.whcond == 'noun_train':
                taskind = 5
            elif self.args.whcond == 'adj_train':
                taskind = 4
            else:
                return

            ### noun begin
            taskind_test = [2, 3]  # , taskind]  # taskind is chosen based on .mat file and is 1-indexed

            if taskind == 5 and self.subj == 'A0056':  # bad data for this task
                break
            if taskind == 5 and wordind == 0:  # training on noun only but wordind is 0
                continue
            if taskind == 4 and wordind == 1:
                continue
            data_train = self.data_select(taskind)
            data_test_0 = self.data_select(taskind_test[0])
            data_test_1 = self.data_select(taskind_test[1])
            ### average data
            # check labels_train va labels_test yeki bashe
            a, avrg_alltime_train, avrg_labels_train = self.avg_data(wordind, taskind, data_train)
            a, avrg_alltime_test_0, avrg_labels_test_0 = self.avg_data(wordind, taskind_test[0], data_test_0)
            a, avrg_alltime_test_1, avrg_labels_test_1 = self.avg_data(wordind, taskind_test[1], data_test_1)

            for timeind, _ in enumerate(self.t_vec):
                if all_2v2[timeind, taskind - 1, wordind] != 0 and all_2v2[timeind, taskind_test[0] - 1, wordind] != 0\
                        and all_2v2[timeind, taskind_test[1] - 1, wordind]:
                    continue
                logging.info('{} trainTask {} testTask {}, word {} t {:.3f}\t'.format(self.subj, taskind,
                                                                                      taskind_test, wordind,
                                                                                      self.t_vec[timeind]))
                ### selecting data

                ### average data
                # check labels_train va labels_test yeki bashe
                avrg_data_train = avrg_alltime_train[:, :,
                                  np.logical_and(np.squeeze(self.adjnoun['time'] > self.t_vec[timeind]),
                                                 np.squeeze(self.adjnoun['time'] <= (
                                                         self.t_vec[timeind] + self.params.time_window)))]
                avrg_data_test_0 = avrg_alltime_test_0[:, :,
                                   np.logical_and(np.squeeze(self.adjnoun['time'] > self.t_vec[timeind]),
                                                  np.squeeze(self.adjnoun['time'] <= (
                                                          self.t_vec[timeind] + self.params.time_window)))]
                avrg_data_test_1 = avrg_alltime_test_1[:, :,
                                   np.logical_and(np.squeeze(self.adjnoun['time'] > self.t_vec[timeind]),
                                                  np.squeeze(self.adjnoun['time'] <= (
                                                          self.t_vec[timeind] + self.params.time_window)))]
                ### training
                word_vecs = self.params.word_vecs[avrg_labels_train.astype(int), :]
                word_vecs = word_vecs[:, self.params.word_dims]

                all_ests, all_targs, all_ests_test_0, all_targs_test_0, all_ests_test_1, all_targs_test_1 = \
                    self.leave_two_out(avrg_data_train, word_vecs, a, 1, avrg_data_test_0, word_vecs, avrg_data_test_1,
                                       word_vecs)

                all_res = self.make_results(all_ests, all_targs, self.params.dist_metric)
                all_res_test_0 = self.make_results(all_ests_test_0, all_targs_test_0, self.params.dist_metric)
                all_res_test_1 = self.make_results(all_ests_test_1, all_targs_test_1, self.params.dist_metric)
                all_2v2[timeind, taskind - 1, wordind] = 100 * np.mean(all_res)
                all_2v2[timeind, taskind_test[0] - 1, wordind] = 100 * np.mean(all_res_test_0)
                all_2v2[timeind, taskind_test[1] - 1, wordind] = 100 * np.mean(all_res_test_1)
                logging.critical('2v2 Acc {:.4f}\t 2v2 Acc test {:.4f}\t 2v2 Acc test {:.4f}\t\n{}\n\n'.format(
                    100 * np.mean(all_res),
                    100 * np.mean(all_res_test_0),
                    100 * np.mean(all_res_test_1),
                    str(datetime.datetime.now())))
                if timeind % 5 == 0:
                    sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': self.t_vec, 'params': self.params,
                                           'parser': self.args})

        sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': self.t_vec, 'params': self.params,
                               'parser': self.args})
        return all_2v2

    def tgm_do_2v2_train_test_noun_adj(self):  # testTaskList
        # TODO: when averaging based on utterance, there's different number of instances
        # TODO: between 1-word and 2word conditions
        # IMPORTANT: assumed the desired channels have been already selected in adjnoun
        # IMPORTANT: labels must be 0-indexed
        # IMPORTANT: trainTaskList and testTaskList are 1-indexed
        weightfolder = '{}/weights/{}'.format(self.params.my_out_dir, self.params.subjs)
        if not os.path.exists(weightfolder):
            os.makedirs(weightfolder)

        fileName = '{}/{}/tgm_{}_classify_{}_on_{}_with_{}_averaging.mat'.format(
            self.params.my_out_dir, self.params.subjs, self.subj, self.args.whcond, self.args.traind, self.args.avg)
        if os.path.exists(fileName):
            loaded = sio.loadmat(fileName)
            all_2v2 = loaded['all_2v2']

        else:
            all_2v2 = np.zeros(
                (self.t_vec.shape[0], self.t_vec.shape[0], 5, 5, 2))  # Note: time, task, word, train,test

        if self.args.whcond == 'noun_train':
            all_tasks = [5, 3, 2]
            wordind = 1
            if self.subj == 'A0056':
                return
        elif self.args.whcond == 'adj_train':
            all_tasks = [4, 3, 2]
            wordind = 0
        else:
            return

        for taskind in all_tasks:  # label index (first word spoken, second word spoken)
            taskind_test = all_tasks.copy()
            taskind_test.remove(taskind)
            if taskind < 4:
                taskind_test.remove(5 - taskind)

            data_train = self.data_select(taskind)
            data_test_0 = self.data_select(taskind_test[0])
            data_test_1 = self.data_select(taskind_test[1])
            ### average data
            # check labels_train va labels_test yeki bashe
            a, avrg_alltime_train, avrg_labels_train = self.avg_data(wordind, taskind, data_train)
            a, avrg_alltime_test_0, avrg_labels_test_0 = self.avg_data(wordind, taskind_test[0], data_test_0)
            a, avrg_alltime_test_1, avrg_labels_test_1 = self.avg_data(wordind, taskind_test[1], data_test_1)

            for traintimeind, _ in enumerate(self.t_vec):
                logging.info('{} trainTask {} testTask {}, word {} t {:.3f}\t'.format(self.subj, taskind,
                                                                                      taskind_test, wordind,
                                                                                      self.t_vec[traintimeind]))
                train_range = range(np.argmax(self.adjnoun['time'] > self.t_vec[traintimeind]),
                                    np.argmax(self.adjnoun['time'] > self.t_vec[traintimeind]) + 99)
                avg_cond_r_time_r = avrg_alltime_train[:, :, train_range]

                for testtimeind, _ in enumerate(self.t_vec):
                    test_range = range(np.argmax(self.adjnoun['time'] > self.t_vec[testtimeind]),
                                       np.argmax(self.adjnoun['time'] > self.t_vec[testtimeind]) + 99)

                    avg_cond_r_time_t = avrg_alltime_train[:, :, test_range]
                    avg_cond_t_time_t_0 = avrg_alltime_test_0[:, :, test_range]
                    avg_cond_t_time_t_1 = avrg_alltime_test_1[:, :, test_range]

                    ### training
                    word_vecs = self.params.word_vecs[avrg_labels_train.astype(int), :]
                    word_vecs = word_vecs[:, self.params.word_dims]

                    all_ests = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    all_targs = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    p_all_ests = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    p_all_targs = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    l_all_ests = np.zeros((a.shape[0], 2, word_vecs.shape[1]))
                    l_all_targs = np.zeros((a.shape[0], 2, word_vecs.shape[1]))

                    for j in range(a.shape[0]):
                        folds = np.zeros(word_vecs.shape[0])
                        folds[a[j, :]] = 1
                        modelfile = '{}/wordind_{}_ttaskind_{}_ttimeind_{}_a_{}.joblib'.format(
                            weightfolder, wordind, taskind, traintimeind, j)
                        if os.path.exists(modelfile):
                            pass
                            # reg = joblib.load(modelfile)
                        else:
                            reg = VectorRegressor(fZscore=1, folds=folds)
                            ## Train
                            reg.fit(avg_cond_r_time_r, word_vecs)
                            # joblib.dump(reg, modelfile)

                        ## predict
                        _, _, scaler = reg.transform(avg_cond_r_time_t, word_vecs)
                        all_ests[j, :, :], all_targs[j, :, :] = reg.predict(avg_cond_r_time_t, word_vecs, scaler)
                        _, _, scaler = reg.transform(avg_cond_t_time_t_0, word_vecs)
                        p_all_ests[j, :, :], p_all_targs[j, :, :] = reg.predict(avg_cond_t_time_t_0, word_vecs, scaler)
                        _, _, scaler = reg.transform(avg_cond_t_time_t_1, word_vecs)
                        l_all_ests[j, :, :], l_all_targs[j, :, :] = reg.predict(avg_cond_t_time_t_1, word_vecs, scaler)

                    all_res = self.make_results(all_ests, all_targs, self.params.dist_metric)
                    all_res_test_0 = self.make_results(p_all_ests, p_all_targs, self.params.dist_metric)
                    all_res_test_1 = self.make_results(l_all_ests, l_all_targs, self.params.dist_metric)
                    all_2v2[traintimeind, testtimeind, taskind - 1, taskind - 1, wordind] = 100 * np.mean(all_res)
                    all_2v2[traintimeind, testtimeind, taskind - 1, taskind_test[0] - 1, wordind] = 100 * np.mean(
                        all_res_test_0)
                    all_2v2[traintimeind, testtimeind, taskind - 1, taskind_test[1] - 1, wordind] = 100 * np.mean(
                        all_res_test_1)

                    logging.critical(
                        '2v2 Acc {:.4f}\t 2v2 Acc test {:.4f}\t 2v2 Acc test {:.4f}\t\n | traintime {} ,  testtime {}\n\n'.format(
                            100 * np.mean(all_res),
                            100 * np.mean(all_res_test_0),
                            100 * np.mean(all_res_test_1),
                            self.t_vec[traintimeind],
                            self.t_vec[testtimeind]))
                if traintimeind % 5 == 0:
                    sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': self.t_vec, 'params': self.params,
                                           'parser': self.args})

        sio.savemat(fileName, {'all_2v2': all_2v2, 't_vec': self.t_vec, 'params': self.params,
                               'parser': self.args})
        return all_2v2

    def data_select(self, taskind, timeind=None):
        if timeind is None:
            data = self.adjnoun['data'][np.squeeze(self.adjnoun['task'] == taskind), :, :]
            data = data * 10 ** 12
            logging.info('\n\t#trials: {}\n'.format(data.shape[0]))
            return data
        data = self.adjnoun['data'][np.squeeze(self.adjnoun['task'] == taskind), :, :]
        data = data[:, :,
               np.logical_and(np.squeeze(self.adjnoun['time'] > self.t_vec[timeind]),
                              np.squeeze(self.adjnoun['time'] <= (
                                      self.t_vec[timeind] + self.params.time_window)))]  # *10 ^ 12
        data = data * 10 ** 12
        logging.info('\n\t#trials: {}\n'.format(data.shape[0]))
        return data

    def avg_data(self, wordind, taskind, data, params=[]):
        ### averaging trials together
        # forall task == cur_task choose 1st / 2nd word cods
        labels = self.adjnoun['labels'][wordind, np.squeeze(self.adjnoun['task'] == taskind)]
        if labels[0] == 255:  # when we have only 1 word and it might be stored in 0 or 1 index
            labels = self.adjnoun['labels'][:, np.squeeze(self.adjnoun['task'] == taskind)]
            labels += 1
            labels = np.sum(labels, axis=0)
            labels -= 1
        # for every unique word have exactly 4(num_per_inst) instance
        total_num_inst = np.unique(labels).size * self.params.num_per_inst

        # average trials together so that we have say 20 ecpochs in total (total_num_inst) 5 * 4 = 20
        avrg_data = np.zeros((total_num_inst, data.shape[1], data.shape[2]))
        avrg_labels = np.zeros(total_num_inst)
        if self.params.avg == 'random':
            # TODO: needs random, cv and the averaging
            np.random.seed(9876)
            avrg_counter = 0
            labs = np.unique(labels)
            for i in range(0, labs.size):
                cur_data = data[np.squeeze(labels == labs[i]), :, :]
                f = KFold(self.params.num_per_inst, True, np.random.randint(1000))
                for _, fold_ind in f.split(cur_data):
                    avrg_data[avrg_counter, :, :] = np.mean(cur_data[fold_ind, :, :], 0)
                    avrg_labels[avrg_counter] = labs[i]
                    avrg_counter = avrg_counter + 1
            a = self.__make_distinct(avrg_labels, total_num_inst)
        elif self.params.avg == 'utterance':  # TODO: rethink this else
            avrg_counter = 0
            labs = np.unique(labels)
            all_labels = self.adjnoun['labels'][:, np.squeeze(self.adjnoun['task'] == taskind)]
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
            a = self.__make_distinct(avrg_labels, total_num_inst)

        elif self.params.avg == 'mul' and params != []:
            word_dims = params.word_dims
            avrg_labels = np.zeros((total_num_inst, len(word_dims)))
            avrg_counter = 0
            labels = self.adjnoun['labels'][:, np.squeeze(self.adjnoun['task'] == taskind)]
            labs1 = np.unique(labels[0, :])
            labs2 = np.unique(labels[1, :])
            for i, adj in enumerate(labs1):
                for j, noun in enumerate(labs2):
                    selected_trials = np.logical_and(np.squeeze(labels[0, :] == adj),
                                                     np.squeeze(labels[1, :] == noun))
                    cur_data = data[selected_trials, :, :]
                    avrg_data[avrg_counter, :, :] = np.mean(cur_data, 0)

                    avrg_labels[avrg_counter] = params.word_vecs[i, word_dims] * params.word_vecs[j, word_dims]
                    avrg_counter += 1
            a = np.array(list(combinations(range(total_num_inst), 2)))
        else:
            raise ('wrong option of averaging')

        return a, avrg_data, avrg_labels

    def make_results(self, ests, targs, dist_metric):
        # word_model by subjects by combo_method by permutation
        results = np.zeros(ests.shape[0])
        for p in range(ests.shape[0]):
            e = np.squeeze(ests[p, :, :])
            t = np.squeeze(targs[p, :, :])
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

    def __make_distinct(self, avrg_labels, total_num_inst):
        ### choosing pairs of distinct labels
        # a is the 2 vs 2 pairs to test on
        a = np.array(list(combinations(range(total_num_inst), 2)))
        keep_vec = np.full(len(a), True)
        # Keep only those 2v2 pairs that have different labels
        for i in range(len(a)):
            if avrg_labels[a[i][0]] == avrg_labels[a[i][1]]:
                keep_vec[i] = False
        a = a[keep_vec, :]
        return a
