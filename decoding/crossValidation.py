#!user/bin/env/ python3
""" cross validation

does a k fold cross validation in general
    (I use it to do the whole 2vs2 test though)
"""


__author__ = 'Maryam Honari'

import numpy as np
from numpy.matlib import repmat
from scipy.stats import zscore
from ridgeReg import RidgeReg
import logging

class CrossVal:
    def __init__(self):

        self.folds = np.array([])
        self.num_sem = np.array([])
        # sem_scaler = np.array([])
        # sem_sigma = np.array([])
        self.sem_scaler = {} #StandardScaler()
        # data_mu = np.array([])
        # data_sigma = np.array([])
        self.data_scaler = {} #StandardScaler()
        self.zscore_folds = np.array([])
        self.dist_metric = np.array([])
        self.ridgreg = RidgeReg()

    def train(self, data, sem_matrix, folds, numFolds,
               numWords, save_file, zscore_folds, dist_metric):
        # logging.info('d: {}\n'.format(data[0, 0, :10]))
        data = np.swapaxes(data,1,2)
        data = zscore(np.reshape(data,(data.shape[0], -1)), ddof=1)  # seems this reshape does not work
        self.num_sem = sem_matrix.shape[1]  # length of semantic vector
        self.folds = folds
        self.zscore_folds = zscore_folds
        self.dist_metric = dist_metric
        num_in_fold = sum(folds == 1)
        cur_sem_train = sem_matrix[folds!=1, :]
        cur_data_train = data[folds!=1,:]
        if zscore_folds == 1:

            self.data_scaler['mean'] = np.mean(cur_data_train,axis=0)
            self.data_scaler['std'] = np.std(cur_data_train, axis=0, ddof=1)
            cur_data_train = zscore(cur_data_train,ddof=1)
            self.sem_scaler['mean'] = np.mean(cur_sem_train, axis=0)
            self.sem_scaler['std'] =np.std(cur_sem_train, axis=0, ddof=1)
            cur_sem_train = zscore(cur_sem_train, ddof=1)
        # logging.info('d: {}\n'.format(cur_data_train[0,:10]))
        self.ridgreg.train(cur_data_train, cur_sem_train)
        pass

    def test(self, data, sem_matrix, numWords):
        data = np.swapaxes(data, 1, 2)
        data = zscore(np.reshape(data, (data.shape[0], -1)), ddof=1)
        targets = np.zeros(sem_matrix.shape)
        ests = np.zeros((numWords, self.num_sem))
        cur_sem_test = sem_matrix[self.folds == 1, :]
        cur_data_test = data[self.folds == 1, :]
        num_in_fold = sum(self.folds == 1)
        if self.zscore_folds == 1:
            # cur_sem_test = self.sem_scaler.transform(cur_sem_test)
            # cur_data_test = self.data_scaler.transform(cur_data_test)
            cur_data_test = (cur_data_test - repmat(self.data_scaler['mean'], num_in_fold, 1)) / repmat(self.data_scaler['std'], num_in_fold, 1)
            cur_sem_test = (cur_sem_test - repmat(self.sem_scaler['mean'], num_in_fold, 1)) / repmat(self.sem_scaler['std'], num_in_fold, 1)
        targets[self.folds == 1,:] = cur_sem_test
        testWordNums = np.where(self.folds == 1)[0]
        # append a vector of ones for the biases

        for cur_word in range(num_in_fold):
            targetEstimate = self.ridgreg.predict(cur_data_test[cur_word, :])
            ests[testWordNums[cur_word],:] = targetEstimate

        return ests, targets
