#!user/bin/env/ python3
""" ridge regression
with svd
    (all alphas are trained at the same time and we choose the best after all)


"""

__author__ = 'Maryam Honari'

import numpy as np
from scipy.spatial.distance import cosine
import logging


class RidgeReg:

    def __init__(self):
        self.weightMat = None
        self.regularizer = None

        # self.weightMat = np.array([1]*10) # to be deleted
        # self.regularizer = np.array([1] * 20) # to be deleted

    def train(self, x_train_in, y_train_in,
              reg_params=np.array([0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 0.5,
                                   1, 5, 10])):
        """ x_train , y_train would be numpy array"""
        x_train = np.array(x_train_in, dtype=np.float32)
        x_train = np.append(x_train, np.ones((x_train.shape[0], 1), dtype=np.float64), axis=1)

        n_features = x_train.shape[1]
        n_instances = x_train.shape[0]
        n_labels = y_train_in.shape[1]
        n_params = len(reg_params)

        K0 = np.dot(x_train, x_train.transpose())

        [U, D, V] = np.linalg.svd(K0)
        D = np.diag(D)
        V = np.transpose(V)

        CVerr = np.zeros((n_params, n_labels))

        for i in range(len(reg_params)):
            reg = reg_params[i]

            dlambda = D + np.dot(reg, np.identity(D.shape[0]))
            dlambdaInv = np.diag(1 / np.diag(dlambda))
            klambdaInv = np.dot(np.dot(V, dlambdaInv), np.transpose(U))

            KP = np.dot(np.transpose(x_train), klambdaInv)

            S = np.dot(x_train, KP)
            weightMatrix = np.dot(KP, y_train)
            tmp = np.reshape(1 - np.diag(S), (n_instances, 1))
            Snorm = np.tile(tmp, (1, n_labels))

            y_pred = np.dot(x_train, weightMatrix)
            YdifMat = (y_train - y_pred)

            YdifMat = YdifMat / Snorm

            CVerr[i, :] = (1.0 / n_instances) * np.sum(np.multiply(YdifMat, YdifMat), axis=0)

        minIndex = np.argmin(CVerr, axis=0)
        self.regularizer = np.zeros(n_labels)
        self.weightMat = np.zeros((n_features, n_labels))
        for t in range(n_labels):
            best_reg = reg_params[minIndex[t]]
            self.regularizer[t] = best_reg

            dlambda = D + np.dot(best_reg, np.identity(D.shape[0]))
            dlambdaInv = np.diag(1 / np.diag(dlambda))
            klambdaInv = np.dot(np.dot(V, dlambdaInv), np.transpose(U))

            self.weightMat[:, t] = np.dot(np.transpose(x_train), np.dot(klambdaInv, y_train[:, t]))
        pass

    def test(self, x_test, y_test):
        """returns wheater the predictions are successful or not
        returns True(success) if straight distance of pred. and truth is less than cross distance
        """
        # add bias column to data mat
        x_test = np.append(x_test, np.ones((x_test.shape[0], 1)), axis=1)
        y_pred = np.dot(x_test, self.weightMat)

        straight = cosine(y_pred[0], y_test[0]) + cosine(y_pred[1], y_test[1])
        cross = cosine(y_pred[0], y_test[1]) + cosine(y_pred[1], y_test[0])
        return y_pred, straight < cross

    def predict(self, x_test):
        # add bias column to data mat
        # x_test = np.append(x_test, np.ones((x_test.shape[0], 1)), axis=1)
        logging.info('\ntestX{}'.format(x_test[:10]))
        x_test = np.append(x_test, np.ones(1))
        y_pred = np.dot(x_test, self.weightMat)
        # logging.info('\ntesty{}\n'.format(y_pred[:10]))
        return y_pred


if __name__ == '__main__':
    model = RidgeReg()
    x_train = np.random.rand(5, 10)
    y_train = np.random.rand(5, 2)
    model.train(x_train, y_train)
