#!/usr/bin/env python

""" Vector Regression Model """

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import RidgeCV
from scipy.stats import zscore
from numpy.matlib import repmat


class VectorRegressor(BaseEstimator):

    def __init__(self, fZscore, folds, alpha=[0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1]):
        self.alpha = alpha
        self.count = 0
        self.fZscore = fZscore
        self.folds = folds
        self.clf = None

    def fit(self, X, y=None):
        # zscore x matrix
        X = np.swapaxes(X, 1, 2)
        X = zscore(np.reshape(X, (X.shape[0], -1)), ddof=1)  # zscore at begining

        X = X[self.folds != 1, :]
        y = y[self.folds != 1, :]
        if self.fZscore:
            ## zscore inside folds
            self.scaler = Zscorer()
            X, y = self.scaler.fit_transform(X, y)
        self.clf = RidgeCV(alphas=self.alpha, gcv_mode='svd', store_cv_values=True)
        # logging.info('x: {} \ny:{}\n'.format(X[0,:10], y[0, :10]))
        self.clf.fit(X, y)
        return self

    def transform(self, X, y):
        X = np.swapaxes(X, 1, 2)
        X = zscore(np.reshape(X, (X.shape[0], -1)), ddof=1)  # zscore at begining

        X = X[self.folds != 1, :]
        y = y[self.folds != 1, :]
        scaler = None
        if self.fZscore:
            ## zscore inside folds
            scaler = Zscorer()
            X, y = scaler.fit_transform(X, y)
        return X, y, scaler

    def predict(self, X, y, scaler=None):
        scaler = self.scaler if scaler is None else scaler
        X = np.swapaxes(X, 1, 2)
        X = zscore(np.reshape(X, (X.shape[0], -1)), ddof=1)  # zscore at begining
        X = X[self.folds == 1, :]
        y = y[self.folds == 1, :]
        if self.fZscore:
            X, y = scaler.transform(X, y)

        # logging.info('tesx:{}\n'.format(X[0,:10]))
        y_pred = self.clf.predict(X)
        # logging.info('\ntesy:{}'.format(y_pred[:10]))
        return y_pred, y


# Refactor: move to utility
class Zscorer(BaseEstimator):
    def __init__(self):
        self.X_scaler = {}
        self.y_scaler = {}
        pass

    def fit(self, X, y):
        self.X_scaler['mean'] = np.mean(X, axis=0)
        self.X_scaler['std'] = np.std(X, axis=0, ddof=1)
        self.y_scaler['mean'] = np.mean(y, axis=0)
        self.y_scaler['std'] = np.std(y, axis=0, ddof=1)

    def fit_transform(self, X, y):
        self.fit(X, y)
        X = zscore(X, ddof=1)
        y = zscore(y, ddof=1)
        return X, y

    def transform(self, X, y):
        X = (X - repmat(self.X_scaler['mean'], X.shape[0], 1)) / repmat(
            self.X_scaler['std'], X.shape[0], 1)
        y = (y - repmat(self.y_scaler['mean'], y.shape[0], 1)) / repmat(
            self.y_scaler['std'], y.shape[0], 1)
        return X, y
