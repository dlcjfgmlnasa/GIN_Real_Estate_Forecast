# -*- coding:utf-8 -*-
from sklearn import svm


def model(features, label):
    # SVM model
    m = svm.SVR(kernel='linear')
    m.fit(X=features, y=label)
    return m
