# -*- coding:utf-8 -*-
from sklearn import linear_model


def model(features, label):
    # Linear Regression model
    m = linear_model.LinearRegression()
    m.fit(X=features, y=label)
    return m
