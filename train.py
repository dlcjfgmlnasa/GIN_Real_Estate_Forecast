# -*- coding:utf-8 -*-
import os
import settings
import argparse
import pandas as pd
from sklearn.externals import joblib
from model import linear_regression, svm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default=settings.save_path)
    parser.add_argument('--features', type=list, default=settings.features)
    parser.add_argument('--model', type=str,  default=settings.model_type,
                        choices=[
                            'linear_regression',
                            'svm',
                            'dnn'
                        ])
    parser.add_argument('--save_path', type=str, default=os.path.join('./model', 'store'))
    parser.add_argument('--model_path', type=str, default=settings.model_path)
    parser.add_argument('--label_name', type=str, default=settings.label_name)
    return parser.parse_args()


def train(feature_df: pd.DataFrame, label_df: pd.DataFrame, model: str):
    if model == 'linear_regression':
        return linear_regression.model(feature_df, label_df)
    elif model == 'svm':
        return svm.model(feature_df, label_df)
    elif model == 'dnn':
        pass


if __name__ == '__main__':
    args = get_args()
    df = pd.read_csv(args.dataset_path)
    m = train(df[args.features], df[args.label_name], args.model)
    joblib.dump(m, args.model_path)

