# -*- coding:utf-8 -*-
import os
import pickle
import argparse
import pandas as pd
from data_helper import label_name
from sklearn.externals import joblib
from model import linear_regression, svm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.join('./dataset/', 'sale_price.csv'))
    parser.add_argument('--features', type=list,
                        default=[
                            'sale_price_with_floor',
                            'sale_price_with_floor_recent',
                            'sale_price_with_floor_group',
                            'sale_price_with_floor_group_recent',
                            'sale_price_with_complex_group',
                            'sale_price_with_complex_group_recent'
                        ])
    parser.add_argument('--model', type=str,  default='linear_regression',
                        choices=[
                            'linear_regression',
                            'svm',
                            'dnn'
                        ])
    parser.add_argument('--save_path', type=str, default=os.path.join('./model', 'store'))
    parser.add_argument('--model_name', type=str, default='linear_regression.model')
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
    m = train(df[args.features], df[label_name], args.model)
    filename = os.path.join(args.save_path, args.model_name)
    joblib.dump(m, filename)

