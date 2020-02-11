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
    parser.add_argument('--model', type=str,  default=settings.model_type,
                        choices=[
                            'linear_regression',
                            'svm',
                            'dnn'
                        ])
    parser.add_argument('--save_path', type=str, default=os.path.join('./model', 'store'))
    parser.add_argument('--model_path', type=str, default=settings.model_path)
    parser.add_argument('--label_name', type=str, default=settings.label_name)
    parser.add_argument('--trade_cd', type=str, choices=['t', 'd'], default=settings.trade_cd)
    return parser.parse_args()


def train(feature_df: pd.DataFrame, label_df: pd.DataFrame, model: str):
    if model == 'linear_regression':
        return linear_regression.model(feature_df, label_df)
    elif model == 'svm':
        return svm.model(feature_df, label_df)
    elif model == 'dnn':
        pass


def main(arguments):
    for model_name in [settings.full_feature_model_name, settings.trade_feature_model_name,
                       settings.sale_feature_model_name]:
        try:
            filename = 'apt_dataset_{}.csv'.format(model_name)
            filepath = os.path.join(arguments.dataset_path, filename)
            df = pd.read_csv(filepath)
            features = settings.features_info[model_name]

            m = train(df[features], df[arguments.label_name], arguments.model)
            os.makedirs(arguments.model_path, exist_ok=True)
            model_path = os.path.join(arguments.model_path, model_name+'.model')
            joblib.dump(m, model_path)
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    args = get_args()
    ## update model_info according to trade_cd
    full_feature_model_name = 'full'
    sale_feature_model_name = 'sale'
    trade_feature_model_name = 'trade'
    if args.trade_cd == 't':
        args.model_path = os.path.join('./model', 'store')
        args.dataset_path = './dataset'
        args.model_info = {
            full_feature_model_name: os.path.join(args.model_path, full_feature_model_name + '.model'),
            sale_feature_model_name: os.path.join(args.model_path, sale_feature_model_name + '.model'),
            trade_feature_model_name: os.path.join(args.model_path, trade_feature_model_name + '.model')
        }
    else:
        args.model_path = os.path.join('./model', 'store_rent')
        args.dataset_path = './dataset_rent'
        args.model_info = {
            full_feature_model_name: os.path.join(args.model_path, full_feature_model_name + '.model'),
            sale_feature_model_name: os.path.join(args.model_path, sale_feature_model_name + '.model'),
            trade_feature_model_name: os.path.join(args.model_path, trade_feature_model_name + '.model')
        }

    main(args)

