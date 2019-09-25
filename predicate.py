# -*- coding:utf-8 -*-
import os
import settings
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from datedelta import datedelta
from database import GinAptQuery
from feature import make_feature
from sklearn.externals import joblib


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apt_detail_pk', type=int, default=2)
    parser.add_argument('--features', type=list, default=settings.features)
    parser.add_argument('--sale_month_size', type=int, default=settings.sale_month_size)
    parser.add_argument('--sale_recent_month_size', type=int, default=settings.sale_recent_month_size)
    parser.add_argument('--trade_month_size', type=int, default=settings.trade_month_size)
    parser.add_argument('--trade_recent_month_size', type=int, default=settings.trade_month_size)

    parser.add_argument('--model_path', type=str, default=settings.model_path)
    parser.add_argument('--date', type=str, default=settings.current_date)
    parser.add_argument('--previous_month_size', type=int, default=settings.predicate_previous_month_size)
    parser.add_argument('--trade_cd', type=str, choices=['t', 'r'], default=settings.trade_cd)
    return parser.parse_args()


def get_month_range(trg_date: datetime, previous_month_size: int):
    time_format = '%Y-%m-%d'
    end_date = datetime.strptime(trg_date, time_format)
    start_date = end_date - datedelta(months=previous_month_size)
    date_range = pd.date_range(start=start_date, end=end_date)
    date_range = [date.strftime(time_format) for date in date_range]
    return date_range


def main(argument):
    # 현재 시장에 나와있는 아직 팔리지 않는 매물들을 바탕으로 예측을 실시
    features = argument.features

    month_range = get_month_range(argument.date, argument.previous_month_size)

    new_trade_list = GinAptQuery.get_new_apt_trade_list(
            apt_detail_pk=argument.apt_detail_pk,
            date_range='","'.join(month_range),
            trade_cd=argument.trade_cd
    ).fetchall()

    # making feature...
    total_feature_df = []
    apt_extent = None
    for apt_master_pk, apt_detail_pk, date, floor, extent, price in new_trade_list:
        apt_extent = float(extent)
        feature_df = make_feature(feature_name_list=features, apt_master_pk=apt_master_pk, apt_detail_pk=apt_detail_pk,
                                  trade_cd=argument.trade_cd, trg_date=date,
                                  sale_month_size=argument.sale_month_size,
                                  sale_recent_month_size=argument.sale_recent_month_size,
                                  trade_month_size=argument.trade_month_size,
                                  trade_recent_month_size=argument.trade_recent_month_size, floor=floor, extent=extent)
        total_feature_df.append(feature_df)
    total_feature_df = pd.concat(total_feature_df).reset_index(drop=True)
    total_feature_df = total_feature_df.astype(np.float)

    # Load model...
    model = joblib.load(argument.model_path)

    # Predication...
    predication = model.predict(total_feature_df) * apt_extent
    total_feature_df['predication'] = predication

    # Predication max, Predicate min, Predicate avg, Predicate avg(+6%), Predicate avg(-6%)
    result = {
        'predicate_price_max': np.max(predication),
        'predicate_price_min': np.min(predication),
        'predicate_price_avg': np.average(predication),
        'predicate_price_avg_+6%': np.average(predication) + (np.average(predication) * 0.06),
        'predicate_price_avg_-6%': np.average(predication) - (np.average(predication) * 0.06)
    }
    print(result)
    return result


if __name__ == '__main__':
    args = get_args()
    main(args)
