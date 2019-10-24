# -*- coding:utf-8 -*-
import settings
import argparse
import numpy as np
import pandas as pd
from grouping import AptGroup
from datetime import datetime
from database import GinAptQuery, cursor, cnx
from feature import make_feature


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calc_similarity_apt', action='store_true')
    parser.add_argument('--make_dataset', action='store_true')
    parser.add_argument('--correlation', action='store_true')

    parser.add_argument('--features', type=list, default=settings.features)
    parser.add_argument('--sale_month_size', type=int, default=settings.sale_month_size)
    parser.add_argument('--sale_recent_month_size', type=int, default=settings.sale_recent_month_size)
    parser.add_argument('--trade_month_size', type=int, default=settings.trade_month_size)
    parser.add_argument('--trade_recent_month_size', type=int, default=settings.trade_recent_month_size)

    parser.add_argument('--similarity_size', type=int, default=settings.similarity_size)

    parser.add_argument('--save_path', type=str, default=settings.save_path)
    parser.add_argument('--correlation_path', type=str, default=settings.correlation_path)
    parser.add_argument('--trade_cd', type=str, choices=['t', 'c'], default=settings.trade_cd)
    parser.add_argument('--label_name', type=str, default=settings.label_name)
    parser.add_argument('--dataset_pk_size', type=int, default=settings.dataset_pk_size)
    return parser.parse_args()


def make_apt_similarity_dataset(argument):
    # pk_list = [pk for pk in query.get_apt_detail_list(argument.dataset_pk_size).fetchall()]
    pk_list = [(1, 1)]

    group = AptGroup()
    for i, (_, apt_detail_pk) in enumerate(pk_list):
        print('i : {} \t apt detail pk : {}'.format(i, apt_detail_pk))
        similarity_ranking_detail_pk = group.apt_apt_similarity(
            apt_detail_pk=apt_detail_pk,
            trade_cd=argument.trade_cd,
            limit_size=argument.similarity_size
        )
        similarity_str = ','.join([str(similarity_value) for similarity_value in similarity_ranking_detail_pk])

        # Update DB
        cursor.execute("""
            INSERT INTO apt_similarity (pk_apt_detail, similarity)
            VALUES (%s, %s);
        """, params=(apt_detail_pk, similarity_str, ))
        cnx.commit()


def make_dataset(argument):
    query = GinAptQuery()
    pk_list = [pk for pk in query.get_apt_detail_list(argument.dataset_pk_size).fetchall()]

    total_data = []
    for i, (apt_master_pk, apt_detail_pk) in enumerate(pk_list):
        try:
            print('i : {} \t apt detail pk : {}'.format(i, apt_detail_pk))
            for trade_pk, _, year, mon, day, floor, extent, price in \
                    query.get_trade_price(apt_detail_pk, argument.trade_cd).fetchall():
                trg_date = '{0}-{1:02d}-{2:02d}'.format(year, int(mon), int(day))
                trg_date = datetime.strptime(trg_date, '%Y-%m-%d')
                price = float(price / extent)

                feature_df = make_feature(feature_name_list=argument.features, apt_master_pk=apt_master_pk,
                                          apt_detail_pk=apt_detail_pk, trade_cd=argument.trade_cd, trg_date=trg_date,
                                          sale_month_size=argument.sale_month_size,
                                          sale_recent_month_size=argument.sale_recent_month_size,
                                          trade_month_size=argument.trade_month_size,
                                          trade_recent_month_size=argument.trade_recent_month_size, floor=floor,
                                          extent=extent, trade_pk=trade_pk)

                if feature_df is not None:
                    # if feature_df in NaN Ignore...
                    if feature_df.isna().any().any():
                        continue

                    feature_df['apt_detail_pk'] = apt_detail_pk
                    feature_df[argument.label_name] = price
                    total_data.append(feature_df)
        except KeyboardInterrupt:
            # file save
            break
        except Exception as e:
            print(e)
            continue

    # file save
    total_df = pd.concat(total_data)
    total_df = total_df.reset_index(drop=True)
    total_df.to_csv(argument.save_path, index=False)


def correlation_analysis(argument):
    df = pd.read_csv(argument.save_path)
    label_name = argument.label_name

    # making feature
    columns = list(df.columns)
    columns.remove(label_name)
    columns.remove('apt_detail_pk')

    correlation_list = []
    for feature_name in columns:
        feature_value = df[feature_name].values
        label_value = df[label_name].values
        size = np.size(feature_value)

        # Calculation correlation ...
        a = feature_value - (np.ones(size) * np.mean(feature_value))
        b = label_value - (np.ones(size) * np.mean(label_value))
        s = sum(a * b)
        p = np.std(feature_value) * np.std(label_value)
        corr = s / (size * p)
        correlation_list.append(corr)

    correlation_df = pd.DataFrame({
        'feature': columns,
        'corr_value': correlation_list
    })
    print(correlation_df)

    # Saving...
    correlation_df.to_csv(argument.correlation_path, index=False)


if __name__ == '__main__':
    args = get_args()

    # making dataset
    if args.make_dataset:
        make_dataset(args)

    # calculation correlation
    if args.correlation:
        correlation_analysis(args)

    # calculation similarity apt
    if args.calc_similarity_apt:
        make_apt_similarity_dataset(args)
