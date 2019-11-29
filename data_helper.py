# -*- coding:utf-8 -*-
import os
import settings
import argparse
import numpy as np
import pandas as pd
from grouping import AptGroup
from datetime import datetime
from database import GinAptQuery, cursor, cnx
from feature import make_feature, optimized_make_feature, FeatureExistsError


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--calc_similarity_apt',
                        action='store_true',
                        help='APT similarity 계산')

    parser.add_argument('--make_dataset',
                        action='store_true',
                        help='데이터셋 생성')

    parser.add_argument('--correlation',
                        action='store_true',
                        help='correlation analysis')

    # feature information
    parser.add_argument('--features',
                        type=list,
                        default=settings.features,
                        help='예측에 필요한 feature (default: setting.py에 있는 features 참조)')

    parser.add_argument('--sale_month_size',
                        type=int,
                        default=settings.sale_month_size,
                        help='예측시 사용될 매물 데이터 크기 (default: setting.py에 있는 sale_month_size 참조)')

    parser.add_argument('--sale_recent_month_size',
                        type=int,
                        default=settings.sale_recent_month_size,
                        help='예측시 사용될 매물 데이터 크기 (default: setting.py에 있는 sale_month_size 참조)')

    parser.add_argument('--trade_month_size',
                        type=int,
                        default=settings.trade_month_size,
                        help='예측시 사용될 매매 데이터 크기 (default: setting.py에 있는 trade_month_size 참조)')

    parser.add_argument('--trade_recent_month_size',
                        type=int,
                        default=settings.trade_recent_month_size,
                        help='예측시 사용될 최근 매매 데이터 크기 (default: setting.py에 있는 trade_recent_month_size 참조)')

    parser.add_argument('--trade_cd',
                        type=str,
                        choices=['t', 'c'],
                        default=settings.trade_cd,
                        help='t : 아파트 매매가격 추정 / r: 아파트 전월세가격 추정')

    # data path information
    parser.add_argument('--similarity_size',
                        type=int,
                        default=settings.similarity_size,
                        help='비슷한 아파트 리스트 출력 갯수 (default: setting.py에 있는 similarity_size 참조')

    parser.add_argument('--save_path',
                        type=str,
                        default=settings.save_path,
                        help='DATASET PATH (default: setting.py에 있는 save_path 참조)')

    parser.add_argument('--correlation_path',
                        type=str,
                        default=settings.correlation_path,
                        help='correlation analysis result DATA PATH (default: setting.py에 있는 correlation_path 참조)')

    parser.add_argument('--label_name',
                        type=str,
                        default=settings.label_name,
                        help='DATASET label name (default: setting.py에 있는 label_name 참조)')

    return parser.parse_args()


def make_apt_similarity_dataset(argument):
    pk_list = [pk for pk in GinAptQuery.get_predicate_apt_list().fetchall()]

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
    pk_list = [pk for pk in query.get_predicate_apt_list().fetchall()]

    total_data = {
        settings.full_feature_model_name: [],
        settings.sale_feature_model_name: [],
        settings.trade_feature_model_name: []
    }
    for i, (apt_master_pk, apt_detail_pk) in enumerate(pk_list):
        try:
            print('i : {} \t apt detail pk : {}'.format(i, apt_detail_pk))
            for trade_pk, _, year, mon, day, floor, extent, price in \
                    query.get_trade_price(apt_detail_pk, argument.trade_cd).fetchall():
                trg_date = '{0}-{1:02d}-{2:02d}'.format(year, int(mon), int(day))
                trg_date = datetime.strptime(trg_date, '%Y-%m-%d')
                price = float(price / extent)

                try:
                    feature = optimized_make_feature(feature_name_list=argument.features, apt_master_pk=apt_master_pk,
                                                     apt_detail_pk=apt_detail_pk, trade_cd=argument.trade_cd,
                                                     trg_date=trg_date, sale_month_size=argument.sale_month_size,
                                                     sale_recent_month_size=argument.sale_recent_month_size,
                                                     trade_month_size=argument.trade_month_size,
                                                     trade_recent_month_size=argument.trade_recent_month_size,
                                                     floor=floor, extent=extent, trade_pk=trade_pk)

                    # feature = make_feature(feature_name_list=argument.features, apt_master_pk=apt_master_pk,
                    #                        apt_detail_pk=apt_detail_pk, trade_cd=argument.trade_cd, trg_date=trg_date,
                    #                        sale_month_size=argument.sale_month_size,
                    #                        sale_recent_month_size=argument.sale_recent_month_size,
                    #                        trade_month_size=argument.trade_month_size,
                    #                        trade_recent_month_size=argument.trade_recent_month_size, floor=floor,
                    #                        extent=extent, trade_pk=trade_pk)
                except FeatureExistsError:
                    # 매매 혹은 매물 데이터를 바탕으로한 feature 하나도 존재하지 않을때...
                    continue
                except Exception as e:
                    print(e)
                    continue

                feature_df = feature['data']
                feature_df['apt_detail_pk'] = apt_detail_pk
                feature_df[argument.label_name] = price

                status = feature['status']
                total_data[status].append(feature_df)
        except KeyboardInterrupt:
            # If KeyboardInterrupt file save...
            break

    # file save...
    print('file saving...')
    for status, feature_df in total_data.items():
        if len(feature_df) == 0:
            continue
        filename = 'apt_dataset_{}.csv'.format(status)
        filepath = os.path.join(argument.save_path, filename)
        print('{} saving start'.format(filepath))
        total_df = pd.concat(feature_df)
        total_df = total_df.reset_index(drop=True)
        total_df.to_csv(filepath, index=False)
        print('{} saving complete'.format(filepath))


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
