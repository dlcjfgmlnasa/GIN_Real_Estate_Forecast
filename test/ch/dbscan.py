# -*- coding:utf-8 -*-
"""
DBSCAN(Density-Based Spatial Clustering of Applications with Noise)
이름에서 알 수 있듯이 데이터가 위치하고 있는 공간 밀접도를 기준으로 클러스터를 구분한다.
자기를 중심으로 반지름 R의 공간에 최소 M개의 포인트가 존재하는 점을 코어 포인트(core point) 라고 부른다.
코어 포인트는 아니지만 반지름 R 안에 다른 코어 포인트가 있을 경우 경계 포인트(border point) 라고 한다.
코어 포인트도 아니고 경계 포인트에도 속하지 않는 점을 Noise(또한 outlier) 라고 분류한다.

하나의 클러스터는 반지름 R안에 서로 위치하는 모든 코어 포인트를 포함하는 방식으로 구성된다.
당현히 각 코어 포인트 주위에 있는 경계 포인트를 포함한다. 서로 밀접한 데이터끼지 하나의 클러스터를
구성하게 되고 어느 클러스터에도 속하지 않는 점들을 Noise 로 남게 된다.
"""

import os
import pandas as pd
from database import GinQuery
from sklearn import preprocessing
from sklearn import cluster
import matplotlib.pyplot as plt
from settings import cnx
import numpy as np
import folium

file_path = './apt.csv'


def make_data_set(size=1000):
    total_lines = []
    for idx in range(size):
        cursor = GinQuery.test(idx)
        result = cursor.fetchall()
        total_lines.extend(result)
    df = pd.DataFrame(total_lines)
    df.columns = [
        'pk_apt_master', '단지명', '시군구 코드', '법정동 코드', '대치위치',
        '동정보', '총 세대수', '총 동수', '최고층', '총 주차대수', '준공일', '위도', '경도',
        'pk_apt_detail', '공급면적', '전용면적', '면적별 세대수', '읍면동 이름', '분양가격'
    ]
    df.index = df.pk_apt_detail
    df.to_csv(file_path)


def data_pre_processing(df):
    label_encoder = preprocessing.LabelEncoder()
    columns = [
        'pk_apt_master', 'pk_apt_detail', '단지명', '시군구 코드', '법정동 코드', '총 세대수', '총 동수',
        '최고층', '준공일', '위도', '경도', '공급면적', '전용면적', '면적별 세대수'
    ]
    t_df = df[columns]

    train_columns = [
        '위도', '경도'
    ]
    train_df = df[train_columns]
    # train_df['dangi_cd'] = label_encoder.fit_transform(train_df['단지명'])
    # train_df = train_df.drop('단지명', axis=1)
    dbm = cluster.DBSCAN(eps=0.1, min_samples=20)
    dbm.fit(train_df)

    cluster_label = dbm.labels_
    t_df['Cluster'] = cluster_label
    grouped = t_df.groupby('Cluster')

    for key, value in grouped:
        import random
        apt_map = folium.Map(location=[37.5102, 126.982], zoom_start=12)
        number_of_colors = 1

        color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                 for i in range(number_of_colors)][0]
        for x, y in zip(value['위도'], value['경도']):
            x = float(x)
            y = float(y)
            folium.RegularPolygonMarker((x, y),
                                        radius=10,
                                        color=color).add_to(apt_map)
        print(value)
        value.to_csv('apt-{}.csv'.format(key))
        apt_map.save('apt-{}.html'.format(key))

    # value.to_csv('test.csv')
    #
    # p_items = []
    # for pk in value.pk_apt_detail:
    #     cursor = GinQuery.get_actual_price(idx=str(pk))
    #     df = pd.read_sql(cursor.statement, cnx)
    #     x_lines = [float(i + '.' + j) for i, j in zip(df['deal_ymd'], df['day'])]
    #     y_lines = df['t_amt']
    #     p = plt.scatter(x_lines, y_lines)
    #     p_items.append(p)
    # plt.legend(p_items, value.pk_apt_detail)
    # plt.show()


if __name__ == '__main__':
    make_data_set()
    frame = pd.read_csv(file_path)
    data_pre_processing(frame)