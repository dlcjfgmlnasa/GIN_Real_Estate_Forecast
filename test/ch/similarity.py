# -*- coding:utf-8 -*-
import sys; sys.path.append('.'); sys.path.append('..'); sys.path.append('...')
import folium
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
from database import GinQuery
import random
from sklearn import preprocessing
import math
import collections
import matplotlib.pyplot as plt

file_path = 'apt.csv'
file_location_path = 'apt_location.csv'


def make_matrix():
    cursor = GinQuery.test2()
    result = cursor.fetchall()
    df = pd.DataFrame(result)
    columns = [
        'pk_apt_master',        # 지인 master 아파트키
        'edit_bldg_nm',         # 단지명
        'sigungu_cd',           # 시군구 코드
        'dong_cd',              # 법정동 코드
        'location_site',        # 대지위치
        'apt_dong_nm',          # 동정보
        'total_num_of_family',  # 총 세대수
        'total_dong_cnt',       # 총 동수
        'max_jisang_floor',     # 최고층
        'total_jucha',          # 총 주차대수
        'search_dt',            # 준공일
        'latlngx',              # 위도
        'latlngy',              # 경도
        'pk_apt_detail',        # 지인 detail 아파트 key
        'supply_extent',        # 공급면적
        'extent',               # 전용면적
        'num_of_family',        # 면적별 세대수
        'apt_dong',             # 읍면동 이름
        'price',                # 분양가격
    ]
    df.columns = columns
    # df = df[:1000]
    df.to_csv(file_path)


def choice_columns():
    choice_col = [
        'sigungu_cd',           # 시군구 코드
        'dong_cd',              # 법정동 코드
        'latlngx',              # 위도
        'latlngy',              # 경도
    ]
    df = pd.read_csv(file_path)
    df = df.drop('Unnamed: 0', axis=1)[choice_col]
    df.to_csv(file_location_path)


def get_similarity_time_series(source_pk, similarity_list):
    def rmse(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    similarity_list.insert(0, int(source_pk))

    value_dict = {}
    source_data_df = None
    scaler = preprocessing.StandardScaler()
    for number, pk in enumerate(similarity_list):
        target_t_amt_list = GinQuery.select_t_amt(pk_detail=pk).fetchall()
        source_df = pd.DataFrame(target_t_amt_list, columns=['master_idx', 'deal_ymd', 'cnt', 'floor', 't_amt'])
        if len(source_df) == 0:
            continue

        date_list = [str(i)[:7] for i in pd.date_range('2006-1', '2019-8', freq='M')]
        date_df = pd.DataFrame(date_list)
        date_df['t_amt'] = np.nan
        date_df.columns = ['date', 't_amt']
        # print(date_df)

        for date in source_df.deal_ymd:
            temp = source_df[source_df.deal_ymd == date]
            date = '{0}-{1:02d}'.format(date[:4], int(date[4:]))
            index = date_df[date_df.date == date].index[0]
            date_df.iloc[index, 1] = int(sum(temp.t_amt) / len(temp))

        # interpolate
        date_df.t_amt = date_df.t_amt.interpolate()
        index_ = date_df[pd.isna(date_df.t_amt) == True].index
        if len(index_) != 0:
            index_ = index_[-1]
            value = date_df.iloc[index_+1, 1]
            for j in range(index_+1):
                date_df.iloc[j, 1] = value

        if number == 0:
            source_data_df = date_df
            continue

        x = scaler.fit_transform(source_data_df.t_amt.to_numpy().reshape(-1, 1))
        y = scaler.fit_transform(date_df.t_amt.to_numpy().reshape(-1, 1))
        value = rmse(y, x)
        value_dict[pk] = value

    sorted_x = dict(sorted(value_dict.items(), key=lambda kv: kv[1]))
    return list(sorted_x.keys())


def make_similarity_matrix():
    Point2D = collections.namedtuple('Point2D', ['x', 'y'])

    def cosine_similarity(a_, b_):
        return dot(a_, b_) / (norm(a_) * norm(b_))

    def x_y_distance(a_, b_):
        p1 = Point2D(x=a_[0], y=a_[1])
        p2 = Point2D(x=b_[0], y=b_[1])
        a_ = p1.x - p2.x
        b_ = p1.y - p2.y

        distance = math.sqrt((a_ * a_) + (b_ * b_))
        return distance

    GinQuery.create_similarity()
    df1 = pd.read_csv(file_path).drop('Unnamed: 0', axis=1)
    df2 = pd.read_csv(file_location_path).drop('Unnamed: 0', axis=1)

    a = df2.to_numpy()
    b = df2.T.to_numpy()
    size = a.shape[0]

    for i in range(size):
        similarity_list = []
        i_a = a[i, :]

        for j in range(size):
            j_b = b[:, j]

            # (x,y) distance
            x_y_dist = x_y_distance(i_a[2:], j_b[2:])

            # result = str(cd_similarity + x_y_dist)
            result = str(x_y_dist)[:10]
            similarity_list.append(result)

        similarity_list = np.array(similarity_list)
        indicate = np.argsort(similarity_list)[:50]
        similarity_pk_apt_detail_list = list(df1.iloc[indicate, :].pk_apt_detail)
        similarity_pk_apt_detail_list.remove(df1.iloc[i, :].pk_apt_detail)

        try:
            similarity_list = get_similarity_time_series(df1.iloc[i, :].pk_apt_detail, similarity_pk_apt_detail_list)
        except Exception as e:
            similarity_list = similarity_pk_apt_detail_list

        similarity_string = ','.join([str(i) for i in similarity_list])
        i_df = df1.iloc[i, :]
        detail_pk = int(i_df.pk_apt_detail)
        GinQuery.insert_similarity(detail_pk, similarity_string)


def get_rgb():
    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0]
    return color


def get_similarity_rank(detail_pk, output_size=100):
    result = GinQuery.select_similarity(detail_pk).fetchone()
    pk_detail_id, bldg_nm, si_cd, dong_cd, x, y, gen_dt, pk_similarity = result
    pk_similarity = np.array([pk for pk in pk_similarity.split(',')])
    pk_similarity = pk_similarity[:output_size]

    apt_map = folium.Map(location=[37.5102, 126.982], zoom_start=12)
    color = get_rgb()
    for pk in pk_similarity:
        pk_apt_master, pk_apt_detail, bldg_nm, location_site, search_dt, x, y \
            = GinQuery.get_apt_simple_detail(idx=str(pk)).fetchone()
        x, y = float(x), float(y)
        print('id : {} bldg_nm : {} (x,y) : ({},{}) gen_dt: {}'.format(pk_detail_id, bldg_nm, x, y, gen_dt))
        folium.RegularPolygonMarker((x, y),
                                    radius=10,
                                    color=color).add_to(apt_map)

        target_t_amt_list = GinQuery.select_t_amt(pk_detail=pk).fetchall()
        source_df = pd.DataFrame(target_t_amt_list, columns=['master_idx', 'deal_ymd', 'cnt', 'floor', 't_amt'])
        if len(source_df) == 0:
            continue
        p = plt.scatter(source_df['deal_ymd'], source_df['t_amt'])

    apt_map.save('apt.html')
    return pk_similarity


def regression(detail_pk, output_size):
    target = GinQuery.select_similarity(pk_apt_detail=detail_pk).fetchone()

    pk_detail_id, bldg_nm, si_cd, dong_cd, x, y, gen_dt, pk_similarity = target
    pk_similarity = [pk for pk in pk_similarity.split(',')]
    pk_similarity = np.array(pk_similarity)

    print(GinQuery.get_apt_detail(idx=pk_detail_id).fetchone())
    print('id : {} bldg_nm : {} (x,y) : ({},{}) gen_dt: {}'.format(pk_detail_id, bldg_nm, x, y, gen_dt))
    print('----------------------------------------------------------------------------------------------------')

    target_t_amt_list = GinQuery.select_t_amt(pk_detail=detail_pk).fetchall()
    source_df = pd.DataFrame(target_t_amt_list, columns=['master_idx', 'deal_ymd', 'cnt', 'floor', 't_amt'])
    p = plt.scatter(source_df['deal_ymd'], source_df['t_amt'])

    p_items = []
    pk_items = []

    temp = {}
    scaler = preprocessing.MinMaxScaler()

    s = 3
    for pk in pk_similarity[:s]:
        pk = str(pk)
        pk_apt_master, pk_apt_detail, bldg_nm, location_site, search_dt, x, y \
            = GinQuery.get_apt_simple_detail(idx=pk).fetchone()

        print(GinQuery.get_apt_detail(idx=pk).fetchone())
        print('id : {} bldg_nm : {} (x,y) : ({},{}) gen_dt: {}'.format(pk, bldg_nm, x, y, gen_dt))

        source_df = pd.DataFrame(GinQuery.select_t_amt(pk_detail=pk).fetchall(),
                                 columns=['master_idx', 'deal_ymd', 'cnt', 'floor', 't_amt'])
        if len(source_df) == 0:
            continue
        p = plt.scatter(source_df['deal_ymd'], source_df['t_amt'])
        p_items.append(p)
        pk_items.append(pk)

    plt.legend(p_items, pk_items)
    plt.show()


if __name__ == '__main__':
    # make_matrix()
    # choice_columns()
    # make_similarity_matrix()
    pk_ = 112
    out_size = 30

    # get_similarity_rank(detail_pk=pk_, output_size=out_size)
    regression(detail_pk=pk_, output_size=out_size)