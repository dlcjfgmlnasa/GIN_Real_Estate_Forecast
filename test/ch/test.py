# -*- coding:utf-8 -*-
from numpy import dot
from numpy.linalg import norm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from database import GinQuery
from settings import cnx
filename = 'apt_detail.csv'


def make_matrix():
    temp = []
    cursor = GinQuery.test2()
    i = 0
    for data in cursor.fetchall():
        (master_idx, edit_bldg_nm, sigungu_cd, dong_cd, location_site, apt_dong_nm, total_num_of_family,
         total_dong_cnt, max_jisang_floor, total_jucha, search_dt, latlngx, latlngy, pk_apt_detail, supply_extent,
         extent, num_of_family, apt_dong, price) = data
        # master_idx : 지인 master 아파트키
        # edit_bldg_nm : 단지명
        # sigungu_cd : 시군구 코드 (O)
        # dong_cd : 법정동 코드 (O)
        # location_site : 대치위치
        # apt_dong_nm : 동정보
        # total_num_of_family : 총 세대수 (O)
        # total_dong_cnt : 총 동수 (O)
        # max_jisang_floor : 최고층 (O)
        # total_jucha : 총 주차대수
        # search_dt : 준공일
        # latlngx : 위도
        # latlngy : 경도
        # pk_apt_detail : 지인 detail 아파트 key
        # supply_extent : 공급면적
        # extent : 전용면적
        # num_of_family : 면적별 세대수
        # apt_dong : 읍면동 이름
        # price : 분양 가격
        if not latlngx or not latlngy:
            continue

        latlngx = float(latlngx)
        latlngy = float(latlngy)

        supply_extent = float(supply_extent)
        extent = float(extent)

        temp.append([
            pk_apt_detail,              # 지인 detail 아파트 key
            master_idx,                 # 지인 master 아파트 key
            sigungu_cd,                 # 시군구 코드
            dong_cd,                    # 법정동 코드
            latlngx,                    # 위도
            latlngy,                    # 경도
            # 근처에 있는 학교
            # 근처에 있는 직장
        ])
        if i == 1000000:
            break
        i += 1
    temp = np.array(temp)
    df = pd.DataFrame(temp)
    df.columns = ['pk_apt_detail', 'master_idx', 'sigungu_cd', 'dong_cd', 'latlngx', 'latlngy']
    df.index = df['pk_apt_detail']
    df = df.drop('pk_apt_detail', axis=1)
    return df


def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))


def pearson_similarity(a, b):
    from sklearn.metrics.pairwise import euclidean_distances
    a = a.reshape(-1, 1)
    b = b.reshape(-1, 1)
    return sum(sum(euclidean_distances(a, b)))


def get_item_based_matrix(data, similarity='cosine') -> np.array:
    a = data    # => (data_size, feature_size)
    b = data.T  # => (feature_size, data_size)

    a = a.astype('float64')
    b = b.astype('float64')
    size = a.shape[0]

    if similarity == 'cosine':
        sim_f = cosine_similarity
    else:
        sim_f = pearson_similarity

    matrix = np.zeros((size, size)).astype('float64')
    for i in range(size):
        a_i = a[i, :]
        i_similarity = []
        for j in range(size):
            b_j = b[:, j]
            similarity_result = sim_f(a_i, b_j)
            i_similarity.append(similarity_result)
        matrix[i] = i_similarity
    return matrix


def get_item_based_matrix_v2(data, pk, similarity='cosine') -> np.array:
    a = data  # => (data_size, feature_size)
    b = data.T  # => (feature_size, data_size)

    a = a.astype('float64')
    b = b.astype('float64')

    if similarity == 'cosine':
        sim_f = cosine_similarity
    elif similarity == 'pearsor':
        sim_f = pearson_similarity

    a_i = a[pk, :]
    i_similarity = []
    size = a.shape[0]
    for j in range(size):
        b_j = b[:, j]
        similarity_result = sim_f(a_i, b_j)
        i_similarity.append(similarity_result)
    return np.array(i_similarity)


def similarity_rank(matrix, pk_apt_detail_id, out_size=10):
    import matplotlib.pyplot as plt
    items = matrix[pk_apt_detail_id]
    indices = np.argsort(items)[-out_size-1:-1]
    values = items[indices]

    return values, indices


def similarity_rank_v2(matrix, pk_apt_detail_id, out_size=10):
    print(matrix)


if __name__ == '__main__':
    from sklearn import preprocessing
    # frame = make_matrix()
    # frame.to_csv('test.csv')

    total_frame = pd.read_csv('test.csv')
    frame = total_frame[['latlngx', 'latlngy']]
    x = preprocessing.StandardScaler().fit_transform(frame)
    output_size = 100

    p = 12
    r = total_frame[total_frame.pk_apt_detail == p]
    latlngx = float(r.latlngx)
    latlngy = float(r.latlngy)

    similarity_matrix_i = get_item_based_matrix_v2(frame.to_numpy().astype('float64'),
                                                   r.index[0], similarity='pearsor')

    ind = np.argsort(similarity_matrix_i)[-output_size-1:-1]
    ind = reversed(ind)

    import random
    import folium

    apt_map = folium.Map(location=[37.5102, 126.982], zoom_start=12)

    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0]
    folium.RegularPolygonMarker((latlngx, latlngy),
                                radius=10,
                                color=color).add_to(apt_map)

    color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])][0]

    for pk in ind:
        temp = total_frame.iloc[pk, :]
        pk_apt_detail = int(temp['pk_apt_detail'])
        print(pk_apt_detail, end=' ')

        cursor = GinQuery.test(str(pk_apt_detail))
        result = cursor.fetchone()
        print(result[1] + '\t : ' + result[4])

        x = temp.latlngx
        y = temp.latlngy
        folium.RegularPolygonMarker((x, y),
                                    radius=10,
                                    color=color).add_to(apt_map)

    apt_map.save('apt_item_based.html')


    # mx.dump('test_similarity.matrix')
    #
    # mx = np.load('test_similarity.matrix', allow_pickle=True)
    # value, indices = similarity_rank(mx, pk_apt_detail_id=10, out_size=8)
    #
    # idx = 4
    # cursor = GinQuery.get_actual_price(idx=str(idx))
    # line = cursor.fetchall()
    # t_amt_dict = dict()
    # for i in line:
    #     t_amt, deal_ymd = i[4], i[6]
    #     t_amt = int(t_amt)
    #     if deal_ymd not in t_amt_dict.keys():
    #         t_amt_dict[deal_ymd] = []
    #     t_amt_dict[deal_ymd].append(t_amt)
    #
    # tmp = {}
    # for key, value in t_amt_dict.items():
    #     result = sum(t_amt_dict[key]) / len(t_amt_dict[key])
    #     tmp[key] = result
    # print(tmp)

    # p_items = []
    # for num, idx in enumerate(indices):
    #     cursor = GinQuery.get_actual_price(idx=str(idx))
    #     df = pd.read_sql(cursor.statement, cnx)
    #     x_lines = [float(i + '.' + j) for i, j in zip(df['deal_ymd'], df['day'])]
    #     pk_apt_master = df['pk_apt_master']
    #     y_lines = df['t_amt']
    #     p = plt.scatter(x_lines, y_lines)
    #     p_items.append(p)
    #
    # plt.legend(p_items, indices)
    # plt.show()
