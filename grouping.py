# -*- coding:utf-8 -*-
import operator
import datetime
import numpy as np
import pandas as pd
from database import GinAptQuery
from settings import db_cursor as cursor


class AptGroup(object):
    # 비슷한 아파트들을 묶는 클래스
    @staticmethod
    def get_price_df(apt_detail_pk: int, trade_cd: str) -> pd.DataFrame:
        query = GinAptQuery()

        price_df = pd.DataFrame(
            query.get_trade_price(
                apt_detail_pk=apt_detail_pk,
                trade_cd=trade_cd
            ),
            columns=['pk_apt_trade', 'pk_apt_detail', 'year', 'mon', 'real_day', 'floor', 'extent', 'price']
        )

        # Dataset Cleaning
        price_df.price = price_df.price / price_df.extent
        trg_date = ['{0}-{1:02d}-{2:02d}'.format(year, int(mon), int(day))
                    for year, mon, day in zip(price_df.year, price_df.mon, price_df.real_day)]
        price_df['date'] = trg_date
        price_df = price_df.drop(['pk_apt_trade', 'pk_apt_detail', 'year', 'mon', 'real_day', 'floor', 'extent'],
                                 axis=1)[['date', 'price']]
        price_df.price = price_df.price.astype(np.float)

        temp = []
        for date in pd.date_range(start='2006-01-01', end=datetime.datetime.now().strftime('%Y-%m-%d')):
            date = date.strftime('%Y-%m-%d')
            df = price_df[price_df.date == date]
            if len(df) != 0:
                temp.append((date, np.average(df.price)))
            else:
                temp.append((date, np.nan))

        # Interpolation...
        interpolate_price_df = pd.DataFrame(temp, columns=['date', 'price']).interpolate()
        try:
            last_index = interpolate_price_df[interpolate_price_df.price.isna()].index[-1] + 1
            first_interpolate_value = interpolate_price_df.price.iloc[last_index]
            interpolate_price_df = interpolate_price_df.fillna(first_interpolate_value)
        except IndexError:
            pass
        return interpolate_price_df

    @staticmethod
    def apt_apt_similarity(apt_detail_pk: int, trade_cd: str, limit_size: int):
        def root_mean_square_error(pred, true):
            return np.sqrt(((pred - true) ** 2).mean())

        # apt_detail_pk 로 부터 3Km 이내의 아파트 리스트 출력
        nearest_apt_df = pd.DataFrame(
            GinAptQuery.get_nearest_3km_apt(
                apt_detail_pk=apt_detail_pk
            ),
            columns=['distance', 'detail_pk']
        )

        target_price_df = AptGroup.get_price_df(
            apt_detail_pk=apt_detail_pk,
            trade_cd=trade_cd
        )

        # Calculation Similarity
        similarity = {}
        nearest_apt_list = nearest_apt_df.detail_pk.values
        size = len(nearest_apt_list)
        for i, detail_pk in enumerate(nearest_apt_list):
            price_df = AptGroup.get_price_df(
                apt_detail_pk=int(detail_pk),
                trade_cd=trade_cd
            )

            if len(price_df) == 0:
                continue

            # Comparing...
            print('\t[{}/{}]  Comparing = target_pk : {} / source_pk : {}'
                  .format(size, i, apt_detail_pk, detail_pk))
            similarity_value = root_mean_square_error(
                pred=price_df.price.values,
                true=target_price_df.price.values
            )
            print('\t\t=> similarity : {}\n'.format(similarity_value))
            similarity[detail_pk] = similarity_value

        # Ranking...
        similarity_ranking_detail_pk = [
            detail_pk
            for detail_pk, similarity_value in sorted(similarity.items(), key=operator.itemgetter(1))
        ][:limit_size]
        similarity_ranking_detail_pk = [apt_detail_pk] + similarity_ranking_detail_pk
        return similarity_ranking_detail_pk


class AptFloorGroup(object):
    # 아파트 층과 관련된 클래스
    low_floor = 3

    @staticmethod
    def get_similarity_apt_floor_lists(apt_detail_pk: int):
        # 같은 아파트의있는 층과 비슷한 아파트의 층을 Grouping 해주는 함수
        max_floor = GinAptQuery.get_max_floor(apt_detail_pk).fetchone()[0]
        low_floor = AptFloorGroup.low_floor

        middle = int((max_floor - low_floor) * 0.8) + low_floor  # 80% 중층 그 이상은 20%

        return {
            'low': list(range(-10, low_floor + 1)),
            'medium': list(range(low_floor + 1, middle + 1)),
            'high': list(range(middle + 1, max_floor + 1)),
            'max_floor': max_floor
        }

    @staticmethod
    def get_similarity_apt_floor_list(apt_detail_pk: int, floor: str):
        # 같은 아파트의있는 층과 비슷한 아파트의 층을 Grouping 해주는 함수
        max_floor = GinAptQuery.get_max_floor(apt_detail_pk).fetchone()[0]
        low_floor = AptFloorGroup.low_floor

        floor = int(floor)

        if floor <= low_floor:
            return list(range(-10, low_floor + 1))
        else:
            middle = int((max_floor - low_floor) * 0.8) + low_floor  # 80% 중층 그 이상은 20%
            middle_range = range(low_floor + 1, middle + 1)
            high_range = range(middle + 1, max_floor + 1)
            if floor in middle_range:
                return middle_range
            if floor in high_range:
                return high_range

    @staticmethod
    def get_floor_level(apt_detail_pk: int, floor: str):
        # 해당 층의 level(low, middle, high) 을 출력해주는 함수
        max_floor = GinAptQuery.get_max_floor(apt_detail_pk).fetchone()[0]
        low_floor = AptFloorGroup.low_floor

        floor = int(floor)

        # 80% 중층 그 이상은 20%
        l_floor_range = range(-10, low_floor + 1)
        middle = int((max_floor - low_floor) * 0.8) + low_floor

        m_floor_range = range(low_floor + 1, middle + 1)
        h_floor_range = range(middle + 1, max_floor + 1)

        if floor in l_floor_range:
            return 'low'
        elif floor in m_floor_range:
            return 'medium'
        elif floor in h_floor_range:
            return 'high'

    @staticmethod
    def get_floor_from_floor_level(apt_detail_pk: int, floor_lvl='low'):
        # 해당 층의 level 에 맞는 floor 를 출력해주는 함수
        max_floor = GinAptQuery.get_max_floor(apt_detail_pk).fetchone()[0]
        low_floor = AptFloorGroup.low_floor

        # 80% 중층 그 이상은 20%
        middle = int((max_floor - low_floor) * 0.8) + low_floor

        if floor_lvl == 'low':
            return range(-10, low_floor + 1)
        elif floor_lvl == 'medium':
            return range(low_floor + 1, middle + 1)
        elif floor_lvl == 'high':
            return range(middle + 1, max_floor + 1)

    @staticmethod
    def get_floor_min_max(floor: int, floor_lists):
        low_floor = AptFloorGroup.low_floor
        floor = int(floor)

        if floor < low_floor:
            floor_list = floor_lists['low']
        else:
            max_floor = int(floor_lists['max_floor'])
            middle = int((max_floor - low_floor) * 0.8) + low_floor
            if floor > middle:
                floor_list = floor_lists['high']
            else:
                floor_list = floor_lists['medium']
        return {
            'max': max(floor_list),
            'min': min(floor_list)
        }

class AptComplexGroup(object):
    # 같은 단지안에 있는 아파트들을 Grouping 해주는 클래스
    @staticmethod
    def get_similarity_apt_list(apt_detail_pk):
        # 비슷한 아파트 리스트 출력
        cursor.execute("""
            SELECT master_idx
            FROM apt_detail
            WHERE idx=%s
        """, params=(apt_detail_pk,))
        apt_master_pk = cursor.fetchone()[0]
        group_list = AptComplexGroup.apt_complex_groping(apt_master_pk)

        # group searching...
        for group in group_list:
            if apt_detail_pk in group:
                return group

    @staticmethod
    def __apt_group__(temp):
        # 아파트의 면적으로 평으로 환산한 후 위아래 3평을 같은 아파트로 묶음
        group = []
        if len(temp) == 0:
            return group

        df = pd.DataFrame()
        num = 0
        while True:
            if num == 0:
                df = pd.DataFrame(temp, columns=['detail_pk', 'extent', 'num_of_family'])
            # 아파트의 면적으로 평으로 환산한 후 위아래 3평을 같은 아파트로 묶음
            df['spm'] = df.extent * 0.3025
            target_spm = df.spm.iloc[0]

            df['spm_range'] = df.spm.apply(lambda spm: spm - 3 < target_spm < spm + 3)

            group_df = df[df.spm_range][['detail_pk', 'extent', 'num_of_family']]
            df = df[~df.spm_range][['detail_pk', 'extent', 'num_of_family']]

            group.append(group_df)
            if len(df) == 0:
                break
            num += 1
        return [list(df.detail_pk) for df in group]

    @staticmethod
    def apt_complex_groping(apt_master_pk):
        # 아파트 단지안에 있는 아파트들을 Grouping
        cursor.execute("""
            SELECT idx, extent, num_of_family
            FROM apt_detail
            WHERE master_idx=%s
            ORDER BY num_of_family DESC
        """, params=(apt_master_pk,))

        group1 = []  # 85 미만  (extent < 85)
        group2 = []  # 85 이상 135 미만  (85 <= extent < 135)
        group3 = []  # 135 이상  (135 <= extent)

        for idx, extent, num_of_family in cursor.fetchall():
            extent = float(extent)
            date = (idx, extent, num_of_family)

            if extent < 85:
                group1.append(date)
            elif 85 <= extent < 135:
                group2.append(date)
            elif extent >= 135:
                group3.append(date)

        # 전체 데이터 Grouping
        group1 = AptComplexGroup.__apt_group__(group1)
        group2 = AptComplexGroup.__apt_group__(group2)
        group3 = AptComplexGroup.__apt_group__(group3)
        total_group = [i for group in (group1, group2, group3) for i in group]
        return total_group
