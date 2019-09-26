# -*- coding:utf-8 -*-
import pandas as pd
from database import GinAptQuery
from settings import db_cursor as cursor


class AptGroup(object):
    # 아파트 층과
    def apt_apt_similarity(self):
        pass


class AptFloorGroup(object):
    # 아파트 층과 관련된 클래스
    low_floor = 3

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
            ORDER BY apt_detail.num_of_family DESC
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
