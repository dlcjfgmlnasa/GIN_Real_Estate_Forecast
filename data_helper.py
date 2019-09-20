# -*- coding:utf-8 -*-
from datetime import datetime
import pandas as pd
from database import GinAptQuery
from settings import db_cursor as cursor
from feature import AptPriceRegressionFeature


class AptComplexGroup(object):
    # 같은 단지안에 있는 아파트들을 Grouping 해주는 클래스
    @staticmethod
    def get_similarity_apt_list(apt_detail_pk):
        # 비슷한 아파트 리스트 출력
        cursor.execute("""
            SELECT master_idx
            FROM apt_detail
            WHERE idx=%s
        """, params=(apt_detail_pk, ))
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

            df['spm_range'] = df.spm.apply(lambda spm: spm-3 < target_spm < spm + 3)

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
        """, params=(apt_master_pk, ))

        group1 = []     # 85 미만  (extent < 85)
        group2 = []     # 85 이상 135 미만  (85 <= extent < 135)
        group3 = []     # 135 이상  (135 <= extent)

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

