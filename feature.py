# -*- coding:utf-8 -*-
import numpy as np
import pandas as pd
from datetime import datetime
from datedelta import datedelta
from database import GinAptQuery
from settings import db_cursor as cursor


class AptFloorGroup(object):
    # 같은 아파트의있는 층과 비슷한 아파트의 층을 Grouping 해주는 클래스
    @staticmethod
    def get_similarity_apt_floor_list(apt_detail_pk: int, floor: str):
        cursor.execute("""
            SELECT b.max_jisang_floor
            FROM apt_detail a
            INNER JOIN apt_master b
              ON a.master_idx = b.idx
            WHERE a.idx = %s
        """, params=(apt_detail_pk, ))
        max_floor = cursor.fetchone()[0]

        low_floor = 3
        floor = int(floor)

        if floor <= low_floor:
            return list(range(-10, low_floor+1))
        else:
            middle = int((max_floor - low_floor) * 0.8)     # 80% 중층 그 이상은 20%
            middle_range = list(range(low_floor+1, middle+1))
            high_range = list(range(middle+1, max_floor+1))
            if floor in middle_range:
                return middle_range
            if floor in high_range:
                return high_range


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


class AptPriceRegressionFeature(object):
    # 아파트 매매가격 예측을 위한 피쳐 클래스
    def __init__(self, apt_master_pk: int, apt_detail_pk: int, trade_cd: str):
        self.apt_master_pk = apt_master_pk
        self.apt_detail_pk = apt_detail_pk
        self.apt_complex_group_list = AptComplexGroup.get_similarity_apt_list(
            apt_detail_pk=apt_detail_pk
        )
        self.trade_cd = trade_cd
        self.query = GinAptQuery()

    # ---------------------------------------------------------------------------------------------- #
    # 1. 매물 정보
    # ---------------------------------------------------------------------------------------------- #
    def sale_price_with_floor(self, trg_date: datetime, month_size: int, floor: str, extent: float)\
            -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매물가격]을 이용한 feature
        # ex) apt detail pk 가 1인 아파트 8층 건물의 [매물가격정보]를 이용하고 싶다.

        pre_date = trg_date - datedelta(months=month_size)
        date_range = pd.date_range(pre_date, trg_date)
        date_range = ','.join([date.strftime('"%Y-%m-%d"') for date in date_range])

        df = pd.DataFrame(
            self.query.get_sale_price_with_floor(
                apt_detail_pk=self.apt_detail_pk,
                date_range=date_range,
                floor=floor,
                trade_cd=self.trade_cd
            ),
            columns=['apt_detail_pk', 'date', 'floor', 'price']
        )
        df.price = df.price / extent
        df.price = df.price.astype(np.float)
        return df

    def sale_price_with_floor_recent(self, trg_date: datetime, max_month_size: int,
                                     recent_month_size: int, floor: str, extent: float)\
            -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매물가격]을 이용한 feature
        # max_month_size 안에서 가장 최근 데이터를 기준으로 recent_month_size 만큼의 [매물가격] 출력
        # ex) apt detail pk 가 1인 아파트 8층 건물의 가장 최근 [매물가격정보]를 이용하고 싶다.

        df = self.sale_price_with_floor(
            trg_date=trg_date,
            month_size=max_month_size,
            floor=floor,
            extent=extent
        )

        if len(df) != 0:
            recent_price = df.iloc[-1:]
            recent_date = list(recent_price.date)[0]
            recent_pre_date = recent_date - datedelta(months=recent_month_size)

            recent_date_range = pd.date_range(recent_pre_date, recent_date)
            df = df[df.date.apply(lambda date: date in recent_date_range)]
            return df
        return pd.DataFrame()

    def sale_price_with_floor_group(self, trg_date: datetime, month_size: int,
                                    floor: str, extent: float) -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매물가격]을 이용한 feature
        # 비슷한 층수 데이터도 같이 사용

        # floor list 출력
        floor_list = AptFloorGroup.get_similarity_apt_floor_list(
            apt_detail_pk=self.apt_detail_pk,
            floor=floor
        )
        df = self.sale_price_with_floor(
            trg_date=trg_date,
            month_size=month_size,
            floor=','.join([str(floor) for floor in floor_list]),
            extent=extent
        )
        return df

    def sale_price_with_floor_group_recent(self, trg_date: datetime, max_month_size: int,
                                           recent_month_size: int, floor: str, extent: float) -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매물가격]을 이용한 feature
        # 비슷한 층수 데이터도 같이 사용
        # 최근 데이터만 사용

        # floor list 출력
        floor_list = AptFloorGroup.get_similarity_apt_floor_list(
            apt_detail_pk=self.apt_detail_pk,
            floor=floor
        )

        df = self.sale_price_with_floor_recent(
            trg_date=trg_date,
            max_month_size=max_month_size,
            recent_month_size=recent_month_size,
            floor=','.join([str(floor) for floor in floor_list]),
            extent=extent
        )
        return df

    def sale_price_with_complex_group(self, trg_date: datetime, month_size: int,
                                      floor: str, extent: float) -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매물가격]을 이용한 feature
        # 비슷한 단지의 건물 데이터를 같이 사용
        # 비슷한 층수 데이터도 같이 사용

        pre_date = trg_date - datedelta(months=month_size)
        date_range = pd.date_range(pre_date, trg_date)
        date_range = ','.join([date.strftime('"%Y-%m-%d"') for date in date_range])

        # floor list 출력
        floor_list = AptFloorGroup.get_similarity_apt_floor_list(
            apt_detail_pk=self.apt_detail_pk,
            floor=floor
        )

        df = pd.DataFrame(
            self.query.get_sale_price_with_floor(
                apt_detail_pk=','.join([str(apt) for apt in self.apt_complex_group_list]),
                date_range=date_range,
                floor=','.join([str(floor) for floor in floor_list]),
                trade_cd=self.trade_cd
            ),
            columns=['apt_detail_pk', 'date', 'floor', 'price']
        )
        df.price = df.price / extent
        df.price = df.price.astype(np.float)
        return df

    def sale_price_with_complex_group_recent(self, trg_date: datetime, max_month_size: int,
                                             recent_month_size: int, floor: str, extent: float):
        # 예측하고자하는 층의 이전 시간대의 [매물가격]을 이용한 feature
        # 비슷한 단지의 건물 데이터를 같이 사용
        # 비슷한 층수 데이터도 같이 사용
        # 최근 데이터만 사용

        df = self.sale_price_with_complex_group(
            trg_date=trg_date,
            month_size=max_month_size,
            floor=floor,
            extent=extent
        )

        if len(df) != 0:
            recent_price = df.iloc[-1:]
            recent_date = list(recent_price.date)[0]
            recent_pre_date = recent_date - datedelta(months=recent_month_size)

            recent_date_range = pd.date_range(recent_pre_date, recent_date)
            df = df[df.date.apply(lambda date: date in recent_date_range)]
            return df
        return pd.DataFrame()

    # ---------------------------------------------------------------------------------------------- #
    # 2. 매매 정보
    # ---------------------------------------------------------------------------------------------- #
    def trade_price_with_floor(self, trg_date: datetime, month_size: int, floor: str, extent: float, trade_pk=None)\
            -> pd.DataFrame():
        # 예측하고자하는 층의 이전 시간대의 [매매가격]을 이용한 feature
        # ex) apt detail pk 가 1인 아파트 8층 건물의 [매매가격정보]를 이용하고 싶다.

        pre_date = trg_date - datedelta(months=month_size)
        date_range = pd.date_range(pre_date, trg_date)
        date_range = ','.join([date.strftime('"%Y%m"') for date in date_range])

        df = pd.DataFrame(
            self.query.get_trade_price_with_floor(
                apt_detail_pk=self.apt_detail_pk,
                date_range=date_range,
                floor=floor,
                trade_cd=self.trade_cd
            ),
            columns=['pk_apt_trade', 'apt_detail_pk', 'date', 'floor', 'price']
        )
        df.price = df.price / extent
        df.price = df.price.astype(np.float)

        if trade_pk:
            # Train 을 위해 trade_pk 값은 제외 시킴
            df = df[df.pk_apt_trade.apply(lambda pk: pk != trade_pk)]
        df = df[['apt_detail_pk', 'date', 'floor', 'price']]
        return df

    def trade_price_with_floor_recent(self, trg_date: datetime, max_month_size: int, recent_month_size: int,
                                      floor: str, extent: float, trade_pk=None)\
            -> pd.DataFrame():
        # 예측하고자하는 층의 이전 시간대의 [매매가격]을 이용한 feature
        # max_month_size 안에서 가장 최근 데이터를 기준으로 recent_month_size 만큼의 [매매가격] 출력
        # ex) apt detail pk 가 1인 아파트 8층 건물의 가장 최근 [매매가격정보]를 이용하고 싶다.

        df = self.trade_price_with_floor(
            trg_date=trg_date,
            month_size=max_month_size,
            floor=floor,
            extent=extent,
            trade_pk=trade_pk
        )

        if len(df) != 0:
            recent_price = df.iloc[-1:]
            recent_date = datetime.strptime(list(recent_price.date)[0], "%Y%m")
            recent_pre_date = recent_date - datedelta(months=recent_month_size)

            recent_date_range = pd.date_range(recent_pre_date, recent_date, freq='MS')
            recent_date_range = [str(data.strftime("%Y%m")) for data in recent_date_range]

            df = df[df.date.apply(lambda date: date in recent_date_range)]
            return df
        return pd.DataFrame()

    def trade_price_with_floor_group(self, trg_date: datetime, month_size: int,
                                     floor: str, extent: float, trade_pk=None) -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매매가격]을 이용한 feature
        # 비슷한 층수 데이터도 같이 사용

        # floor list 출력
        floor_list = AptFloorGroup.get_similarity_apt_floor_list(
            apt_detail_pk=self.apt_detail_pk,
            floor=floor
        )
        df = self.trade_price_with_floor(
            trg_date=trg_date,
            month_size=month_size,
            floor=','.join([str(floor) for floor in floor_list]),
            extent=extent,
            trade_pk=trade_pk
        )
        return df

    def trade_price_with_floor_group_recent(self, trg_date: datetime, max_month_size: int,
                                            recent_month_size: int, floor: str, extent: float, trade_pk=None) \
            -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매매가격]을 이용한 feature
        # 비슷한 층수 데이터도 같이 사용
        # 최근 데이터만 이용

        # floor list 출력
        floor_list = AptFloorGroup.get_similarity_apt_floor_list(
            apt_detail_pk=self.apt_detail_pk,
            floor=floor
        )

        df = self.trade_price_with_floor_recent(
            trg_date=trg_date,
            max_month_size=max_month_size,
            recent_month_size=recent_month_size,
            floor=','.join([str(floor) for floor in floor_list]),
            extent=extent,
            trade_pk=trade_pk
        )
        return df

    def trade_price_with_complex_group(self, trg_date: datetime, month_size: int,
                                       floor: str, extent: float, trade_pk=None) -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매매가격]을 이용한 feature
        # 비슷한 단지의 건물 데이터를 같이 사용
        # 비슷한 층수 데이터도 같이 사용

        pre_date = trg_date - datedelta(months=month_size)
        date_range = pd.date_range(pre_date, trg_date)
        date_range = ','.join([date.strftime('"%Y%m"') for date in date_range])

        # floor list 출력
        floor_list = AptFloorGroup.get_similarity_apt_floor_list(
            apt_detail_pk=self.apt_detail_pk,
            floor=floor
        )

        df = pd.DataFrame(
            self.query.get_trade_price_with_floor(
                apt_detail_pk=','.join([str(apt) for apt in self.apt_complex_group_list]),
                date_range=date_range,
                floor=','.join([str(floor) for floor in floor_list]),
                trade_cd=self.trade_cd
            ),
            columns=['pk_apt_trade', 'apt_detail_pk', 'date', 'floor', 'price']
        )
        df.price = df.price / extent
        df.price = df.price.astype(np.float)

        if trade_pk:
            # Train 을 위해 trade_pk 값을 제외 시킴
            df = df[df.pk_apt_trade.apply(lambda pk: pk != trade_pk)]

        df = df[['apt_detail_pk', 'date', 'floor', 'price']]
        return df

    def trade_price_with_complex_group_recent(self, trg_date: datetime, max_month_size: int,
                                              recent_month_size: int, floor: str, extent: float, trade_pk=None) \
            -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매매가격]을 이용한 feature
        # 비슷한 단지의 건물 데이터를 같이 사용
        # 비슷한 층수 데이터도 같이 사용
        # 최근 데이터만 사용

        df = self.trade_price_with_complex_group(
            trg_date=trg_date,
            month_size=max_month_size,
            floor=floor,
            extent=extent,
            trade_pk=trade_pk
        )

        if len(df) != 0:
            recent_price = df.iloc[-1:]
            recent_date = datetime.strptime(list(recent_price.date)[0], '%Y%m')
            recent_pre_date = recent_date - datedelta(months=recent_month_size)
            recent_date_range = pd.date_range(recent_pre_date, recent_date, freq='MS')
            recent_date_range = [str(data.strftime("%Y%m")) for data in recent_date_range]

            df = df[df.date.apply(lambda date: date in recent_date_range)]
            return df
        return df


def make_feature(feature_name_list, apt_master_pk, apt_detail_pk, trade_cd,
                 trg_date, sale_month_size, sale_recent_month_size,
                 trade_month_size, trade_recent_month_size,
                 floor, extent, trade_pk=None):
    feature = AptPriceRegressionFeature(
        apt_master_pk=apt_master_pk,
        apt_detail_pk=apt_detail_pk,
        trade_cd=trade_cd
    )

    features = []
    for feature_name in feature_name_list:
        df = pd.DataFrame()
        if feature_name == 'sale_price_with_floor':
            df = feature.sale_price_with_floor(
                trg_date=trg_date,
                month_size=sale_month_size,
                floor=floor,
                extent=extent
            )
            if len(df) == 0:
                return None
        elif feature_name == 'sale_price_with_floor_recent':
            df = feature.sale_price_with_floor_recent(
                trg_date=trg_date,
                max_month_size=sale_month_size,
                recent_month_size=sale_recent_month_size,
                floor=floor,
                extent=extent
            )
        elif feature_name == 'sale_price_with_floor_group':
            df = feature.sale_price_with_floor_group(
                trg_date=trg_date,
                month_size=sale_month_size,
                floor=floor,
                extent=extent
            )
        elif feature_name == 'sale_price_with_floor_group_recent':
            df = feature.sale_price_with_floor_recent(
                trg_date=trg_date,
                max_month_size=sale_month_size,
                recent_month_size=sale_recent_month_size,
                floor=floor,
                extent=extent
            )
        elif feature_name == 'sale_price_with_complex_group':
            df = feature.sale_price_with_complex_group(
                trg_date=trg_date,
                month_size=sale_month_size,
                floor=floor,
                extent=extent
            )
        elif feature_name == 'sale_price_with_complex_group_recent':
            df = feature.sale_price_with_complex_group_recent(
                trg_date=trg_date,
                max_month_size=sale_month_size,
                recent_month_size=sale_recent_month_size,
                floor=floor,
                extent=extent
            )

        elif feature_name == 'trade_price_with_floor':
            df = feature.trade_price_with_floor(
                trg_date=trg_date,
                month_size=trade_month_size,
                floor=floor,
                extent=extent,
                trade_pk=trade_pk
            )
        elif feature_name == 'trade_price_with_floor_recent':
            df = feature.trade_price_with_floor_recent(
                trg_date=trg_date,
                max_month_size=trade_month_size,
                recent_month_size=trade_recent_month_size,
                floor=floor,
                extent=extent,
                trade_pk=trade_pk
            )
        elif feature_name == 'trade_price_with_floor_group':
            df = feature.trade_price_with_floor_group(
                trg_date=trg_date,
                month_size=trade_month_size,
                floor=floor,
                extent=extent,
                trade_pk=trade_pk
            )
        elif feature_name == 'trade_price_with_floor_group_recent':
            df = feature.trade_price_with_floor_group_recent(
                trg_date=trg_date,
                max_month_size=trade_month_size,
                recent_month_size=trade_recent_month_size,
                floor=floor,
                extent=extent,
                trade_pk=trade_pk
            )
        elif feature_name == 'trade_price_with_complex_group':
            df = feature.trade_price_with_complex_group(
                trg_date=trg_date,
                month_size=trade_month_size,
                floor=floor,
                extent=extent,
                trade_pk=trade_pk
            )
        elif feature_name == 'trade_price_with_complex_group_recent':
            df = feature.trade_price_with_complex_group_recent(
                trg_date=trg_date,
                max_month_size=trade_month_size,
                recent_month_size=trade_recent_month_size,
                floor=floor,
                extent=extent,
                trade_pk=trade_pk
            )

        if len(df) != 0:
            value = np.average(df.price)
        else:
            value = np.nan
        features.append(value)

    feature_df = pd.DataFrame([features], columns=feature_name_list)
    return feature_df
