# -*- coding:utf-8 -*-
import settings
import numpy as np
import pandas as pd
from datetime import datetime
from datedelta import datedelta
from database import GinAptQuery
from grouping import AptFloorGroup, AptComplexGroup


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

    def sale_price_with_similarity_apt_group(self, trg_date: datetime, month_size: int, floor: str) -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매물가격]을 이용한 feature
        # 비슷한 아파트 건물 데이터를 같이 사용
        # 비슷한 층수 데이터도 같이 사용

        pre_date = trg_date - datedelta(months=month_size)
        date_range = pd.date_range(pre_date, trg_date)
        date_range = ','.join([date.strftime('"%Y-%m-%d"') for date in date_range])

        apt_similarity_list = self.query.get_similarity_apt_list(apt_detail_pk=self.apt_detail_pk).fetchone()[0]
        apt_similarity_list = apt_similarity_list.split(',')

        # floor level 추정(저층, 중층 고층)
        floor_level = AptFloorGroup.get_floor_level(
            apt_detail_pk=self.apt_detail_pk,
            floor=floor
        )

        similarity_data = {
            pk: list(AptFloorGroup.get_floor_from_floor_level(pk, floor_level))
            for pk in apt_similarity_list
        }

        total_df = []
        for similarity_apt_detail_pk, similarity_floor_list in similarity_data.items():
            if len(similarity_floor_list) == 0:
                continue

            df = pd.DataFrame(
                self.query.get_sale_price_with_floor_extent(
                    apt_detail_pk=similarity_apt_detail_pk,
                    date_range=date_range,
                    floor=','.join([str(floor) for floor in similarity_floor_list]),
                    trade_cd=self.trade_cd
                ),
                columns=['apt_detail_pk', 'date', 'floor', 'extent', 'price']
            )
            df.price = df.price.astype(np.float)
            df.extent = df.extent.astype(np.float)
            df.price = df.price / df.extent
            df = df.drop('extent', axis=1)

            total_df.append(df)
        total_df = pd.concat(total_df)

        return total_df

    def sale_price_with_similarity_apt_group_recent(self, trg_date: datetime, max_month_size: int,
                                                    recent_month_size: int, floor: str) -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매물가격]을 이용한 feature
        # 비슷한 아파트 건물 데이터를 같이 사용
        # 비슷한 층수 데이터도 같이 사용
        # 최근 데이터만 사용

        df = self.sale_price_with_similarity_apt_group(
            trg_date=trg_date,
            month_size=max_month_size,
            floor=floor
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
            recent_date = datetime.strptime(list(df.iloc[-1:].date)[0], "%Y%m")
            recent_pre_date = recent_date - datedelta(months=recent_month_size)

            recent_date_range = pd.date_range(recent_pre_date, recent_date, freq='MS')
            recent_date_range = [str(data.strftime("%Y%m")) for data in recent_date_range]

            df = df[df.date.apply(lambda date: date in recent_date_range)]
            return df
        return pd.DataFrame()

    def trade_price_with_similarity_apt_group(self, trg_date: datetime, month_size: int,
                                              floor: str, trade_pk=None) -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매매가격]을 이용한 feature
        # 비슷한 아파트 건물 데이터를 같이 사용
        # 비슷한 층수 데이터도 같이 사용

        pre_date = trg_date - datedelta(months=month_size)
        date_range = pd.date_range(pre_date, trg_date, freq='MS')
        date_range = ','.join([date.strftime('"%Y%m"') for date in date_range])

        apt_similarity_list = self.query.get_similarity_apt_list(apt_detail_pk=self.apt_detail_pk).fetchone()[0]
        apt_similarity_list = apt_similarity_list.split(',')

        # floor level 추정(저층, 중층 고층)
        floor_level = AptFloorGroup.get_floor_level(
            apt_detail_pk=self.apt_detail_pk,
            floor=floor
        )

        similarity_data = {
            pk: list(AptFloorGroup.get_floor_from_floor_level(pk, floor_level))
            for pk in apt_similarity_list
        }

        total_df = []
        for similarity_apt_detail_pk, similarity_floor_list in similarity_data.items():
            if len(similarity_floor_list) == 0:
                continue
            df = pd.DataFrame(
                self.query.get_trade_price_with_floor_extent(
                    apt_detail_pk=similarity_apt_detail_pk,
                    date_range=date_range,
                    floor=','.join([str(floor) for floor in similarity_floor_list]),
                    trade_cd=self.trade_cd
                ),
                columns=['pk_apt_trade', 'apt_detail_pk', 'date', 'floor', 'extent', 'price']
            )
            df.price = df.price.astype(np.float)
            df.extent = df.extent.astype(np.float)
            df.price = df.price / df.extent
            df = df.drop('extent', axis=1)

            total_df.append(df)
        total_df = pd.concat(total_df)

        if trade_pk:
            # Train 을 위해 trade_pk 값을 제외 시킴
            total_df = total_df[total_df.pk_apt_trade != trade_pk]
        return total_df

    def trade_price_with_similarity_apt_group_recent(self, trg_date: datetime, max_month_size: int,
                                                     recent_month_size: int, floor: str, trade_pk=None) -> pd.DataFrame:
        # 예측하고자하는 층의 이전 시간대의 [매매가격]을 이용한 feature
        # 비슷한 아파트 건물 데이터를 같이 사용
        # 비슷한 층수 데이터도 같이 사용
        # 최근 데이터만 사용

        df = self.trade_price_with_similarity_apt_group(
            trg_date=trg_date,
            month_size=max_month_size,
            floor=floor,
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

    # ---------------------------------------------------------------------------------------------- #
    # 3. 거래량
    # ---------------------------------------------------------------------------------------------- #
    def training_volume_standard_area(self, trg_date: datetime, extent: float) -> float:
        # 면적별 거래량을 바탕으로 한 [거래량] feature
        trg_date = trg_date.strftime('%Y-%m') + '%'
        cursor = GinAptQuery.get_training_volume_standard_area(
            apt_detail_pk=self.apt_detail_pk,
            trg_date=trg_date
        )
        df = pd.DataFrame(
            cursor.fetchall(),
            columns=['date',
                     'area_20lt_trade_volume_cnt', 'area_20ge_30lt_trade_volume_cnt',
                     'area_30ge_40lt_trade_volume_cnt', 'area_40ge_50lt_trade_volume_cnt', 'area_50ge_trade_volume_cnt',
                     'standard_area_20lt_trade_volume_cnt', 'standard_area_20ge_30lt_trade_volume_cnt',
                     'standard_area_30ge_40lt_trade_volume_cnt', 'standard_area_40ge_50lt_trade_volume_cnt',
                     'standard_area_50ge_trade_volume_cnt']
        ).sum()

        try:
            if extent < 20:
                feature = df['area_20lt_trade_volume_cnt'] / df['standard_area_20lt_trade_volume_cnt']
            elif 20 <= extent < 30:
                feature = df['area_20ge_30lt_trade_volume_cnt'] / df['standard_area_20ge_30lt_trade_volume_cnt']
            elif 30 <= extent < 40:
                feature = df['area_30ge_40lt_trade_volume_cnt'] / df['standard_area_30ge_40lt_trade_volume_cnt']
            elif 40 <= extent < 50:
                feature = df['area_40ge_50lt_trade_volume_cnt'] / df['standard_area_40ge_50lt_trade_volume_cnt']
            else:
                feature = df['area_50ge_trade_volume_cnt'] / df['standard_area_50ge_trade_volume_cnt']
        except ZeroDivisionError:
            feature = 0
        except KeyError:
            feature = 0

        if feature == np.inf:
            feature = 0
        feature = float(feature)
        return feature

    def training_volume_standard_year(self, trg_date: datetime) -> float:
        cursor = settings.db_cursor
        cursor.execute("""
            SELECT gen_dt FROM GIN.apt_master
            WHERE idx=%s
        """, params=(self.apt_master_pk, ))

        cur_build_year = int(trg_date.strftime('%Y'))
        trg_build_year = int(cursor.fetchone()[0].split('.')[0])
        build_year = cur_build_year - trg_build_year

        cursor = GinAptQuery.get_training_volume_standard_year(self.apt_detail_pk, trg_date.strftime('%Y-%m')+'%')
        df = pd.DataFrame(
            cursor.fetchall(),
            columns=['date',
                     'year_05lt_trade_volume_cnt', 'year_05ge_10lt_trade_volume_cnt',
                     'year_10ge_15lt_trade_volume_cnt', 'year_15ge_25lt_trade_volume_cnt', 'year_25ge_trade_volume_cnt',
                     'standard_year_05lt_trade_volume_cnt', 'standard_year_05ge_10lt_trade_volume_cnt',
                     'standard_year_10ge_15lt_trade_volume_cnt', 'standard_year_15ge_25lt_trade_volume_cnt',
                     'standard_year_25ge_trade_volume_cnt']
        ).sum()

        try:
            if build_year < 5:
                feature = df['year_05lt_trade_volume_cnt'] / df['standard_year_05lt_trade_volume_cnt']
            elif 5 <= build_year < 10:
                feature = df['year_05ge_10lt_trade_volume_cnt'] / df['standard_year_05ge_10lt_trade_volume_cnt']
            elif 10 <= build_year < 15:
                feature = df['year_10ge_15lt_trade_volume_cnt'] / df['standard_year_10ge_15lt_trade_volume_cnt']
            elif 15 <= build_year < 25:
                feature = df['year_15ge_25lt_trade_volume_cnt'] / df['standard_year_15ge_25lt_trade_volume_cnt']
            else:
                feature = df['year_25ge_trade_volume_cnt'] / df['standard_year_25ge_trade_volume_cnt']
        except ZeroDivisionError:
            feature = 0
        except KeyError:
            feature = 0

        if feature == np.inf:
            feature = 0
        feature = float(feature)
        return feature


def make_feature(feature_name_list, apt_master_pk, apt_detail_pk, trade_cd,
                 trg_date, sale_month_size, sale_recent_month_size,
                 trade_month_size, trade_recent_month_size, floor, extent,
                 trade_pk=None):
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
                trg_date=trg_date, month_size=sale_month_size,
                floor=floor, extent=extent
            )

        elif feature_name == 'sale_price_with_floor_recent':
            df = feature.sale_price_with_floor_recent(
                trg_date=trg_date, max_month_size=sale_month_size, recent_month_size=sale_recent_month_size,
                floor=floor, extent=extent)

        elif feature_name == 'sale_price_with_floor_group':
            df = feature.sale_price_with_floor_group(
                trg_date=trg_date, month_size=sale_month_size,
                floor=floor, extent=extent)

        elif feature_name == 'sale_price_with_floor_group_recent':
            df = feature.sale_price_with_floor_group_recent(
                trg_date=trg_date, max_month_size=sale_month_size, recent_month_size=sale_recent_month_size,
                floor=floor, extent=extent)

        elif feature_name == 'sale_price_with_complex_group':
            df = feature.sale_price_with_complex_group(
                trg_date=trg_date, month_size=sale_month_size,
                floor=floor, extent=extent)

        elif feature_name == 'sale_price_with_complex_group_recent':
            df = feature.sale_price_with_complex_group_recent(
                trg_date=trg_date, max_month_size=sale_month_size, recent_month_size=sale_recent_month_size,
                floor=floor, extent=extent)

        elif feature_name == 'sale_price_with_similarity_apt_group':
            df = feature.sale_price_with_similarity_apt_group(
                trg_date=trg_date, month_size=sale_recent_month_size,
                floor=floor)

        elif feature_name == 'sale_price_with_similarity_apt_group_recent':
            df = feature.sale_price_with_similarity_apt_group_recent(
                trg_date=trg_date, max_month_size=sale_month_size, recent_month_size=sale_recent_month_size,
                floor=floor)

        elif feature_name == 'trade_price_with_floor':
            df = feature.trade_price_with_floor(
                trg_date=trg_date, month_size=trade_month_size,
                floor=floor, extent=extent, trade_pk=trade_pk)

        elif feature_name == 'trade_price_with_floor_recent':
            df = feature.trade_price_with_floor_recent(
                trg_date=trg_date, max_month_size=trade_month_size, recent_month_size=trade_recent_month_size,
                floor=floor, extent=extent, trade_pk=trade_pk)

        elif feature_name == 'trade_price_with_floor_group':
            df = feature.trade_price_with_floor_group(
                trg_date=trg_date, month_size=trade_month_size,
                floor=floor, extent=extent, trade_pk=trade_pk)

        elif feature_name == 'trade_price_with_floor_group_recent':
            df = feature.trade_price_with_floor_group_recent(
                trg_date=trg_date, max_month_size=trade_month_size, recent_month_size=trade_recent_month_size,
                floor=floor, extent=extent, trade_pk=trade_pk)

        elif feature_name == 'trade_price_with_complex_group':
            df = feature.trade_price_with_complex_group(
                trg_date=trg_date, month_size=trade_month_size,
                floor=floor, extent=extent, trade_pk=trade_pk)

        elif feature_name == 'trade_price_with_complex_group_recent':
            df = feature.trade_price_with_complex_group_recent(
                trg_date=trg_date, max_month_size=trade_month_size, recent_month_size=trade_recent_month_size,
                floor=floor, extent=extent, trade_pk=trade_pk)

        elif feature_name == 'trade_price_with_similarity_apt_group':
            df = feature.trade_price_with_similarity_apt_group(
                trg_date=trg_date, month_size=trade_month_size,
                floor=floor, trade_pk=trade_pk)

        elif feature_name == 'trade_price_with_similarity_apt_group_recent':
            df = feature.trade_price_with_similarity_apt_group_recent(
                trg_date=trg_date, max_month_size=trade_month_size, recent_month_size=trade_recent_month_size,
                floor=floor, trade_pk=trade_pk)

        elif feature_name == 'training_volume_standard_area':
            df = feature.training_volume_standard_area(
                trg_date=trg_date, extent=extent
            )

        elif feature_name == 'training_volume_standard_year':
            df = feature.training_volume_standard_year(
                trg_date=trg_date
            )

        if feature_name in settings.training_volume_feature:
            # 거래량 데이터를 위한 예외 처리
            value = df
        elif len(df) != 0:
            value = np.average(df.price)
        else:
            value = np.nan
        features.append(value)
    feature_df = pd.DataFrame([features], columns=feature_name_list)

    # 매물 데이터 빈 데이터 추합
    sale_feature_df = feature_df[settings.sale_features]
    sale_feature_mean = float(sale_feature_df.dropna(axis=1).mean(axis=1))
    sale_feature_df = sale_feature_df.fillna(sale_feature_mean)

    # 매매 데이터 빈 데이터 추합
    trade_feature_df = feature_df[settings.trade_features]
    trade_feature_mean = float(trade_feature_df.dropna(axis=1).mean(axis=1))
    trade_feature_df = trade_feature_df.fillna(trade_feature_mean, axis=0)

    trade_volume_feature_df = feature_df[settings.training_volume_feature]

    feature_df = pd.concat([sale_feature_df, trade_feature_df, trade_volume_feature_df], axis=1)

    status = None
    if not np.isnan(sale_feature_mean) and not np.isnan(trade_feature_mean):
        status = settings.full_feature_model_name
        feature_df = pd.concat([sale_feature_df, trade_feature_df, trade_volume_feature_df], axis=1)
    elif not np.isnan(sale_feature_mean):
        status = settings.sale_feature_model_name
        feature_df = pd.concat([sale_feature_df, trade_volume_feature_df], axis=1)
    elif not np.isnan(trade_feature_mean):
        status = settings.trade_feature_model_name
        feature_df = pd.concat([trade_feature_df, trade_volume_feature_df], axis=1)

    if status is None:
        raise FeatureExistsError()

    return {
        'status': status,
        'data': feature_df
    }


class FeatureExistsError(Exception):
    def __init__(self):
        super().__init__('매매, 매물 데이터가 존재하지 않음')