# -*- coding:utf-8 -*-
import pandas as pd
from datetime import datetime, timedelta
from datedelta import datedelta
from database import GinAptQuery


class AptPriceRegressionFeature(object):
    # 아파트 매매가격 예측을 위한 피쳐 클래스
    def __init__(self, apt_master_pk: int, apt_detail_pk: int, trade_cd: str):
        self.apt_master_pk = apt_master_pk
        self.apt_detail_pk = apt_detail_pk
        self.trade_cd = trade_cd
        self.query = GinAptQuery()

    @staticmethod
    def _get_price_recent(frame, recent_month_size):
        recent_price = frame.iloc[-1:]
        recent_date = list(recent_price.date)[0]
        print(recent_date)
        # exit()
        # recent_date = list(recent_sale_price.date)[0]
        # recent_pre_date = recent_date - datedelta(months=recent_month_size)
        # recent_date_range = pd.date_range(recent_pre_date, recent_date)
        # df = frame[frame.date.apply(lambda date: date in recent_date_range)]
        return frame

    def previous_time_sale_price_with_floor(self, trg_date: datetime, month_size: int, floor: int, extent: float):
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
        return df

    def previous_time_sale_price_with_floor_recent(self, trg_date: datetime, max_month_size: int,
                                                   recent_month_size: int, floor: int, extent: float):
        # 예측하고자하는 층의 이전 시간대의 [매물가격]을 이용한 feature
        # max_month_size 안에서 가장 최근 데이터를 기준으로 recent_month_size 만큼의 [매물가격] 출력
        # ex) apt detail pk 가 1인 아파트 8층 건물의 가장 최근 [매물가격정보]를 이용하고 싶다.
        df = self.previous_time_sale_price_with_floor(
            trg_date=trg_date,
            month_size=max_month_size,
            floor=floor,
            extent=extent
        )
        print(df)
        if len(df) != 0:
            df = self._get_price_recent(
                frame=df,
                recent_month_size=recent_month_size
            )
        return df

    def previous_time_trade_price_with_floor(self, trade_pk: int, trg_date: datetime,
                                             month_size: int, floor: int, extent: float):
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

        # Train 을 위해 trade_pk 값은 제외 시킴
        df = df[df.pk_apt_trade.apply(lambda pk: pk != trade_pk)]
        df = df[['apt_detail_pk', 'date', 'floor', 'price']]
        return df

    def previous_time_trade_price_with_floor_recent(self, trade_pk: int, trg_date: datetime,
                                                    max_month_size: int, recent_month_size: int,
                                                    floor: int, extent: float):
        # 예측하고자하는 층의 이전 시간대의 [매매가격]을 이용한 feature
        # max_month_size 안에서 가장 최근 데이터를 기준으로 recent_month_size 만큼의 [매매가격] 출력
        # ex) apt detail pk 가 1인 아파트 8층 건물의 가장 최근 [매매가격정보]를 이용하고 싶다.
        df = self.previous_time_trade_price_with_floor(
            trade_pk=trade_pk,
            trg_date=trg_date,
            month_size=max_month_size,
            floor=floor,
            extent=extent
        )

        if len(df) != 0:
            df = self._get_price_recent(
                frame=df,
                recent_month_size=recent_month_size
            )
            print(df)


def make_feature():
    pass
