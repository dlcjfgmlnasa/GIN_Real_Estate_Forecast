# -*- coding:utf-8 -*-
import os
import time
import argparse
import settings
import numpy as np
import pandas as pd
from datetime import datetime
from datedelta import datedelta
from database import GinAptQuery
from sklearn.externals import joblib
from feature import FeatureExistsError, optimized_make_feature, make_feature
import matplotlib.pyplot as plt
import matplotlib.dates as dates
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def get_args():
    parser = argparse.ArgumentParser()
    # target information
    parser.add_argument('--full_pk', action='store_true')
    parser.add_argument('--full_date', action='store_true')
    parser.add_argument('--db_inject', action='store_true')
    parser.add_argument('--evaluation', action='store_true')
    parser.add_argument('--evaluation_plot', action='store_true')

    parser.add_argument('--apt_detail_pk', type=int, default=3)
    parser.add_argument('--date', default=datetime.now().strftime('%Y-%m-%d'))

    # feature information
    parser.add_argument('--feature_list', type=list, default=settings.features)
    parser.add_argument('--sale_month_size', type=int, default=settings.sale_month_size)
    parser.add_argument('--sale_recent_month_size', type=int, default=settings.sale_recent_month_size)
    parser.add_argument('--trade_month_size', type=int, default=settings.trade_month_size)
    parser.add_argument('--trade_recent_month_size', type=int, default=settings.trade_recent_month_size)
    parser.add_argument('--model_info', type=str, default=settings.model_info)
    parser.add_argument('--previous_month_size', type=int, default=settings.predicate_previous_month_size)
    parser.add_argument('--feature_engine', type=str, default='optimizer', choices=['default', 'optimizer'])
    parser.add_argument('--trade_cd', type=str, choices=['t', 'r'], default=settings.trade_cd)
    return parser.parse_args()


def get_month_range(trg_date: datetime, previous_month_size: int):
    time_format = '%Y-%m-%d'
    end_date = datetime.strptime(trg_date, time_format)
    start_date = end_date - datedelta(months=previous_month_size)
    date_range = pd.date_range(start=start_date, end=end_date)
    date_range = [date.strftime(time_format) for date in date_range]
    return date_range


def transformer_floor(apt_detail_pk: int, floor: int):
    low_floor = ['저', '저층']
    middle_floor = ['중', '중층', '-']
    high_floor = ['고', '고층']

    if floor in low_floor:
        return 0
    elif floor in middle_floor:
        return 4
    elif floor in high_floor:
        max_floor = GinAptQuery.get_max_floor(apt_detail_pk).fetchone()[0]
        return max_floor
    else:
        return floor


def get_model(model_info=settings.model_info):
    models = {}
    for model_name, model_path in model_info.items():
        # Loading model...
        model = joblib.load(model_path)
        models[model_name] = model
    return models


class AptPredicate(object):
    # 아파트 예측을 위한 통합 클래스
    def __init__(self, apt_detail_pk, models, feature_engine,
                 feature_list=settings.features, previous_month_size=settings.predicate_previous_month_size,
                 sale_month_size=settings.sale_month_size, sale_recent_month_size=settings.sale_recent_month_size,
                 trade_month_size=settings.trade_month_size, trade_recent_month_size=settings.trade_recent_month_size,
                 trade_cd=settings.trade_cd):
        super().__init__()
        self.apt_detail_pk = apt_detail_pk
        self.models = models
        self.feature_engine = feature_engine
        self.feature_list = feature_list
        self.previous_month_size = previous_month_size
        self.sale_month_size = sale_month_size
        self.sale_recent_month_size = sale_recent_month_size
        self.trade_month_size = trade_month_size
        self.trade_recent_month_size = trade_recent_month_size
        self.trade_cd = trade_cd

    def predicate(self, date):
        # 현재 시장에 나와있는 아직 팔리지 않는 매물들을 바탕으로 예측을 실시
        month_range = get_month_range(date, self.previous_month_size)

        # 현재 시장에 나와있는 매물데이터 수집
        new_trade_list = GinAptQuery.get_new_apt_trade_list(
            apt_detail_pk=self.apt_detail_pk,
            date_range='","'.join(month_range),
            trade_cd=self.trade_cd
        ).fetchall()

        if len(new_trade_list) == 0:
            # 만약 매물 데이터가 존재하지 않을시 매매 데이터를 활용
            month_df = pd.DataFrame([month.split('-') for month in month_range],
                                    columns=['year', 'month', 'day'])
            date_sql_df = month_df.T.apply(lambda x: '(year = {0} AND mon = {1} AND real_day = {2})'.format(
                x.year, x.month, x.day))
            date_sql = ' OR '.join(date_sql_df.values)

            new_trade_df = pd.DataFrame(
                GinAptQuery.get_trade_price_with_sql_date(
                    apt_detail_pk=self.apt_detail_pk,
                    trade_cd=self.trade_cd,
                    date_sql=date_sql).fetchall(),
                columns=['master_idx', 'pk_apt_detail', 'year', 'month', 'day', 'floor', 'extent', 'price']
            )

            date_list = [datetime.strptime(f'{year}-{month}-{day}', '%Y-%m-%d') for year, month, day in
                         zip(new_trade_df.year, new_trade_df.month, new_trade_df.day)]
            del new_trade_df['year'], new_trade_df['month'], new_trade_df['day']
            new_trade_df['date'] = date_list
            new_trade_df = new_trade_df[['master_idx', 'pk_apt_detail', 'date', 'floor', 'extent', 'price']]
            new_trade_list = new_trade_df.values

        # 예측에 필요한 매매, 매물 데이터가 전혀 존재하지 않을떄...
        if len(new_trade_list) == 0:
            raise FeatureExistsError()

        # 매매, 매물 데이터 리스트 중 일부 select
        new_trade_list = self.__select_trade_list(new_trade_list)

        # making feature...
        total_feature = {
            settings.full_feature_model_name: [],
            settings.sale_feature_model_name: [],
            settings.trade_feature_model_name: []
        }
        apt_extent = None
        for apt_master_pk, apt_detail_pk, date, floor, extent, price in new_trade_list:
            apt_extent = float(extent)
            floor = transformer_floor(apt_detail_pk, floor)
            try:
                features = self.feature_engine(feature_name_list=self.feature_list, apt_master_pk=apt_master_pk,
                                               apt_detail_pk=apt_detail_pk, trade_cd=self.trade_cd, trg_date=date,
                                               sale_month_size=self.sale_month_size,
                                               sale_recent_month_size=self.sale_recent_month_size,
                                               trade_month_size=self.trade_month_size,
                                               trade_recent_month_size=self.trade_recent_month_size,
                                               floor=floor, extent=extent)
            except FeatureExistsError:
                # 매매 혹은 매물 데이터를 바탕으로한 feature 하나도 존재하지 않을때...
                continue

            status = features['status']
            data = features['data']
            total_feature[status].append(data)

        predication_list = []
        feature_name = None
        for feature_name, data in total_feature.items():
            if len(data) == 0:
                continue
            else:
                df = pd.concat(data)
                feature_df = df.reset_index(drop=True).astype(np.float)
                # Get model...
                model = self.models[feature_name]
                predication = model.predict(feature_df) * apt_extent
                predication_list.append(predication)
                break

        # 예측에 필요한 매매, 매물 데이터가 전혀 존재하지 않을떄...
        if len(predication_list) == 0:
            raise FeatureExistsError()

        # Predication max, Predicate min, Predicate avg
        result = {
            'predicate_price_max': np.max(predication_list),
            'predicate_price_min': np.min(predication_list),
            'predicate_price_avg': np.average(predication_list),
            'select_model': feature_name
        }
        return result

    def predicate_full(self):
        # 실제 매매 데이터 정보
        real_price_df = pd.DataFrame(
            GinAptQuery.get_trade_price_simple(
                apt_detail_pk=self.apt_detail_pk,
                trade_cd=self.trade_cd
            ).fetchall(),
            columns=['year', 'month', 'day', 'price']
        )
        real_price_df['date'] = \
            real_price_df.T.apply(lambda x: datetime.strptime(
                '{0}-{1:02d}-{2:02d}'.format(int(x.year), int(x.month), int(x.day)),
                '%Y-%m-%d'
            ))

        # 전체 DataFrame 생성
        start_date = str(real_price_df.date.values[0]).split('T')[0]
        end_date = str(real_price_df.date.values[-1]).split('T')[0]

        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        df = pd.DataFrame(date_range, columns=['date'])

        real_value = []
        for date in df.date:
            price = real_price_df[date == real_price_df.date].price.values
            if len(price) == 0:
                real_value.append(np.nan)
            else:
                real_value.append(float(price[0]))
        df['real_price'] = real_value

        # 전체 예측 실시
        predicate_price_avg = []
        predicate_price_max = []
        predicate_price_min = []

        for i, trg_date in enumerate(df.date.values):
            trg_date = str(trg_date).split('T')[0]

            day = int(trg_date.split('-')[-1])
            if i != 0 and i != len(df) - 1 and day not in [1, 10, 20, 30]:
                predicate_price_avg.append(np.nan)
                predicate_price_max.append(np.nan)
                predicate_price_min.append(np.nan)
                continue
            # Predicate....
            try:
                each_start_time = time.time()
                predicate_result = self.predicate(date=trg_date,)
                each_end_time = time.time()

                each_predicate_time = each_end_time - each_start_time

                print('date : {0:} \t predicate_time : {1:.2f}sec \t select_model : {2:s} \t pred_avg : {3:.4f}'.format(
                    trg_date, each_predicate_time,
                    predicate_result['select_model'], predicate_result['predicate_price_avg']))

                predicate_price_avg.append(predicate_result['predicate_price_avg'])
                predicate_price_max.append(predicate_result['predicate_price_max'])
                predicate_price_min.append(predicate_result['predicate_price_min'])

            except FeatureExistsError:
                predicate_price_avg.append(np.nan)
                predicate_price_max.append(np.nan)
                predicate_price_min.append(np.nan)

        # interpolate
        df['predicate_price_max'] = predicate_price_max
        df['predicate_price_avg'] = predicate_price_avg
        df['predicate_price_min'] = predicate_price_min
        df['predicate_price_max'] = df['predicate_price_max'].interpolate()
        df['predicate_price_avg'] = df['predicate_price_avg'].interpolate()
        df['predicate_price_min'] = df['predicate_price_min'].interpolate()
        return df

    def predicate_transform_range(self, df):
        predicate_price_max_range = df.predicate_price_avg.values * (1 + 0.06)
        predicate_price_min_range = df.predicate_price_avg.values * (1 - 0.06)

        # transform range
        df['predicate_price_max'] = self.__transform_range(df.predicate_price_max.values,
                                                           predicate_price_max_range, t='max')
        df['predicate_price_min'] = self.__transform_range(df.predicate_price_min.values,
                                                           predicate_price_min_range, t='min')
        # smoothing
        df['predicate_price_max'] = self.__smooth_triangle(df['predicate_price_max'].values, 60)
        df['predicate_price_avg'] = self.__smooth_triangle(df['predicate_price_avg'].values, 60)
        df['predicate_price_min'] = self.__smooth_triangle(df['predicate_price_min'].values, 60)
        return df

    def predicate_full_evaluation(self, plot=True):
        # predicate evaluation & plot
        # Predicate & Transformer Range
        df = self.predicate_full()
        df = self.predicate_transform_range(df)

        # evaluation
        evaluation_df = df[~np.isnan(df.real_price)]
        evaluation_df['evaluation'] = evaluation_df.T.apply(
            lambda x: x.predicate_price_min <= x.real_price <= x.predicate_price_max)
        accuracy = float(np.average(evaluation_df.evaluation))

        if plot:
            # Plot Show
            years = dates.YearLocator()  # every year
            months = dates.MonthLocator()  # every month
            years_fmt = dates.DateFormatter('%Y')

            fig, ax = plt.subplots()
            ax.scatter(df.date, df.real_price, marker='x', color='orange', label='price')
            ax.plot(df.date, df.predicate_price_max, color='r', label='max')
            ax.plot(df.date, df.predicate_price_avg, color='b', label='avg')
            ax.plot(df.date, df.predicate_price_min, color='g', label='min')

            ax.xaxis.set_major_locator(years)
            ax.xaxis.set_major_formatter(years_fmt)
            ax.xaxis.set_minor_locator(months)

            data_min = np.datetime64(df.date.values[0], 'Y')
            date_max = np.datetime64(df.date.values[-1], 'Y') + np.timedelta64(1, 'Y')
            ax.set_xlim(data_min, date_max)

            ax.format_xdata = dates.DateFormatter('%Y-%m-%d')
            ax.grid(True)

            fig.autofmt_xdate()
            plt.title('pk : {0} / accuracy : {1:.2f}'.format(self.apt_detail_pk, accuracy))
            plt.legend(loc='upper left')
            return accuracy, plt
        return accuracy

    @staticmethod
    def __transform_range(values, range_values, t='max'):
        predicate_values = []
        for smoothing_value, range_value in zip(values, range_values):
            if t == 'max':
                if smoothing_value >= range_value:
                    predicate_values.append(smoothing_value)
                else:
                    predicate_values.append(range_value)
            else:
                if smoothing_value <= range_value:
                    predicate_values.append(smoothing_value)
                else:
                    predicate_values.append(range_value)
        return predicate_values

    @staticmethod
    def __smooth_triangle(data, degree):
        triangle = np.concatenate((np.arange(degree + 1), np.arange(degree)[::-1]))
        smoothed = []

        for i in range(degree, len(data) - degree * 2):
            point = data[i:i + len(triangle)] * triangle
            smoothed.append(np.sum(point) / np.sum(triangle))
        # Handle boundaries
        smoothed = [smoothed[0]] * int(degree + degree / 2) + smoothed
        while len(smoothed) < len(data):
            smoothed.append(smoothed[-1])
        return smoothed

    @staticmethod
    def __select_trade_list(trade_list):
        # 매매, 매물 데이터 리스트 중 Selection
        trade_df = pd.DataFrame(
            trade_list,
            columns=['master_idx', 'pk_apt_detail', 'date', 'floor', 'extent', 'price']
        )

        new_trade_list = []
        for _, group in trade_df.groupby(['floor', 'extent', 'price']):
            new_trade_list.append(group.iloc[-1:])
        new_trade_df = pd.concat(new_trade_list)
        new_trade_df = new_trade_df.sort_values('date')

        new_trade_list = []
        for _, group in new_trade_df.groupby(['floor', 'extent']):
            standard_price = float(group[-1:].price)
            standard_price_max = standard_price * (1 + settings.predicate_standard_price_rate)
            standard_price_min = standard_price * (1 - settings.predicate_standard_price_rate)

            group = group[group.price.apply(lambda price: standard_price_min <= price <= standard_price_max)]
            price_mean = group.price.mean()
            df = group.copy()
            df.price = price_mean
            new_trade_list.append(df.iloc[-1:])

        new_trade_df = pd.concat(new_trade_list)
        new_trade_df = new_trade_df.sort_values('date')
        new_trade_list = new_trade_df.values
        return new_trade_list


def get_feature_engine(name):
    if name == 'default':
        return make_feature
    elif name == 'optimizer':
        return optimized_make_feature
    else:
        raise NotImplemented()


if __name__ == '__main__':
    argument = get_args()                               # Get arguments

    f_e = get_feature_engine(argument.feature_engine)   # Get feature engine
    m = get_model(argument.model_info)                  # Get model (full.model, sale.model, trade.model)

    pk_list = [pk[0] for pk in GinAptQuery.get_predicate_apt_list().fetchall()]

    if argument.full_pk and argument.full_date:
        # [전체 아파트 리스트]를 [처음부터 끝]까지 아파트 가격 예측
        for detail_pk in pk_list:
            try:
                regression = AptPredicate(apt_detail_pk=detail_pk, models=m,
                                          feature_engine=f_e, feature_list=argument.feature_list,
                                          previous_month_size=argument.previous_month_size,
                                          sale_month_size=argument.sale_month_size,
                                          sale_recent_month_size=argument.sale_recent_month_size,
                                          trade_month_size=argument.trade_month_size,
                                          trade_recent_month_size=argument.trade_recent_month_size,
                                          trade_cd=argument.trade_cd)
                _result_df = regression.predicate_full()
                _result_df_smooth = regression.predicate_transform_range(_result_df)
                print(_result_df_smooth)

                # Database injection
                if argument.db_inject:
                    # TODO: Database Injection
                    pass
            except FeatureExistsError:
                pass

    elif argument.full_pk:
        # [전체 아파트 리스트]를 [지정한 날짜]의 아파트 가격 예측
        for detail_pk in pk_list:
            try:
                regression = AptPredicate(apt_detail_pk=detail_pk, models=m,
                                          feature_engine=f_e, feature_list=argument.feature_list,
                                          previous_month_size=argument.previous_month_size,
                                          sale_month_size=argument.sale_month_size,
                                          sale_recent_month_size=argument.sale_recent_month_size,
                                          trade_month_size=argument.trade_month_size,
                                          trade_recent_month_size=argument.trade_recent_month_size,
                                          trade_cd=argument.trade_cd)
                _result = regression.predicate(date=argument.date)
                print('price_min : {0:.4f}   price_avg : {1:.4f}   price_max : {2:.4f}'.format(
                    _result['predicate_price_min'], _result['predicate_price_avg'], _result['predicate_price_max']
                ))

                # Database injection
                if argument.db_inject:
                    # TODO: Database Injection
                    pass

            except FeatureExistsError:
                pass

    elif argument.full_date and argument.apt_detail_pk:
        # [apt_detail_pk]의 [처음부터 끝]까지 아파트 가격 예측
        regression = AptPredicate(apt_detail_pk=argument.apt_detail_pk, models=m,
                                  feature_engine=f_e, feature_list=argument.feature_list,
                                  previous_month_size=argument.previous_month_size,
                                  sale_month_size=argument.sale_month_size,
                                  sale_recent_month_size=argument.sale_recent_month_size,
                                  trade_month_size=argument.trade_month_size,
                                  trade_recent_month_size=argument.trade_recent_month_size,
                                  trade_cd=argument.trade_cd)
        # TODO : Test Code
        _result_df = pd.read_csv('test.csv')
        del _result_df['Unnamed: 0']
        del _result_df['real_price']
        _result_df['apt_detail_pk'] = argument.apt_detail_pk
        _result_df_smooth = regression.predicate_transform_range(_result_df)
        _result_df_smooth['apt_detail_pk'] = argument.apt_detail_pk

        # _result_df = regression.predicate_full()
        # _result_df.to_csv('test.csv')
        # _result_df_smooth = regression.predicate_transform_range(_result_df)
        # print(_result_df_smooth)

        # Database injection
        if argument.db_inject:
            # TODO: Database Injection
            print(_result_df)
            print(_result_df_smooth)
            pass

    elif argument.evaluation:
        # [apt_detail_pk]의 [evaluation]
        regression = AptPredicate(apt_detail_pk=argument.apt_detail_pk, models=m,
                                  feature_engine=f_e, feature_list=argument.feature_list,
                                  previous_month_size=argument.previous_month_size,
                                  sale_month_size=argument.sale_month_size,
                                  sale_recent_month_size=argument.sale_recent_month_size,
                                  trade_month_size=argument.trade_month_size,
                                  trade_recent_month_size=argument.trade_recent_month_size,
                                  trade_cd=argument.trade_cd)
        _result = regression.predicate_full_evaluation(plot=argument.evaluation_plot)
        print(_result)

        if argument.evaluation_plot:
            image_name = '{}.png'.format(argument.apt_detail_pk)
            image_path = os.path.join(settings.image_path, image_name)
            _result, img = _result
            print(_result)
            img.savefig(image_path)     # Saving Image
            img.show()

    else:
        # [apt_detail_pk]의 [지정한 날짜]의 아파트 가격 예측
        regression = AptPredicate(apt_detail_pk=argument.apt_detail_pk, models=m,
                                  feature_engine=f_e, feature_list=argument.feature_list,
                                  previous_month_size=argument.previous_month_size,
                                  sale_month_size=argument.sale_month_size,
                                  sale_recent_month_size=argument.sale_recent_month_size,
                                  trade_month_size=argument.trade_month_size,
                                  trade_recent_month_size=argument.trade_recent_month_size,
                                  trade_cd=argument.trade_cd)
        _result = regression.predicate(date=argument.date)
        print('price_min : {0:.4f}   price_avg : {1:.4f}   price_max : {2:.4f}'.format(
            _result['predicate_price_min'], _result['predicate_price_avg'], _result['predicate_price_max']
        ))

        # Database injection
        if argument.db_inject:
            # TODO: Database Injection
            pass
