# -*- coding:utf-8 -*-
import os
import copy
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
    parser.add_argument('--full_pk', action='store_true',
                        help='대상 아파트 전체 예측')

    parser.add_argument('--full_date', action='store_true',
                        help='2006년도부터 현재까지 예측')

    parser.add_argument('--db_inject', action='store_true',
                        help='mysql database injection')

    parser.add_argument('--evaluation', action='store_true',
                        help='대상 아파트 정확도 평가')

    parser.add_argument('--evaluation_plot', action='store_true',
                        help='대상 아파트 정확도 시각화 및 저장 (setting.py에 있는 image_path 값을 참조하여 저장)')

    parser.add_argument('--apt_detail_pk', action='store', type=int,
                        help='예측 대상 아파트 pk')

    parser.add_argument('--date', action='store', type=str,
                        default=datetime.now().strftime('%Y-%m-%d'),
                        help='예측하고 싶은 날짜 ex) 2018-01-01 (default : 현재 날짜)')

    # feature information
    parser.add_argument('--feature_list',
                        type=list,
                        default=settings.features,
                        help='예측에 필요한 feature (default: setting.py에 있는 features 참조)')

    parser.add_argument('--sale_month_size',
                        type=int,
                        default=settings.sale_month_size,
                        help='예측시 사용될 매물 데이터 크기 (default: setting.py에 있는 sale_month_size 참조)')

    parser.add_argument('--sale_recent_month_size',
                        type=int,
                        default=settings.sale_recent_month_size,
                        help='예측시 사용될 최근 매물 데이터 크기 (default: setting.py에 있는 sale_recent_month_size 참조')

    parser.add_argument('--trade_month_size',
                        type=int,
                        default=settings.trade_month_size,
                        help='예측시 사용될 매매 데이터 크기 (default: setting.py에 있는 trade_month_size 참조)')

    parser.add_argument('--trade_recent_month_size',
                        type=int,
                        default=settings.trade_recent_month_size,
                        help='예측시 사용될 최근 매매 데이터 크기 (default: setting.py에 있는 trade_recent_month_size 참조)')

    parser.add_argument('--model_info',
                        type=str,
                        default=settings.model_info,
                        help='모델 위치 정보 (default: setting.py에 있는 model_info)')

    parser.add_argument('--previous_month_size', type=int, default=settings.predicate_previous_month_size,
                        help='예측시 사용하는 과거 매물&매매 사이즈 '
                             '(default: setting.py에 있는 predicate_previous_month_size')

    parser.add_argument('--feature_engine',
                        type=str,
                        default='optimizer',
                        choices=['default', 'optimizer'],
                        help='feature engineering 을 하는 방법')

    parser.add_argument('--trade_cd',
                        type=str,
                        choices=['t', 'r'],
                        default=settings.trade_cd,
                        help='t : 아파트 매매가격 추정 / r: 아파트 전월세가격 추정')

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

    def predicate_full(self, evaluation=False):
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

        start_date = str(real_price_df.date.values[0]).split('T')[0]

        predicate_price_avg = []
        predicate_price_max = []
        predicate_price_min = []
        if evaluation:
            end_date = str(real_price_df.date.values[-1]).split('T')[0]

            # 전체 DataFrame 생성
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
            for i, trg_date in enumerate(df.date.values):
                trg_date = str(trg_date).split('T')[0]

                day = int(trg_date.split('-')[-1])
                if i != 0 and i != len(df) - 1 and day not in [1, 11, 21]:
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

                    print('date : {0:} \t predicate_time : {1:.2f}sec \t select_model : {2:s} \t pred_avg : {3:.4f}'.
                          format(trg_date, each_predicate_time, predicate_result['select_model'],
                                 predicate_result['predicate_price_avg']))

                    predicate_price_avg.append(predicate_result['predicate_price_avg'])
                    predicate_price_max.append(predicate_result['predicate_price_max'])
                    predicate_price_min.append(predicate_result['predicate_price_min'])

                except FeatureExistsError:
                    predicate_price_avg.append(np.nan)
                    predicate_price_max.append(np.nan)
                    predicate_price_min.append(np.nan)

            df['predicate_price_max'] = predicate_price_max
            df['predicate_price_avg'] = predicate_price_avg
            df['predicate_price_min'] = predicate_price_min

            # interpolate
            df['predicate_price_max'] = df['predicate_price_max'].interpolate()
            df['predicate_price_avg'] = df['predicate_price_avg'].interpolate()
            df['predicate_price_min'] = df['predicate_price_min'].interpolate()
            return df
        else:
            end_date = datetime.now().strftime('%Y-%m-%d')
            # 전체 DataFrame 생성
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            date_temp = []
            for trg_date in date_range:
                trg_date = str(trg_date).split()[0]
                day = int(trg_date.split('-')[-1])
                if day in [1, 11, 21]:
                    date_temp.append(trg_date)
                    # Predicate....
                    try:
                        each_start_time = time.time()
                        predicate_result = self.predicate(date=trg_date, )
                        each_end_time = time.time()

                        each_predicate_time = each_end_time - each_start_time

                        print('date : {0:} \t predicate_time : {1:.2f}sec \t select_model : {2:s} \t '
                              'pred_avg : {3:.4f}'.format(trg_date, each_predicate_time,
                                                          predicate_result['select_model'],
                                                          predicate_result['predicate_price_avg']))

                        predicate_price_avg.append(predicate_result['predicate_price_avg'])
                        predicate_price_max.append(predicate_result['predicate_price_max'])
                        predicate_price_min.append(predicate_result['predicate_price_min'])

                    except FeatureExistsError:
                        predicate_price_avg.append(np.nan)
                        predicate_price_max.append(np.nan)
                        predicate_price_min.append(np.nan)
            df = pd.DataFrame()
            df['date'] = date_temp

        df['predicate_price_max'] = predicate_price_max
        df['predicate_price_avg'] = predicate_price_avg
        df['predicate_price_min'] = predicate_price_min

        # interpolate
        df['predicate_price_max'] = df['predicate_price_max'].interpolate()
        df['predicate_price_avg'] = df['predicate_price_avg'].interpolate()
        df['predicate_price_min'] = df['predicate_price_min'].interpolate()
        return df

    @staticmethod
    def predicate_transform_range(df):
        df = copy.deepcopy(df)
        predicate_price_max_range = df.predicate_price_avg.values * (1 + 0.06)
        predicate_price_min_range = df.predicate_price_avg.values * (1 - 0.06)

        # transform range
        df['predicate_price_max'] = AptPredicate.__transform_range(df.predicate_price_max.values,
                                                                   predicate_price_max_range, t='max')
        df['predicate_price_min'] = AptPredicate.__transform_range(df.predicate_price_min.values,
                                                                   predicate_price_min_range, t='min')
        # smoothing
        df['predicate_price_max'] = AptPredicate.__smooth_triangle(df['predicate_price_max'].values, 10)
        df['predicate_price_avg'] = AptPredicate.__smooth_triangle(df['predicate_price_avg'].values, 10)
        df['predicate_price_min'] = AptPredicate.__smooth_triangle(df['predicate_price_min'].values, 10)
        return df

    def predicate_full_evaluation(self, plot=True):
        # predicate evaluation & plot
        # Predicate & Transformer Range
        df = self.predicate_full(evaluation=True)
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

    pk_list = [pk[1] for pk in GinAptQuery.get_predicate_apt_list().fetchall()]

    def formatting_df(_df, columns, _apt_detail_pk):
        # formatting pandas frame for inject database
        _df.columns = columns
        _df['apt_detail_pk'] = _apt_detail_pk
        _df['trade_cd'] = argument.trade_cd
        _df = _df.dropna(axis=0)
        return _df

    price_predicate_columns = ['reg_date', 'price_max', 'price_avg', 'price_min']
    price_smoothing_predicate_columns = ['reg_date', 'price_max_smoothing', 'price_avg_smoothing',
                                         'price_min_smoothing']

    # -------------------------------------------------------------------------------------------------------------- #
    # command :
    #       - python predicate.py --full_pk --full_date
    #
    # explanation : [전체 아파트 리스트]를 [처음부터 끝]까지 가격 예측
    # option: --db_inject 을 추가하면 mysql 에 예측된 결과 저장
    # -------------------------------------------------------------------------------------------------------------- #
    if argument.full_pk and argument.full_date:
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

                # Database injection
                if argument.db_inject:
                    _result_df = formatting_df(_result_df, price_predicate_columns, detail_pk)
                    _result_df_smooth = formatting_df(_result_df_smooth, price_smoothing_predicate_columns, detail_pk)
                    # database injection
                    GinAptQuery.insert_or_update_predicate_value(list(_result_df.values))
                    GinAptQuery.insert_or_update_predicate_smoothing_value(list(_result_df_smooth.values))
                    print(f'{detail_pk} pk data - Database Injection')
            except FeatureExistsError:
                pass

        print('Complete Database Injection')

    # -------------------------------------------------------------------------------------------------------------- #
    # Command :
    #       - python predicate.py --full_pk                 [현재 날짜를 예측]
    #       - python predicate.py --full_pk --date={날짜}    [지정된 날짜를 예측]
    #           + ex) python predicate.py --full_pk --date='2018-01-01'
    #
    # Explanation : [전체 아파트 리스트]를 [지정한 날짜]의 가격 예측
    # Reference: --db_inject 을 추가하면 mysql 에 예측된 결과 저장
    # -------------------------------------------------------------------------------------------------------------- #
    elif argument.full_pk:
        temp = []
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

                if argument.db_inject:
                    # Store data for Database injection
                    temp.append([argument.date,
                                 _result['predicate_price_min'],
                                 _result['predicate_price_avg'],
                                 _result['predicate_price_max'],
                                 detail_pk,
                                 argument.trade_cd])
            except FeatureExistsError:
                pass

        # Database injection
        if argument.db_inject:
            # Integrated frame
            price_predicate_columns.extend(['apt_detail_pk', 'trade_cd'])
            _result_df = pd.DataFrame(temp, columns=price_predicate_columns)

            # Database injection (price_min, price_max, price_avg)
            GinAptQuery.insert_or_update_predicate_value(list(_result_df.values))

            # Database injection (price_min_smoothing, price_max_smoothing, price_avg_smoothing)
            for detail_pk in _result_df.apt_detail_pk:
                db_to_frame = pd.read_sql_query(f"SELECT * FROM predicate_price WHERE apt_detail_pk={detail_pk};",
                                                settings.cnx)
                db_to_frame = db_to_frame[['reg_date', 'price_min', 'price_avg', 'price_max']]
                db_to_frame.columns = ['date', 'predicate_price_min', 'predicate_price_avg', 'predicate_price_max']

                db_frame_smoothing = AptPredicate.predicate_transform_range(db_to_frame)
                db_frame_smoothing = formatting_df(db_frame_smoothing, price_smoothing_predicate_columns, detail_pk)
                GinAptQuery.insert_or_update_predicate_smoothing_value(list(db_frame_smoothing.values))

            print('Complete Database Injection')

    # -------------------------------------------------------------------------------------------------------------- #
    # Command :
    #       - python predicate.py --full_date --apt_detail_pk={아파트 pk}
    #           + ex) python predicate.py --full_date --apt_detail_pk=1
    #
    # Explanation : [apt_detail_pk]의 전체 날짜에 대한 가격 예측
    # Reference: --db_inject 을 추가하면 mysql 에 예측된 결과 저장
    # -------------------------------------------------------------------------------------------------------------- #
    elif argument.full_date and argument.apt_detail_pk:
        regression = AptPredicate(apt_detail_pk=argument.apt_detail_pk, models=m,
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

        if argument.db_inject:
            # formatting for database injection
            _result_df = formatting_df(_result_df, price_predicate_columns, argument.apt_detail_pk)
            _result_df_smooth = formatting_df(_result_df_smooth, price_smoothing_predicate_columns,
                                              argument.apt_detail_pk)
            # database injection
            GinAptQuery.insert_or_update_predicate_value(list(_result_df.values))
            GinAptQuery.insert_or_update_predicate_smoothing_value(list(_result_df_smooth.values))

            print('Complete Database Injection')

    # -------------------------------------------------------------------------------------------------------------- #
    # Command :
    #       - python predicate.py --evaluation --apt_detail_pk={아파트 pk}
    #       - python predicate.py --evaluation --evaluation_plot --apt_detail_pk={아파트 pk}   [그림 출력]
    #
    #           ex) python predicate.py --evaluation --apt_detail_pk=1
    #           ex) python predicate.py --evaluation --evaluation_plot --apt_detail_pk=1
    #
    # Explanation : [apt_detail_pk] 정확도 평가
    # -------------------------------------------------------------------------------------------------------------- #
    elif argument.evaluation and argument.apt_detail_pk:
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

    # -------------------------------------------------------------------------------------------------------------- #
    # Command :
    #       - python predicate.py --apt_detail_pk={아파트 pk}
    #       - python predicate.py --apt_detail_pk={아파트 pk} --date={날짜}     [지정된 날짜를 예측]
    #
    #           ex) python predicate.py --apt_detail_pk=1
    #           ex) python predicate.py --apt_detail_pk=1 --date=2018-01-01
    #
    # Explanation : [apt_detail_pk]의 [지정한 날짜]의 가격 예측
    # Reference: --db_inject 을 추가하면 mysql 에 예측된 결과 저장
    # -------------------------------------------------------------------------------------------------------------- #
    elif argument.apt_detail_pk:
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
            price_predicate_columns.extend(['apt_detail_pk', 'trade_cd'])
            _result_df = pd.DataFrame([[argument.date, _result['predicate_price_min'], _result['predicate_price_avg'],
                                        _result['predicate_price_max'], argument.apt_detail_pk, argument.trade_cd]],
                                      columns=['date', 'price_min', 'price_avg', 'price_max',
                                               'apt_detail_pk', 'trade_cd'])

            # Database injection (price_min, price_max, price_avg)
            GinAptQuery.insert_or_update_predicate_value(list(_result_df.values))

            # Database injection (price_min_smoothing, price_max_smoothing, price_avg_smoothing)
            db_to_frame = pd.read_sql_query(
                f"SELECT * FROM predicate_price WHERE apt_detail_pk={argument.apt_detail_pk};", settings.cnx)
            db_to_frame = db_to_frame[['reg_date', 'price_min', 'price_avg', 'price_max']]
            db_to_frame.columns = ['date', 'predicate_price_min', 'predicate_price_avg', 'predicate_price_max']

            db_frame_smoothing = AptPredicate.predicate_transform_range(db_to_frame)
            db_frame_smoothing = formatting_df(db_frame_smoothing, price_smoothing_predicate_columns,
                                               argument.apt_detail_pk)
            GinAptQuery.insert_or_update_predicate_smoothing_value(list(db_frame_smoothing.values))

            print('Complete Database Injection')
    else:
        raise argparse.ArgumentError()
