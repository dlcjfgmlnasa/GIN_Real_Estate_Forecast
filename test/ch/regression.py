# -*- coding:utf-8 -*-
import datetime
import argparse
import numpy as np
import pandas as pd
from database import GinQuery
import matplotlib.pyplot as plt
from sklearn import linear_model
import pickle


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pk', default=112, type=str)
    parser.add_argument('--time_size', default=100, type=int)
    parser.add_argument('--recent_time_size', default=3, type=int)
    parser.add_argument('--sale_trend_time_size', default=30 * 6, type=int)
    parser.add_argument('--trade_trend_time_size', default=30 * 6, type=int)
    parser.add_argument('--similarity_size', default=10, type=int)
    return parser.parse_args()


def get_time_list(trg_time: str, size: int) -> list:
    time_format = '%Y-%m-%d'
    trg_time = datetime.datetime.strptime(trg_time, time_format)
    time_list = []
    for _ in range(size):
        trg_time = trg_time - datetime.timedelta(days=1)
        time_list.append("'" + trg_time.strftime(time_format) + "'")
    return time_list


def feature_engine(df: pd.DataFrame, recent_size: int, sale_trend_size: int, trade_trend_size: int) -> dict:
    # feature engineering (피쳐 엔지니어링)

    # 1. price average
    #  => 전체 가격의 평균가를 구함
    price_avg = np.average(df.price)

    # 2. recent price average
    #  => 현재 날짜로 부터 최근가의 평균가를 구함
    recent_price_avg = np.average(df.price[-recent_size:])

    # 3. floor
    #  => 층
    floor = int(df.floor[0])
    pk = str(df.detail_pk[0])
    date = str(df.reg_date[0])

    try:
        similarity_list = GinQuery.get_similarity(pk).fetchall()[0][0].split(',')
    except IndexError:
        similarity_list = []

    # 4. 매물 트렌트 출력
    #  => cf. sale_trend_time_size
    trend_time_str = ','.join(get_time_list(date, sale_trend_size))
    sale_price = GinQuery.get_naver_sale_price_for_trend(pk, trend_time_str).fetchall()

    if len(sale_price) == 0:
        # if 현재 건물의 매물 데이터가 없으면 비슷한 건물의 데이터를 뽑는다.
        # for similarity_pk in similarity_list:
        #     sale_price = GinQuery.get_naver_sale_price_for_trend(similarity_pk, trend_time_str).fetchall()
        #     if len(sale_price) != 0:
        #         break
        pass
    if similarity_list is None or len(sale_price) == 0:
        # if similarity 가 없거나 매물의 갯수가 하나도 존재하지 않을때
        sale_trend_value = 0.0
    else:
        sale_price = pd.DataFrame(sale_price, columns=['detail_pk', 'reg_date', 'floor', 'price'])
        sale_price_temp = {}
        for date, price in zip(sale_price.reg_date, sale_price.price):
            date = date.strftime('%Y-%m-%d')
            try:
                sale_price_temp[date].append(price)
            except KeyError:
                sale_price_temp[date] = [price]
        sale_price = []
        for value in sale_price_temp.values():
            result = sum(value) / len(value)
            sale_price.append(result)
        sale_trend_value = sale_price[-1] - sale_price[0]

    # 5. 매매 트렌트 출력
    #  => cf. trade_trend_time_size
    time_list = [date[1:-2] for date in get_time_list(date, trade_trend_size)]
    t = set()
    for time in time_list:
        t_ = time.split('-')
        t.add("'"+t_[0] + t_[1]+"'")
    trade_price = GinQuery.get_trade_price_with_date(pk, ','.join(t)).fetchall()

    if len(trade_price) != 0:
        trade_price = pd.DataFrame(trade_price, columns=['date', 'price'])
        trade_price_temp = {}
        for date, price in zip(trade_price.date, trade_price.price):
            price = int(price)
            try:
                trade_price_temp[date].append(price)
            except KeyError:
                trade_price_temp[date] = [price]
        trade_price = []
        for value in trade_price_temp.values():
            result = sum(value) / len(value)
            trade_price.append(result)
        trade_trend_value = trade_price[-1] - trade_price[0]
    else:
        trade_trend_value = 0

    return {
        'sale_price_avg': price_avg,
        'sale_recent_price_avg': recent_price_avg,
        'sale_floor': floor,
        'sale_trend': sale_trend_value,
        'trade_trend': trade_trend_value
    }


def supplement(pk: int, floor: int, trg_date: str, time_size: int):
    # 만약 해당 건물의 매매가격이 없을떄
    src_time_str = ','.join(get_time_list(trg_date, time_size))

    # 1. 해당 날짜로부터 이전 데이터의 전체 데이터
    sale_trend = GinQuery.get_naver_sale_price_for_trend(pk, src_time_str).fetchall()
    df = pd.DataFrame(sale_trend, columns=['pk', 'date', 'floor', 'price'])

    if len(df) == 0:
        return
    else:
        # Merge
        temp = {}
        for date, price in zip(df.date, df.price):
            date = date.strftime('%Y-%m-%d')
            try:
                temp[date].append(price)
            except KeyError:
                temp[date] = [price]

        values = [sum(value) / len(value) for value in temp.values()]\

        df = pd.DataFrame({
            'reg_date': list(temp.keys()),
            'price': values
        })
        df['detail_pk'] = pk
        df['floor'] = floor
        return df
    pass


def pk_make_dataset(pk: int, time_size: int,
                    recent_time_size: int, sale_trend_time_size: int, trade_trend_time_size: int) -> pd.DataFrame:
    # 매매가격 예측을 위한 학습할 데이터셋 생성
    # => 매물가격 (size) 만큼을 이용하여 매매 가격 예측

    # 데이터 셋 생성
    temp = []
    db_cursor = GinQuery.get_trade_price(pk)
    for i, (pk, year, mon, day, floor, t_amt) in enumerate(db_cursor.fetchall()):
        trg_date = '{0}-{1:02d}-{2:02d}'.format(year, int(mon), int(day))

        # size 크기 만큼 매물 가격 불러오기
        src_time_str = ','.join(get_time_list(trg_date, time_size))
        sale_price = GinQuery.get_naver_sale_price_with_floor(pk, src_time_str, floor).fetchall()

        # 만약 size 크기만큼의 매물가격이 없다면
        cd = 0
        if len(sale_price) == 0:
            df = supplement(pk, floor, trg_date, time_size)
            cd = 1
            if df is None:
                continue
        else:
            df = pd.DataFrame(sale_price, columns=['detail_pk', 'reg_date', 'floor', 'price'])

        # 피쳐 엔지니어링
        feature = feature_engine(
            df,
            recent_size=recent_time_size,
            sale_trend_size=sale_trend_time_size,
            trade_trend_size=trade_trend_time_size
        )
        feature['cd'] = cd
        feature['real_value'] = t_amt

        temp.append(feature)

    f_df = pd.DataFrame(temp)
    return f_df


def similarity_merge_df(args) -> pd.DataFrame:
    # similarity pk 값을 출력하여 merge
    target_pk = args.pk
    similarity_pk_list = GinQuery.get_similarity(target_pk).fetchone()[0]
    similarity_pk_list = similarity_pk_list.split(',')
    similarity_pk_list.insert(0, target_pk)

    total = []
    for i in range(args.similarity_size):
        pk = similarity_pk_list[i]

        f_df = pk_make_dataset(pk, args.time_size,
                               args.recent_time_size, args.sale_trend_time_size, args.trade_trend_time_size)
        f_df['similarity'] = i
        total.append(f_df)
    total_df = pd.concat(total)
    return total_df


def total_merge_df(args) -> pd.DataFrame:
    # 전체 pk 값을 merge
    count = GinQuery.get_pk_count().fetchone()[0]
    pk_list = [pk[0] for pk in GinQuery.get_pk_list(size=count).fetchall()]

    total = []
    for pk in pk_list:
        f_df = pk_make_dataset(pk, args.time_size,
                               args.recent_time_size, args.sale_trend_time_size, args.trade_trend_time_size)
        total.append(f_df)
        print('make df pk : {}'.format(pk))
    total_df = pd.concat(total)
    return total_df


def total_merge_df_save(args):
    # 전체 pk 값을 저장
    filename = 'total.csv'
    count = GinQuery.get_pk_count().fetchone()[0]
    pk_list = [pk[0] for pk in GinQuery.get_pk_list(size=count).fetchall()]

    with open(filename, 'w') as f:
        for num, pk in enumerate(pk_list):
            df = pk_make_dataset(pk, args.time_size,
                                 args.recent_time_size, args.sale_trend_time_size, args.trade_trend_time_size)
            if num == 0:
                df.to_csv('total.csv', index=False, mode='w')
            else:
                df.to_csv('total.csv', index=False, mode='a', header=None)
            print('make df pk : {}'.format(pk))


def total_merge_df_load():
    filename = 'total.csv'
    df = pd.read_csv(filename)
    return df


def split_train_test(dataset):
    train, test = dataset[:-10], dataset[-10:]
    return train, test


def split_train_test_similarity(dataset):
    df = dataset[dataset.similarity == 0]
    train, test = df[:-10], df[-10:]
    df2 = dataset[dataset.similarity != 0]
    train = pd.concat([train, df2])
    test = dataset[-10:]

    return train, test


def save_model(train: pd.DataFrame):
    # model save
    filename = 'model.sav'

    columns = list(train.columns)
    label_column = 'real_value'
    columns.remove(label_column)
    feature_columns = columns

    # Train Split feature / label
    train_x, train_y = train[feature_columns], train[label_column]

    # Linear Regression Train model
    linear = linear_model.LinearRegression()
    linear.fit(train_x, train_y)
    pickle.dump(linear, open(filename, 'wb'))

    print('model save : {}'.format(filename))


def predicate(test: pd.DataFrame):
    def root_mean_square_error(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())
    filename = 'model.sav'

    columns = list(test.columns)
    label_column = 'real_value'
    columns.remove(label_column)
    feature_columns = columns

    # Test Split feature / label
    test_x, test_y = test[feature_columns], test[label_column]
    test_y = np.array([int(y) for y in test_y])

    linear = pickle.load(open(filename, 'rb'))
    pred = linear.predict(test_x)

    # result
    df = pd.DataFrame({
        'real': list(test_y),
        'pred': pred,
        'diff': np.abs(np.array([int(y) for y in test_y]) - pred),
        'error_percent': np.abs(100 - ((pred / test_y) * 100)),
        'error range(6%)': np.abs(100 - ((pred / test_y) * 100)) <= 6
    })

    error = root_mean_square_error(pred, test_y)
    print(df, end='\n\n')
    print('root_mean_square_error : {}'.format(error))
    percent = (1 - (np.mean(test_y) - error) / np.mean(test_y)) * 100
    print('error percent : {:.2f}%'.format(percent))
    print('success percent : {}%'.format(np.mean(np.abs(100 - ((pred / test_y) * 100)) <= 6) * 100))

    args = get_args()
    size = len(df)
    plt.title('pk : {}  / error percent : {:.2f}'.format(args.pk, float(percent)))
    plt.scatter([i for i in range(size)], df['real'], label='real')
    plt.scatter([i for i in range(size)], df['pred'], label='pred')
    plt.legend()
    plt.show()


def model_run(args):
    df = pk_make_dataset(args.pk, args.time_size,
                         args.recent_time_size, args.sale_trend_time_size, args.trade_trend_time_size)
    train, test = split_train_test(df)
    save_model(train)
    predicate(test)


def similarity_model_run(args):
    df = similarity_merge_df(args)
    df.to_csv('test.csv')
    train, test = split_train_test_similarity(df)
    save_model(train)
    predicate(test)


def total_model_run(args):
    df = total_merge_df_load()
    train, test = split_train_test(df)
    train = pd.concat([train, test])

    test = pk_make_dataset(args.pk, args.time_size,
                           args.recent_time_size, args.sale_trend_time_size, args.trade_trend_time_size)
    # test = test[:10]
    save_model(train)
    predicate(test)


def main():
    args = get_args()
    similarity_model_run(args)

    # test = pk_make_dataset(args.pk, args.time_size,
    #                        args.recent_time_size, args.sale_trend_time_size, args.trade_trend_time_size)
    # # total_model_run(args)


if __name__ == '__main__':
    main()
