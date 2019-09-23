# -*- coding:utf-8 -*-
import os
import pandas as pd
from datetime import datetime
from database import GinAptQuery
from feature import make_feature


DATASET_DIR = './dataset'
label_name = 'price'
feature_name_list = [
    'sale_price_with_floor',
    'sale_price_with_floor_recent',
    'sale_price_with_floor_group',
    'sale_price_with_floor_group_recent',
    'sale_price_with_complex_group',
    'sale_price_with_complex_group_recent',
    # 'trade_price_with_floor',
    # 'trade_price_with_floor_recent',
    # 'trade_price_with_floor_group',
    # 'trade_price_with_floor_group_recent',
    # 'trade_price_with_complex_group',
    # 'trade_price_with_complex_group_recent'
]


def make_dataset(filename, trade_cd='t'):
    query = GinAptQuery()
    #
    pk_list = [pk for pk in GinAptQuery.get_apt_detail_list().fetchall()]
    # pk_list = [(3, 1)]

    total_data = []
    for i, (apt_master_pk, apt_detail_pk) in enumerate(pk_list):
        try:
            print('i : {} apt detail pk : {}'.format(i, apt_detail_pk))
            for trade_pk, _, year, mon, day, floor, extent, price in query.get_trade_price(apt_detail_pk, trade_cd).fetchall():
                trg_date = '{0}-{1:02d}-{2:02d}'.format(year, int(mon), int(day))
                trg_date = datetime.strptime(trg_date, '%Y-%m-%d')
                price = price / extent

                feature_df = make_feature(
                    feature_name_list=feature_name_list,
                    apt_master_pk=apt_master_pk,
                    apt_detail_pk=apt_detail_pk,
                    trade_cd=trade_cd,
                    trg_date=trg_date,
                    sale_month_size=6,
                    sale_recent_month_size=2,
                    trade_month_size=6,
                    trade_recent_month_size=2,
                    floor=floor,
                    extent=extent,
                    trade_pk=trade_pk
                )
                if feature_df is not None:
                    feature_df[label_name] = price
                    total_data.append(feature_df)
        except KeyboardInterrupt:
            # file save
            filepath = os.path.join(DATASET_DIR, filename)
            total_df = pd.concat(total_data)
            total_df = total_df.reset_index(drop=True)
            total_df.to_csv(filepath, index=False)

    # file save
    filepath = os.path.join(DATASET_DIR, filename)
    total_df = pd.concat(total_data)
    total_df = total_df.reset_index(drop=True)
    total_df.to_csv(filepath, index=False)


make_dataset('sale_price.csv', trade_cd='t')
