# -*- coding:utf-8 -*-
import settings
import argparse
import pandas as pd
from datetime import datetime
from database import GinAptQuery
from feature import make_feature


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features', type=list, default=settings.features)
    parser.add_argument('--sale_month_size', type=int, default=settings.sale_month_size)
    parser.add_argument('--sale_recent_month_size', type=int, default=settings.sale_recent_month_size)
    parser.add_argument('--trade_month_size', type=int, default=settings.trade_month_size)
    parser.add_argument('--trade_recent_month_size', type=int, default=settings.trade_recent_month_size)

    parser.add_argument('--save_path', type=str, default=settings.save_path)
    parser.add_argument('--trade_cd', type=str, choices=['t', 'c'], default=settings.trade_cd)
    parser.add_argument('--label_name', type=str, default=settings.label_name)
    parser.add_argument('--dataset_pk_size', type=int, default=100)
    return parser.parse_args()


def main(argument):
    query = GinAptQuery()
    pk_list = [pk for pk in query.get_apt_detail_list(argument.dataset_pk_size).fetchall()]

    total_data = []
    for i, (apt_master_pk, apt_detail_pk) in enumerate(pk_list):
        try:
            print('i : {} \t apt detail pk : {}'.format(i, apt_detail_pk))
            for trade_pk, _, year, mon, day, floor, extent, price in \
                    query.get_trade_price(apt_detail_pk, argument.trade_cd).fetchall():
                trg_date = '{0}-{1:02d}-{2:02d}'.format(year, int(mon), int(day))
                trg_date = datetime.strptime(trg_date, '%Y-%m-%d')
                price = price / extent

                feature_df = make_feature(feature_name_list=argument.features, apt_master_pk=apt_master_pk,
                                          apt_detail_pk=apt_detail_pk, trade_cd=argument.trade_cd, trg_date=trg_date,
                                          sale_month_size=argument.sale_month_size,
                                          sale_recent_month_size=argument.sale_recent_month_size,
                                          trade_month_size=argument.trade_month_size,
                                          trade_recent_month_size=argument.trade_recent_month_size, floor=floor,
                                          extent=extent, trade_pk=trade_pk)
                if feature_df is not None:
                    feature_df['apt_detail_pk'] = apt_detail_pk
                    feature_df[argument.label_name] = price
                    total_data.append(feature_df)
        except KeyboardInterrupt:
            # file save
            total_df = pd.concat(total_data)
            total_df = total_df.reset_index(drop=True)
            total_df.to_csv(argument.save_path, index=False)
        except Exception as e:
            print(e)
            continue

    # file save
    total_df = pd.concat(total_data)
    total_df = total_df.reset_index(drop=True)
    total_df.to_csv(argument.save_path, index=False)


if __name__ == '__main__':
    args = get_args()
    main(args)
