# -*- coding:utf-8 -*-
import time
import argparse
import settings
from database import GinAptQuery
from predicate import predicate, get_model, FeatureExistsError, predicate_full_range


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apt_predicate', action='store_true')
    parser.add_argument('--model_info', type=str, default=settings.model_info)
    return parser.parse_args()


def apt_predicate(arguments):
    apt_model = get_model(arguments.model_info)

    total_start_time = time.time()
    total_predicate_apt_count = 0
    total_except_predicate_apt_count = 0

    for i, apt_detail_pk in enumerate(GinAptQuery.get_predicate_apt_list().fetchall()):
        apt_detail_pk = apt_detail_pk[0]
        try:
            start_time = time.time()
            result = predicate(
                apt_detail_pk=apt_detail_pk,
                models=apt_model
            )
            end_time = time.time()
            print('num: {0:06d} \t apt:{1:07d} \t time: {2:.2f}sec \t pred_max: {3:07.2f} \t pred_avg: {4:07.2f} \t '
                  'pred_min: {5:07.2f} \t select : {6:}'.
                  format(i, apt_detail_pk, (end_time - start_time),
                         result['predicate_price_max'], result['predicate_price_avg'], result['predicate_price_min'],
                         result['select_model']))
            total_predicate_apt_count += 1

        except FeatureExistsError:
            total_except_predicate_apt_count += 1

        except KeyboardInterrupt:
            break

        except Exception as e:
            continue

    total_end_time = time.time()
    print('\n총 걸린 시간 : {0:.2f}sec \t 총 예측 건물 수 : {1: } \t 예외 건물 수 : {2:}'.format(
        total_end_time - total_start_time, total_predicate_apt_count, total_except_predicate_apt_count
    ))


if __name__ == '__main__':
    args = get_args()

    if args.predicate_time_checking:
        predicate_time_testing(args)
