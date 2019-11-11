# -*- coding:utf-8 -*-
import time
import argparse
import settings
from database import GinAptQuery
from predicate import predicate, get_model, FeatureExistsError


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predicate_time_checking', action='store_true')
    parser.add_argument('--model_info', type=str, default=settings.model_info)
    return parser.parse_args()


def predicate_time_testing(arguments):
    apt_model = get_model(arguments.model_info)

    start_time = time.time()
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
            print('num: {0:} \t apt:{1:} \t time: {2:.2f}sec \t result: {3:}'.format(
                i, apt_detail_pk, (end_time - start_time), result
            ))
            total_predicate_apt_count += 1

        except FeatureExistsError:
            print(f'apt:{apt_detail_pk} - 매매, 매물 데이터가 존재하지 않음')
            total_except_predicate_apt_count += 1

        except KeyboardInterrupt:
            break

    end_time = time.time()
    print('\n총 걸린 시간 : {0:.2f}sec \t 총 예측 건물 수 : {1: } \t 예외 건물 수 : {2:}'.format(
        end_time - start_time, total_predicate_apt_count, total_except_predicate_apt_count
    ))


if __name__ == '__main__':
    args = get_args()

    if args.predicate_time_checking:
        predicate_time_testing(args)