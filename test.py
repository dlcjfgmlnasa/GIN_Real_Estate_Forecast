# -*- coding:utf-8 -*-
import os
import argparse
import numpy as np
import pandas as pd
from train import train
from database import GinAptQuery
from data_helper import label_name
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default=os.path.join('./dataset', 'sale_price.csv'))
    parser.add_argument('--features', type=list,
                        default=[
                            'sale_price_with_floor',
                            'sale_price_with_floor_recent',
                            'sale_price_with_floor_group',
                            'sale_price_with_floor_group_recent',
                            'sale_price_with_complex_group',
                            'sale_price_with_complex_group_recent'
                        ])
    parser.add_argument('--model', type=str, default='linear_regression',
                        choices=[
                            'linear_regression',
                            'svm',
                            'dnn'
                        ])
    parser.add_argument('--n_fold', type=int, default=10)
    parser.add_argument('--plot_flag', type=bool, default=False)
    parser.add_argument('--result_path', type=str,
                        default=os.path.join('./result', 'linear_regression', 'test03.xlsx'))
    return parser.parse_args()


def save_result(info, eval_result_list):
    # Calculation Evaluation Result
    total_evaluation = []
    for eval_result in eval_result_list:
        total_evaluation.append(eval_result['evaluation'])
    total_evaluation = pd.concat(total_evaluation)
    total_evaluation_sum = pd.DataFrame({
        column: [np.average(total_evaluation[column])]
        for column in total_evaluation.columns
    }, index=['average'])
    total_evaluation = total_evaluation.reset_index(drop=True)
    total_evaluation = pd.concat([total_evaluation, total_evaluation_sum])
    print(total_evaluation)

    # Save...
    with pd.ExcelWriter(info.result_path, engine='xlsxwriter') as writer:
        total_evaluation.to_excel(writer, sheet_name='result', startcol=1, startrow=5)
        worksheet = writer.sheets['result']
        # model info save
        worksheet.write(1, 1, 'features : {}'.format(', '.join(info.features)))
        worksheet.write(2, 1, 'model : {}'.format(args.model))
        worksheet.write(3, 1, 'n_fold : {}'.format(args.n_fold))
        worksheet.set_column('B:G', 20)

        for i, eval_result in enumerate(eval_result_list):
            sheet_name = '{}_fold'.format(i)
            eval_result['evaluation'].to_excel(writer, sheet_name=sheet_name, startcol=1, startrow=2)
            eval_result['result'].to_excel(writer, sheet_name=sheet_name, startcol=1, startrow=5)
            worksheet = writer.sheets[sheet_name]
            worksheet.set_column('B:G', 20)


def test(feature_df: pd.DataFrame, label_df: pd.DataFrame, pk_df: pd.DataFrame,
         model_type: str, n_fold: int, plot=True):
    # N-fold cross validation
    test_result = []
    cv = KFold(n_splits=n_fold, shuffle=True, random_state=0)
    for i, (idx_train, idx_test) in enumerate(cv.split(X=feature_df, y=label_df)):
        print('Testing {}-fold ...'.format(i))
        x_train, x_test = feature_df.iloc[idx_train], feature_df.iloc[idx_test]
        y_train, y_test = label_df.iloc[idx_train], label_df.iloc[idx_test]

        # Calculation extent
        test_pk = pk_df[idx_test]
        pk_list = list(set(pk_df[idx_test].astype(str).values))
        extent_df = pd.DataFrame(
            GinAptQuery.get_extent(','.join(pk_list)).fetchall(),
            columns=['apt_detail_pk', 'extent']
        )
        extent_df = test_pk.apply(lambda pk: float(extent_df[extent_df.apt_detail_pk == pk].extent))

        # Training...
        model = train(x_train, y_train, model_type)
        predicate = model.predict(x_test) * extent_df.values
        real = y_test.values * extent_df.values

        # Evaluation...
        n_result = evaluation(real=real, pred=predicate, plot=plot)

        # Evaluation store
        test_result.append(n_result)
    return test_result


def evaluation(real: np.ndarray, pred: np.ndarray, plot=True):
    def root_mean_square_error(predictions, targets):
        return np.sqrt(((predictions - targets) ** 2).mean())

    def mean_absolute_percentage_error(predictions, targets):
        return np.mean(np.abs((targets - predictions) / targets)) * 100

    result_df = pd.DataFrame({
        'real': real,
        'pred': pred,
        'diff': np.abs(np.array([int(y) for y in real]) - pred),
        'error_percent': np.abs(100 - ((pred / real) * 100)),
        'error range(6%)': np.abs(100 - ((pred / real) * 100)) <= 6
    })

    evaluation_df = pd.DataFrame({
        'rmse': [root_mean_square_error(pred, real)],
        'mape': [mean_absolute_percentage_error(pred, real)],
        'success_percent': np.mean(result_df['error range(6%)']) * 100,
        'error_percent': (1 - np.mean(result_df['error range(6%)'])) * 100
    }, index=['result'])

    if plot:
        plt.scatter([i for i in range(len(pred))], result_df['real'], label='real')
        plt.scatter([i for i in range(len(pred))], result_df['pred'], label='pred')
        plt.legend()

    return {
        'result': result_df,
        'evaluation': evaluation_df
    }


if __name__ == '__main__':
    args = get_args()
    df = pd.read_csv(args.dataset_path)
    result = test(df[args.features], df[label_name], df.apt_detail_pk, args.model, args.n_fold, args.plot_flag)
    save_result(args, result)

