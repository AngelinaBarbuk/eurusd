#!/usr/bin/env python3
import os
from os.path import join

import pandas as pd
import numpy as np
import datetime
import argparse

from sklearn.preprocessing import StandardScaler


def _parse_args():
    parser = argparse.ArgumentParser(prog='Aggregates cryptocurrency data from all subdirectories to two files with'
                                          ' indexes and series. Data is filtered according to chosen dates range.')
    parser.add_argument('--data_dir', default="history", help='Path to directory with cryptocurrency data')
    parser.add_argument('--begin_date', default="2017.01.01", help="Begin date in format 'dd.mm.yyyy'")
    parser.add_argument('--end_date', default="2018.12.31", help="End date in format 'dd.mm.yyyy'")
    parser.add_argument('--columns', default='O,H,L,C', help="List of columns to fetch, separated by ','")
    parser.add_argument('--output_filename', default='aggregated',
                        help='Common name for output files. Pass without extension')
    return parser.parse_args()


def _get_filepath(data_dir, filename):
    return join(data_dir, filename)


def aggregate(data_dir, begin_date, end_date, columns):
    """
    Aggregates cryptocurrency data from all subdirectories in 'data_dir'.
    Data is filtered according to chosen dates range.
    :param data_dir: Path to directory with cryptocurrency data
    :param begin_date: Begin date in format 'yyyy.mm.dd'
    :param end_date: End date in format 'yyyy.mm.dd'
    :param columns: List of columns to fetch
    :return: List of time series names and dataframe with time series values.
            Each name corresponds to appropriate column in dataframe. Dataframe has dates as index.
    """
    begin_date = datetime.datetime.strptime(begin_date, '%Y.%m.%d')
    end_date = datetime.datetime.strptime(end_date, '%Y.%m.%d')

    time_series_df = None

    files = os.listdir(data_dir)

    for file in files:
        filepath = _get_filepath(data_dir, file)
        year = int(file.split('_')[-1].split('.')[0])
        if year < begin_date.year or year > end_date.year:
            print(f"Ignoring '{year}'' year. It doesn't contain enough data for given date range.")
            continue

        data = pd.read_csv(filepath, delimiter=',', decimal='.', header=0, names=['DT', 'T', 'O', 'H', 'L', 'C', 'V'])
        data['DT'] = data['DT'].str.cat(data['T'], sep=" ")
        data.drop('T', 1, inplace=True)
        data['DT'] = data['DT'].map(lambda string: datetime.datetime.strptime(string, '%Y.%m.%d %H:%M'))

        if data['DT'].iloc[-1] < begin_date or data['DT'].iloc[0] > end_date:
            print(f"Ignoring '{year}'' year. It doesn't contain enough data for given date range.")
            continue

        data = data[(data['DT'] >= begin_date) & (data['DT'] <= end_date)]
        data.drop_duplicates(subset=['DT'], inplace=True)
        data.set_index('DT', inplace=True)
        data = data[columns]

        time_series_df = data if time_series_df is None else time_series_df.append(data)
    return time_series_df


def save_aggregated(time_series_df, output_filename):
    """
    Saves aggreagated time series names, values and dates to separate files with.
    :param time_series_df: Time series dataframe.
    :param output_filename: Common name for output files.
    """

    ts_values_filename = output_filename + '_values.csv'
    time_series_df.to_csv(ts_values_filename, na_rep='n', header=False, index=False)


sc = StandardScaler()


def create_test_train(dataset, n=60, percentage=0.9):
    dataset_scaled = sc.fit_transform(dataset)
    p = int(len(dataset_scaled) * percentage)
    training_set = dataset_scaled[:p]
    test_set = dataset_scaled[p:]

    X_train = []
    y_train = []
    for i in range(n, len(training_set) - 1):
        X_train.append(training_set[i - n:i])
        y_train.append(training_set[i + 1, -1])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_test = []
    y_test = test_set[60:, -1]
    for i in range(n, len(test_set)):
        X_test.append(test_set[i - n:i])
    X_test = np.array(X_test)

    return X_train, y_train, X_test, y_test


def _main(args):
    ts_values_df = aggregate(args.data_dir, args.begin_date, args.end_date, args.columns.split(','))
    # get_slice(ts_values_df,0,0,0)
    create_test_train(ts_values_df)


if __name__ == '__main__':
    _main(_parse_args())
