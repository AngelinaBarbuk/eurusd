import argparse

from aggregate import aggregate, create_test_train
from model_1 import Model_1
from model_2 import Model_2
from model_3 import Model_3
from model_4 import Model_4
from model_5 import Model_5
from simple_model import SimpleModel


def _parse_args():
    parser = argparse.ArgumentParser(prog='Aggregates cryptocurrency data from all subdirectories to two files with'
                                          ' indexes and series. Data is filtered according to chosen dates range.')
    parser.add_argument('--data_dir', default="history", help='Path to directory with cryptocurrency data')
    parser.add_argument('--begin_date', default="2016.01.01", help="Begin date in format 'dd.mm.yyyy'")
    parser.add_argument('--end_date', default="2017.12.31", help="End date in format 'dd.mm.yyyy'")
    parser.add_argument('--columns', default='O,H,L,C', help="List of columns to fetch, separated by ','")
    parser.add_argument('--output_filename', default='aggregated',
                        help='Common name for output files. Pass without extension')
    return parser.parse_args()

def run_model(model, X_train, y_train,X_test):
    model.create()
    model.fit(X_train, y_train)
    return model


def test_model(model,X_test, y_test):
    model.evaluate(X_test)
    return model,  model.predict(X_test)

def run_model(model, X_train, y_train,X_test):
    model.create()
    model.fit(X_train, y_train)
    return model,  model.predict(X_test)

def _main(args):
    ts_values_df = aggregate(args.data_dir, args.begin_date, args.end_date, args.columns.split(','))
    X_train, y_train, X_test, y_test = create_test_train(ts_values_df)

    model_1, predicted_1 = run_model(Model_1(), X_train, y_train,X_test)
    model_2, predicted_2 = run_model(Model_2(), X_train, y_train, X_test)
    model_3, predicted_3 = run_model(Model_3(), X_train, y_train, X_test)
    model_4, predicted_4 = run_model(Model_4(), X_train, y_train, X_test)
    model_5, predicted_5 = run_model(Model_5(), X_train, y_train, X_test)

    print(predicted_1)


if __name__ == '__main__':
    _main(_parse_args())