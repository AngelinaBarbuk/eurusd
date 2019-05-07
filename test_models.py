import argparse

from aggregate import aggregate, create_test_train
from simple_model import SimpleModel


def _parse_args():
    parser = argparse.ArgumentParser(prog='Aggregates cryptocurrency data from all subdirectories to two files with'
                                          ' indexes and series. Data is filtered according to chosen dates range.')
    parser.add_argument('--data_dir', default="history", help='Path to directory with cryptocurrency data')
    parser.add_argument('--begin_date', default="2018.01.01", help="Begin date in format 'dd.mm.yyyy'")
    parser.add_argument('--end_date', default="2018.12.31", help="End date in format 'dd.mm.yyyy'")
    parser.add_argument('--columns', default='O,H,L,C', help="List of columns to fetch, separated by ','")
    parser.add_argument('--output_filename', default='aggregated',
                        help='Common name for output files. Pass without extension')
    return parser.parse_args()


def _main(args):
    ts_values_df = aggregate(args.data_dir, args.begin_date, args.end_date, args.columns.split(','))
    X_train, y_train, X_test, y_test = create_test_train(ts_values_df)
    model = SimpleModel()
    model.fit(X_train, y_train)
    y_predicted = model.predict(X_test)
    print(y_predicted)


if __name__ == '__main__':
    _main(_parse_args())