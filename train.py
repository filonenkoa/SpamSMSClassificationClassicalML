"""
The main training file
"""
import argparse

from dataloaders import SmsSpam, implemented_readers
from solvers import implemented_solvers

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the dataset.')
    parser.add_argument('--dataset', metavar='S', type=str,
                        help='Path to the dataset')
    parser.add_argument('--dataset_type', metavar='S', type=str, default="smsspam",
                        help=f'Type of the dataset. One of {implemented_readers}')
    parser.add_argument('--log', metavar='S', type=str,
                        help='Path to the log folder')
    parser.add_argument('--split', metavar='S', type=int, nargs="+",
                        help='Train|Val|Test split as 3 values with a sum of 1. Example: 0.7 0.2 0.1')
    parser.add_argument('--solver', metavar='S', type=str, default="linear_regression",
                        help=f'Type of machine learning solved. Implemented: {implemented_solvers}')
    args = parser.parse_args()

    assert args.dataset_type in implemented_readers
    assert args.solver in implemented_solvers

    return args


def main():
    args = parse_args()


if __name__ == "__main__":
    main()
