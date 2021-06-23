"""
The main training file
"""
import argparse
import logging
from os import makedirs
from os.path import join
from datetime import datetime

from dataloaders import implemented_readers, get_dataloader
from solvers import implemented_solvers, get_solver

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze the dataset.')
    parser.add_argument('--dataset', metavar='S', type=str,
                        help='Path to the dataset')
    parser.add_argument('--dataset_type', metavar='S', type=str, default="smsspam",
                        help=f'Type of the dataset. One of {implemented_readers}')
    parser.add_argument('--log', metavar='S', type=str,
                        help='Path to the log folder')
    parser.add_argument('--split', metavar='N', type=float, default=0.8,
                        help='Train/Test split. Default: 0.8')
    parser.add_argument('--solver', metavar='S', type=str, default="logistic_regression",
                        help=f'Type of machine learning solved. Implemented: {implemented_solvers}')
    args = parser.parse_args()

    assert args.dataset_type in implemented_readers
    assert args.solver in implemented_solvers

    makedirs(args.log, exist_ok=True)

    return args


def config_logger(logdir):
    handlers = []
    handlers.append(logging.StreamHandler())
    if logdir is not None:
        file_name = f"{datetime.now().strftime('%Y_%m_%d-%H_%M_%S')}.log"
        handlers.append(logging.FileHandler(join(logdir, file_name)))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers
    )


def main():
    debug = True
    args = parse_args()
    config_logger(args.log)
    logging.info(datetime.now())
    dataloader = get_dataloader(args.dataset_type)(args.dataset, args.split)
    solver = get_solver(args.solver)(dataloader, debug)
    solver.train()


if __name__ == "__main__":
    main()
