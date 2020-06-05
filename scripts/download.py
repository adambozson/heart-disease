import os
from argparse import ArgumentParser, Namespace

import wget

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"


def download_data(dir_path: str = ".") -> str:
    if not os.path.isdir(dir_path):
        raise NotADirectoryError(dir_path)
    return wget.download(DATA_URL, out=dir_path)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-d", "--dir", default=".")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path = download_data(args.dir)
    print(f"\nSaved to {path}")
