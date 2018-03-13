from data_downloader import download_datasets
import argparse
import sys
import numpy as np
import os
import tensorflow as tf


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--download_data', action='store_true', default=False,
                        help='Turn on to download data to disk.')
    parser.add_argument('--ModelNet10_dir', default='../data/',
                        help='Directory for ModelNet10 Dataset.')
    parser.add_argument('--ModelNet40_dir', default='../data/',
                        help='Directory for ModelNet40 Dataset.')

    args = parser.parse_args()

    os.makedirs(args.ModelNet10_dir, exist_ok=True)
    os.makedirs(args.ModelNet40_dir, exist_ok=True)

    if args.download_data:
        download_datasets(args)

if __name__ == '__main__':
    main(sys.argv)