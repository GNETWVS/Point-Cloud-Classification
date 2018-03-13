from data_downloader import download_datasets
from model import Model
import argparse
import sys
import numpy as np
import os
import tensorflow as tf


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--download_data', action='store_true', default=False,
                        help='Turn on to download data to disk.')
    parser.add_argument('--data_dir', default='../data/',
                        help='Directory for downloading ModelNet10/40')
    parser.add_argument('--Net10_data_dir', default='../data/ModelNet10',
                        help='Directory for ModelNet10')
    parser.add_argument('--Net10_data_dir', default='../data/ModelNet40',
                        help='Directory for ModelNet40')

    parser.add_argument('--Augment training data with rotations.', action='store_true', default=False,
                        help='Call to apply random rotations to input points during training.')

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.download_data:
        download_datasets(args)




if __name__ == '__main__':
    main(sys.argv)