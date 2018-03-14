from data_downloader import download_datasets
from data_utils import remove_small_point_clouds
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
    parser.add_argument('--Net40_data_dir', default='../data/ModelNet40',
                        help='Directory for ModelNet40')

    parser.add_argument('--augment_training_data_with_rotations', action='store_true', default=False,
                        help='Call to apply random rotations to input points during training.')
    parser.add_argument('--small_sample_threshold', type=int, default=1024,
                        help='Threshold for removing samples with low numbers of points.')
    parser.add_argument('--n_points', type=int, default=1024,
                        help='Number of points from each cloud to sample for training/testing.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training/testing')
    parser.add_argument('--n_epochs', type=int, default=50,
                        help='Number of passes through training set.')
    parser.add_argument('--early_stopping_max_checks', type=int, default=20,
                        help='Stop early when loss does not improve for max_checks.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer.')

    parser.add_argument('--load_checkpoint', action='store_true', default=False,
                        help='Call with *.ckpt file to load a saved model.')
    parser.add_argument('--saved_model_directory', type=str, default='../models/',
                        help='Directory for saving trained models.')
    parser.add_argument('--model', type=str, default=None,
                        help='Name of checkpoint to load into graph.')

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)

    if args.download_data:
        download_datasets(args)

    model = Model(args)
    print('Model built')
    model.train()


if __name__ == '__main__':
    main(sys.argv)