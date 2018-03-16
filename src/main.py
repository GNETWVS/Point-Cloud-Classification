import argparse
import os
import sys

from download_prepare_data import download_datasets, prepare_datasets
from model import Model


def main(argv):

    parser = argparse.ArgumentParser()

    parser.add_argument('--download_data', action='store_true', default=False,
                        help='Turn on to download data to disk.')
    parser.add_argument('--data_dir', default='../data/',
                        help='Directory for downloading ModelNet10')
    parser.add_argument('--Net10_data_dir', default='../data/ModelNet10/',
                        help='Directory for ModelNet10')
    parser.add_argument('--train', action='store_true', default=False,
                        help='Call to train PointNet.')
    parser.add_argument('--test', action='store_true', default=False,
                        help='Call to test PointNet. Must load trained network file.')

    parser.add_argument('--augment_training', action='store_true', default=False,
                        help='Call to apply random rotations to input points during training.')
    parser.add_argument('--small_sample_threshold', type=int, default=1024,
                        help='Threshold for removing samples with low numbers of points.')
    parser.add_argument('--n_points', type=int, default=1024,
                        help='Number of points from each cloud to sample for training/testing.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training/testing')
    parser.add_argument('--n_epochs', type=int, default=1000,
                        help='Number of passes through training set.')
    parser.add_argument('--early_stopping_max_checks', type=int, default=100,
                        help='Stop early when loss does not improve for max_checks.')

    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer.')
    parser.add_argument('--keep_prob', type=float, default=0.8,
                        help='Keep probability for dropout. Set to 1.0 to remove dropout.')

    parser.add_argument('--load_checkpoint', action='store_true', default=False,
                        help='Call with *.ckpt file to load a saved model.')
    parser.add_argument('--saved_model_directory', type=str, default='../models/',
                        help='Directory for saving trained models.')
    parser.add_argument('--model_name', type=str, default=None,
                        help='Name of checkpoint to load into graph.')

    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.saved_model_directory, exist_ok=True)

    if args.download_data:
        download_datasets(args)
        prepare_datasets(args)

    if args.train:
        model = Model(args)
        model.train()

    if args.test:
        model = Model(args)
        model.test()

if __name__ == '__main__':
    main(sys.argv)