# PointNet

## Introduction

This code is based on the work by (Qi, Su, Mo, & Guibas, 2017) in their paper [PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593). The netword archictecture allows for classification of unstructured point cloud data. This code includes a modification of the original paper replacing the Transformation mini networds with a simple linear transformation that aligns each point cloud to the coordinate axes by the cloud's ranked eigenvectors. Eigenvectors are found from the Singular Value Decomposition of each point cloud's covariance matrix. This implementation was trained and evaluated using the [ModelNet10](http://3dshapenets.cs.princeton.edu/) data set.

## To Run

Clone the repo, and move to the main directory.
`cd ~/PointNet/`
Install the dependencies.
`$ pip install -r requirements.txt`
If running for the first time run:
`python main.py --download_data`
This will generate the data directory, download and unzip the raw datafiles, remove small point clouds, split into train, create a data dictionary with object class as the key and point cloud filename as the value, and split the data into train, evaluation and test sets. The output is pickled, so this will only need to be performed a single time.

## To Train

`python main.py --train`

#### Optional settings include:

`--n_points`: Number of points from each cloud to sample for training/testing. Required for mini-batch training.

`--batch_size`: Batch size for training/testing

`--n_epochs`: Number of passes through training set.

`--early_stopping_max_checks`: Stop early when loss does not improve for max_checks.

`--learning_rate`: Learning rate for Adam Optimizer. This is the initial learning rate. The rate will is halved every 50 epochs.

`--keep_prob`: Keep probability for dropout. Set to 1.0 to remove dropout.

## To Test

Testing requires saved model after training. 

To run test:

`python main.py --test --load_checkpoint <saved_model_name>.ckpt --keep_prob 1.0`
