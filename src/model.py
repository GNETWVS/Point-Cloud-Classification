import os
import numpy as np
import tensorflow as tf
from data_utils import get_filenames_and_class, generate_class_str_to_num_dict
from data_utils import get_points_and_class, read_off_file_into_nparray

class Model:
    def __init__(self, args):
        self.args = args
        self.train_list, self.test_list = get_filenames_and_class(args.Net10_data_dir)
        self.class_string_to_num = generate_class_str_to_num_dict(args.Net10_data_dir)

    def build_point_net(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def save(self):
        pass

    def load(self):
        pass