import numpy as np
import os
import glob
from random import shuffle

def get_filenames_and_class(data_dir):
    train_list = []
    test_list = []
    classes = os.listdir(data_dir)
    for point_class in classes:
        train_dir = os.path.join(data_dir, point_class + '/train')
        test_dir = os.path.join(data_dir, point_class + '/test')
        for file in glob.glob(os.path.join(train_dir, '*.off')):
            # train_list.append({point_class: os.path.join(data_dir, point_class, 'train', file)})
            train_list.append({point_class: file})
        for file in glob.glob(os.path.join(test_dir, '*.off')):
            # test_list.append({point_class: os.path.join(data_dir, point_class, 'test', file)})
            test_list.append({point_class: file})
    return train_list, test_list

def generate_class_str_to_num_dict(data_dir):
    classes = sorted(os.listdir(data_dir))
    class_dict = {}
    for pt_class, i in enumerate(classes):
        class_dict[i] = pt_class
    return class_dict

def read_off_file_into_nparray(fname, n_points_to_read):
    with open(fname) as f:
        content = f.readlines()
        n_points = int(content[1].split()[0])
        points = content[2:n_points + 2]
        if n_points_to_read is not None:
            points = points[:n_points_to_read]
        points = np.array([list(map(float, row.split())) for row in points])
        return points

def get_points_and_class(file_dict, class_dict, n_points): ### Needs to iterate through list. return np.array(point_cloud, and class)
    point_cloud = list()
    point_cloud_class = list()
    for row in file_dict:
        point_cloud_class.append(class_dict[list(row.items())[0][0]])  #Hacky fix.
        point_cloud.append(read_off_file_into_nparray(list(row.items())[0][1], n_points))
    return np.array(point_cloud), np.array(point_cloud_class)

def remove_small_point_clouds(train_list, threshold):
    new_list = list()
    for file_dict in train_list:
            point_cloud = read_off_file_into_nparray(list(file_dict.items())[0][1], n_points_to_read=None)
            if point_cloud.shape[0] >= threshold:
                new_list.append(file_dict)
    return new_list

def generate_random_rotation_matrix():
    pass

def apply_random_rotation(point_cloud):
    rotation_matrix = generate_random_rotation_matrix()
    return np.dot(point_cloud, rotation_matrix)