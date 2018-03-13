import numpy as np
import os

def get_filenames_and_class(data_dir):
    train_list = []
    test_list = []
    classes = os.listdir(data_dir)
    for point_class in classes:
        train_dir = os.path.join(data_dir, point_class + '/train')
        test_dir = os.path.join(data_dir, point_class + '/test')
        for file in os.listdir(train_dir):
            train_list.append({point_class: os.path.join(data_dir, point_class, 'train', file)})
        for file in os.listdir(test_dir):
            test_list.append({point_class: os.path.join(data_dir, point_class, 'test', file)})
    return train_list, test_list

def generate_class_str_to_num_dict(data_dir):
    classes = sorted(os.listdir(data_dir))
    class_dict = {}
    for pt_class, i in enumerate(classes):
        class_dict[i] = pt_class
    return class_dict

def read_off_file_into_nparray(fname):
    with open(fname) as f:
        content = f.readlines()
        if 'OFF' not in content[0]:
            print('File is not an OFFSET file.')
        else:
            n_points = int(content[1].split()[0])
            points = content[2:n_points + 2]
            points = np.array([list(map(float, row.split())) for row in points])
            return points

def get_points_and_class(file_dict, class_dict):
    point_cloud_class = class_dict[list(file_dict.items())[0][0]]
    point_cloud = read_off_file_into_nparray(list(file_dict.items())[0][1])
    return point_cloud_class, point_cloud