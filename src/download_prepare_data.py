import glob
import os
import pickle
from math import floor
from random import shuffle
from urllib.request import urlopen
from zipfile import ZipFile

from data_utils import read_off_file_into_nparray


def download_datasets(args):
    model_net10_url = 'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip'

    print('[*] Downloading and unzipping datasets.')

    unzip_files(model_net10_url, args.data_dir)
    os.remove(os.path.join(args.Net10_data_dir, '.DS_Store'))
    os.remove(os.path.join(args.Net10_data_dir, 'README.txt'))


def unzip_files(url, destination):
    zip_resp = urlopen(url)
    temp_zip = open('/tmp/tempfile.zip', 'wb')
    temp_zip.write(zip_resp.read())
    temp_zip.close()
    zf = ZipFile('/tmp/tempfile.zip')
    zf.extractall(path=destination)
    zf.close()


def prepare_datasets(args):
    data = dict()
    data['class_dict'] = generate_class_str_to_num_dict(args.Net10_data_dir)
    master_list = get_filenames_and_class(args.Net10_data_dir)
    master_list = remove_small_point_clouds(master_list, args.small_sample_threshold)
    shuffle(master_list)
    n_samples = len(master_list)
    data['train_list'] = master_list[:floor(0.8*n_samples)]
    data['eval_list'] = master_list[floor(0.8*n_samples):floor(0.9*n_samples)]
    data['test_list'] = master_list[floor(0.9*n_samples):]
    pickle.dump(data, open(os.path.join(args.data_dir, 'data.pickle'), "wb"))


def get_filenames_and_class(data_dir):
    master_list = list()
    classes = os.listdir(data_dir)
    for point_class in classes:
        train_dir = os.path.join(data_dir, point_class + '/train')
        test_dir = os.path.join(data_dir, point_class + '/test')
        for file in glob.glob(os.path.join(train_dir, '*.off')):
            master_list.append({point_class: file})
        for file in glob.glob(os.path.join(test_dir, '*.off')):
            master_list.append({point_class: file})
    return master_list


def generate_class_str_to_num_dict(data_dir):
    classes = sorted(os.listdir(data_dir))
    class_dict = {}
    for pt_class, i in enumerate(classes):
        class_dict[i] = pt_class
    return class_dict


def remove_small_point_clouds(train_list, threshold):
    new_list = list()
    for file_dict in train_list:
            point_cloud = read_off_file_into_nparray(list(file_dict.items())[0][1], n_points_to_read=None)
            if point_cloud.shape[0] >= threshold:
                new_list.append(file_dict)
    return new_list