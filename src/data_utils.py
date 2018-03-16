import numpy as np
from math import pi, cos, sin


def read_off_file_into_nparray(fname, n_points_to_read):
    with open(fname) as f:
        content = f.readlines()
        n_points = int(content[1].split()[0])
        points = content[2:n_points + 2]
        if n_points_to_read is not None:
            points = points[:n_points_to_read]
        points = np.array([list(map(float, row.split())) for row in points])
        return points


def get_points_and_class(file_dict, class_dict, n_points, rotate=False):
    point_cloud_list = list()
    point_cloud_class_list = list()
    for row in file_dict:
        if rotate:
            point_cloud_class_list.append(class_dict[list(row.items())[0][0]])
            point_cloud = read_off_file_into_nparray(list(row.items())[0][1], n_points)
            rotation_matrix = generate_random_rotation_matrix()
            point_cloud = np.dot(point_cloud, rotation_matrix)
            point_cloud_list.append(list(point_cloud))
        else:
            point_cloud_class_list.append(class_dict[list(row.items())[0][0]])
            point_cloud_list.append(read_off_file_into_nparray(list(row.items())[0][1], n_points))
    return np.array(point_cloud_list), np.array(point_cloud_class_list)


def generate_random_rotation_matrix():
    theta_x, theta_y, theta_z = np.random.uniform(low=-0.05, high=0.05, size=3) * 2 * pi
    rot_x = np.array([[1,0,0],[0,cos(theta_x),-sin(theta_x)],[0,sin(theta_x), cos(theta_x)]])
    rot_y = np.array([[cos(theta_y),0,sin(theta_y)],[0,1,0],[-sin(theta_y),0,cos(theta_y)]])
    rot_z = np.array([[cos(theta_z),-sin(theta_z),0],[sin(theta_z),cos(theta_z),0],[0,0,1]])
    return np.dot(np.dot(rot_x, rot_y), rot_z)
