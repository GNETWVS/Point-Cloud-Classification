import numpy as np

def read_off_file_into_nparray(fname, n_points_to_read):
    with open(fname) as f:
        content = f.readlines()
        n_points = int(content[1].split()[0])
        points = content[2:n_points + 2]
        if n_points_to_read is not None:
            points = points[:n_points_to_read]
        points = np.array([list(map(float, row.split())) for row in points])
        return points

def get_points_and_class(file_dict, class_dict, n_points):
    point_cloud = list()
    point_cloud_class = list()
    for row in file_dict:
        point_cloud_class.append(class_dict[list(row.items())[0][0]])
        point_cloud.append(read_off_file_into_nparray(list(row.items())[0][1], n_points))
    return np.array(point_cloud), np.array(point_cloud_class)

def generate_random_rotation_matrix():
    pass

def apply_random_rotation(point_cloud):
    rotation_matrix = generate_random_rotation_matrix()
    return np.dot(point_cloud, rotation_matrix)