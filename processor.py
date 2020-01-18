import os
import numpy as np
import cv2
import pickle

# this is a helper function to get the labels


def get_dirlist(rootdir):
    dirlist = []
    with os.scandir(rootdir) as folders:
        for entry in folders:
                path = str(entry.path)
                dirlist.append(path.replace(rootdir+'\\', ''))
    return dirlist

# retrieves the data from images for machine learning


def get_data(main_path, x, y, root_dir):
    data = []
    labels = get_dirlist(root_dir)
    for label in labels:
        path = os.path.join(main_path, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_COLOR)
                resized_arr = cv2.resize(img_arr, (x, y))
                data.append([resized_arr, class_num])
            except IOError as e:
                print(e)
    return data

# processes the data  c = 1 is gray scale and c = 3 is color


def process_data(data, x, y, c, feature_name, label_name):
    features = []
    labels = []
    for a, b in data:
        features.append(a)
        labels.append(b)
    features = np.array(features)
    labels = np.array(labels)

    # reshape features and labels for training
    features = features.reshape(-1, x, y, c)
    labels = labels.reshape(-1, x, y, c)
    pickle.dump(features, open(feature_name, 'wb'))
    pickle.dump(labels, open(label_name, 'wb'))
