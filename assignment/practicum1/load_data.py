import numpy as np
import os
import glob
import cv2
import scipy.io


def load_data():
    data_dir = os.path.join(os.environ['PRACTICUM1_DATA_DIR'], 'classification')
    
    train_dir = os.path.join(data_dir, 'train')
    test_dir = os.path.join(data_dir, 'test')

    X_train, y_train = load_data_from_dir(train_dir)
    X_test, y_test = load_data_from_dir(test_dir)

    return X_train, X_test, y_train, y_test


def load_data_from_dir(dir_path):
    pedestrian_dir = os.path.join(dir_path, 'pedestrian')
    non_pedestrian_dir = os.path.join(dir_path, 'non_pedestrian')
 
    pedestrian_filepaths = [os.path.join(pedestrian_dir, filename) for filename in os.listdir(pedestrian_dir)] 
    non_pedestrian_filepaths = [os.path.join(non_pedestrian_dir, filename) for filename in os.listdir(non_pedestrian_dir)] 

    images = []
    labels = []
    for image_path in pedestrian_filepaths:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.flatten()
            images.append(image)
            labels.append(True)
    
    for image_path in non_pedestrian_filepaths:
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = image.flatten()
            images.append(image)
            labels.append(False)
    
    images = np.array(images)
    labels = np.array(labels)

    return images, labels 


def load_bbox_data(filename):
    filepath = os.path.join(os.environ['PRACTICUM1_DATA_DIR'], filename)
    return np.loadtxt(fname=filepath, delimiter=',')


def load_calib(filename):
    filepath = os.path.join(os.environ['PRACTICUM1_DATA_DIR'], filename)
    calib_params = {}
    with open(filepath) as f:
        for line in f:
            key, value = line.split(':')
            calib_params[key] = np.array(list(map(float, value.split())))
    return calib_params 


def limit_class_members(data, limit):
    peds_count = 0
    non_peds_count = 0
    peds = []
    non_peds = []
    for entry in data:
        if peds_count < limit and entry[1] == True:
            peds.append(entry)
            peds_count +=1
        if non_peds_count < limit and entry[1] == False:
            non_peds.append(entry)
            non_peds_count +=1
    
    lim_data = np.concatenate([peds, non_peds])
    return lim_data

