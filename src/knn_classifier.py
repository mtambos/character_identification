__author__ = 'mtambos'

from __future__ import division, print_function

import os

import pandas as pd
from PIL import Image
import numpy as np


def to_gray_scale(img):
    img_array = np.asarray(img)
    luminosity = lambda x: 0.21*x[0] + 0.72*x[1] + 0.07*x[2]
    return np.apply_along_axis(func1d=luminosity, axis=2, arr=img_array)


def load_files(path):
    # get files list
    files_list = os.walk(path).next()[2]
    # load bmp files. traditional for instead of comprehension due to the need of closing file pointers
    img_files = pd.DataFrame(columns=range(400), index=np.array([], dtype=np.int), dtype=np.float)
    for f in files_list:
        name, ext = os.path.splitext(f)
        # use only bmp files
        if ext.lower() == '.bmp':
            with file(os.path.join(path, f), 'rb') as img:
                bitmap = Image.open(img)
                bands = bitmap.getbands()
                # check whether the image is color or b/w
                if len(bands) == 3:
                    # convert to gray scale and append
                    bitmap = to_gray_scale(bitmap).flatten()
                elif len(bands) == 1:
                    bitmap = np.asarray(bitmap).flatten()
                # add image as a row with the file name as key
                img_files.loc[int(name)] = bitmap
    # sort the indexes so they coincide with the label's indexes
    img_files.sort_index(inplace=True)
    return img_files


def distance(v, r):
    # subtract r (a row) from each row in v (a matrix),
    # square the individual elements,
    # sum all elements per row and finally
    # take the square root of each row
    return np.sum((v - r)**2, axis=1)**0.5


def get_nearest_neighbors(train_set, labels, point, k):
    # calculate the distance from point to all points in train_set
    distances = distance(train_set, point)
    # choose the k smallest distances' indexes
    indexes = np.argpartition(distances.values, kth=k)[:k]
    # return the k smallest labels and distances
    return labels.iloc[indexes], distances.iloc[indexes]


def classify(train_set, test_set, labels, k):
    # create data frame for the results and set its index's name
    classes = pd.DataFrame(columns=['Class'])
    classes.index.name = 'ID'
    for i, r in enumerate(test_set.iterrows()):
        # get the k points in the train set
        # nearest to the point r in the test set
        knn, distances = get_nearest_neighbors(train_set, labels, r[1], k)
        value_counts = knn.Class.value_counts()
        # value_counts[0] = 1 means that all
        # k training points have different labels
        # TODO: check case where 2 or more labels
        # TODO: have the same amount of counts
        # TODO: (and it's higher than 1)
        if value_counts[0] > 1:
            winner = value_counts.index[0]
        else:
            index = np.argmin(distances.values)
            winner = knn.iloc[index, 0]
        classes.loc[test_set.index[i]] = winner
    return classes


def optimize_k(base_path):
    # load labels
    labels_path = os.path.join(base_path, 'trainLabels.csv')
    labels = pd.read_csv(labels_path, index_col=0, dtype={'ID': np.int, 'Class': np.str})
    # load train set
    train_path = os.path.join(base_path, 'train')
    train_set = load_files(train_path)
    train_set_index = train_set.index
    # select random subset of train set as test set
    test_set_size = len(train_set) // 3
    test_indexes = set(np.random.choice(train_set_index, test_set_size))
    test_set = train_set.loc[list(test_indexes)]
    # select the labels corresponding to the test set
    test_labels = labels.loc[list(test_indexes)]
    # remove test elements from train set
    train_set = train_set.loc[list(set(train_set_index) - test_indexes)]

    accuracies = {}
    for k in range(1, 10):
        print('k=%s' % k)
        # classify the test set with knn
        classes = classify(train_set, test_set, labels, k)
        # check what portion of the test were correctly classified
        # TODO: select new random subset of train as test
        acc = (classes == test_labels).sum()/len(test_labels)
        accuracies[k] = acc

    return accuracies


def run(base_path, k):
    labels_path = os.path.join(base_path, 'trainLabels.csv')
    labels = pd.read_csv(labels_path, index_col=0, dtype={'ID': np.int, 'Class': np.str})
    train_path = os.path.join(base_path, 'train')
    train_set = load_files(train_path)
    test_path = os.path.join(base_path, 'test')
    test_set = load_files(test_path)
    classes = classify(train_set, test_set, labels, k)
    classes.to_csv('submission%sk.csv' % k)
