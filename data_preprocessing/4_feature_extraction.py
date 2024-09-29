"""
Author: Oh, Seungwoo
Description: (4) Feature extraction from data file - for ML models
Last modified: 2024.09.09.
"""

import numpy as np
from scipy.stats import skew, kurtosis, iqr
from numpy.fft import fft
import os

from sklearnex import patch_sklearn

patch_sklearn()
num_of_features = 11
path = 'D:\\data_npy'

object_npy = 'data_object.npy'    # 'data_object.npy', 'phone_data_object.npy', 'watch_data_object.npy'

if object_npy == 'data_object.npy':
    features_object_npy = 'features_data_object.npy'  
if object_npy == 'phone_data_object.npy':
    features_object_npy = 'phone_features_data_object.npy'  
if object_npy == 'watch_data_object.npy':
    features_object_npy = 'watch_features_data_object.npy'  


def single_feature_extract(data): # single sensor
    """
    :param data: (300, ) # np.array
    :return: features - (11, )
    """
    # print(data.shape)
    mean = np.average(data)
    var = np.var(data)
    minimum = np.amax(data)
    maximum = np.amin(data)
    signal_energy = np.sum(np.square(data)) / data.shape[0]  # RMS
    skw = skew(data)
    kur = kurtosis(data)
    interquartile_range = iqr(data)
    fft_magnitude = max(abs(fft(data)))
    zcr = (data[:-1] * data[1:] < 0).sum() # zero crossing rate
    median = np.median(data)

    features = np.array(
        [mean, var, minimum, maximum, signal_energy, skw, kur, interquartile_range, fft_magnitude, zcr, median])

    return features


def multiple_feature_extract(data, num_of_sensors): # multiple sensors
    """
    :param data: (300, 9)
    :return: features - (11, 9)
    """
    data = np.transpose(data)

    result = np.empty((num_of_sensors * 3, num_of_features))  # 3: x, y, z axis # 9: number of features
    idx = 0
    for d in data:
        feature = single_feature_extract(d)
        result[idx] = feature
        idx += 1

    return result


def concat_features(data, num_of_sensors):
    """
    :param data: (, 300, 9)
    :return: features - (, 9 * 11)
    """

    result = np.empty((data.shape[0], num_of_sensors * 3 * num_of_features)) # 3: x, y, z axis # 9: number of features
    idx = 0
    for d in data:
        feature = multiple_feature_extract(d, num_of_sensors)
        feature = feature.flatten()
        result[idx] = feature
        idx += 1
        print(str(idx) + '/' + str(data.shape[0]) + ' line extracted!')

    return result


if __name__ == "__main__":
    num_of_sensors = 3

    data = np.load(os.path.join(path, object_npy), allow_pickle=True)
    print(data.shape)

    data_features = concat_features(data, num_of_sensors=num_of_sensors)
    print(data_features.shape)

    np.save(os.path.join(path, features_object_npy), data_features)

