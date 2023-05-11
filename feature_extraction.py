import numpy as np
from scipy.stats import skew, kurtosis, iqr
from numpy.fft import fft
from utils import *
from sklearn.model_selection import ShuffleSplit, LeaveOneGroupOut, TimeSeriesSplit

from sklearnex import patch_sklearn

patch_sklearn()
num_of_feataures = 11

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
        # [mean, var, minimum, maximum, signal_energy, skw, kur, interquartile_range, fft_magnitude])
        [mean, var, minimum, maximum, signal_energy, skw, kur, interquartile_range, fft_magnitude, zcr, median])

    return features


def multiple_feature_extract(data, num_of_sensors): # multiple sensors
    """
    :param data: (300, 9)
    :return: features - (11, 9)
    """
    data = np.transpose(data)

    result = np.empty((num_of_sensors * 3, num_of_feataures))  # 3: x, y, z axis # 9: number of features
    idx = 0
    for d in data:  # np.array?
        feature = single_feature_extract(d)
        result[idx] = feature
        idx += 1

    return result


def concat_features(data, num_of_sensors):
    """
    :param data: (58968, 300, 9)
    :return: features - (58968, 9 * 11)
    """

    result = np.empty((data.shape[0], num_of_sensors * 3 * num_of_feataures)) # 3: x, y, z axis # 9: number of features
    idx = 0
    for d in data:
        # d: (300, 12)
        feature = multiple_feature_extract(d, num_of_sensors)
        feature = feature.flatten()
        result[idx] = feature
        idx += 1
        print(str(idx) + '/' + str(data.shape[0]) + ' line extracted!')

    return result


if __name__ == "__main__":
    scenario = "all-6"  #'disabled', 'wheelchairs','indoor', 'outdoor','all-6', 'all-12'
    split = "random"  # random, forward, user-indep

    # data, label = load_data(scenario='all-6')
    # print(data.shape, label.shape)
    # data = np.load('E:\\2s_publish_data_object.npy')
    data = np.load('E:\\phone_watch_npy\\5s_valid_data_obj.npy', allow_pickle=True)
    # data = np.load('E:\\phone_watch_npy\\x_phone_reorder_col.npy')
    # data = np.load('E:\\phone_watch_npy\\x_watch_reorder_col.npy')
    print(data.shape)

    data_features = concat_features(data, num_of_sensors=6)
    print(data_features.shape)

    # data_path = 'C:\\Users\\user\\Documents\\2022_2\\HAR\\data\\4_data_concat'
    data_path = 'E:\\phone_watch_npy'
    # np.save(os.path.join(data_path, '2s_features_data_object.npy'), data_features)
    np.save(os.path.join(data_path, '5s_valid_features_data_obj.npy'), data_features)
    # np.save(os.path.join(data_path, 'features_data_object.npy'), data_features)
    # np.save(os.path.join(data_path, 'features_data_phone_object.npy'), data_features)
    # np.save(os.path.join(data_path, 'features_data_watch_object.npy'), data_features)

    ''' 
    for split in ["random", "user-indep"]:  # 'random', 'user-indep'
        for s in ['disabled', 'wheelchairs','indoor', 'outdoor','all-6', 'all-12']:  #'disabled', 'wheelchairs','indoor', 'outdoor','all-6', 'all-12'
            scenario = s

            # load data
            data, label = load_data(scenario=scenario)
            # split data (cross validation)
            rs = ShuffleSplit(n_splits=5, random_state=0)
            for train_index, test_index in rs.split(data):
                x_train = data[train_index]
                x_test = data[test_index]

            # x_train, x_test, y_train, y_test = split_data(split_type=split, data=data, label=label)
            # x_train = x_train[:,:,:12]
            # x_test = x_test[:,:,:12]

            print("x_train" + str(x_train.shape))
            print("x_test" + str(x_test.shape))

            x_train_features = concat_features(x_train, 4)
            x_test_features = concat_features(x_test, 4)
            print("x_train_features " + str(x_train_features.shape))
            print("x_test_features " + str(x_train_features.shape))

            np.save("x_train_features_" + str(split) + "_" + str(scenario) + "_sensors" + ".npy", x_train_features)
            np.save("x_test_features_" + str(split) + "_" + str(scenario) + "_sensors" + ".npy", x_test_features)
    '''