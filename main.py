"""
Author: Oh, Seungwoo
Description: Main file.
Last modified: 2024.09.09.
"""

import model_deep_learning
import model_machine_learning
import numpy as np
from utils import *

data_path = "D:\\data_npy"

only_phone = False  # 1~120 phone
phone_watch = True  # 61~120 phone+watch

scenario = "wheelchairs"  # "disabled", "wheelchairs", "six_activities"
num_of_sensors = 1  # 1, 2, 3
model_type = "MLP"  # "DT", "RF", "XGB", "SVM", "MLP", "CNN", "LSTM", "TR"
split = "random" # "random", "user-indep"

if __name__ == '__main__':
    print(model_type, split, scenario, num_of_sensors)

    dl = model_deep_learning.ModelDeepLearning(data_split=split,
                                            num_of_sensors=num_of_sensors,
                                            scenario=scenario,
                                            only_phone=only_phone)
    ml = model_machine_learning.ModelMachineLearning(data_split=split,
                                                    num_of_sensors=num_of_sensors,
                                                    scenario=scenario,
                                                    only_phone=only_phone)

    if only_phone:
        # load data
        data = np.load(os.path.join(data_path, "data_object.npy"))
        data_features = np.load(os.path.join(data_path, "features_data_object.npy"))

        # load label
        if scenario == "disabled":
            label = np.load(os.path.join(data_path, "disabled_label_object.npy"))
        if scenario == "wheelchairs":
            label = np.load(os.path.join(data_path, "wheelchairs_label_object.npy"))
        if scenario == "six_activities":
            label = np.load(os.path.join(data_path, "six_activities_label_object.npy"))

        # choose sensors
        if num_of_sensors == 3:  # lacc, gyro, mag
            data = data[:, :, :9]
            data_features = data_features[:, :9*11]
        if num_of_sensors == 2:  # lacc, gyro
            data = data[:, :, :6]
            data_features = data_features[:, :6*11]
        if num_of_sensors == 1:  # lacc
            data = data[:, :, :3]
            data_features = data_features[:, :3*11]
            
    if phone_watch:
        # load data
        data_phone = np.load(os.path.join(data_path, "phone_data_object.npy"))
        data_watch = np.load(os.path.join(data_path, "watch_data_object.npy"))
        data_phone_features = np.load(os.path.join(data_path, "phone_features_data_object.npy"))
        data_watch_features = np.load(os.path.join(data_path, "watch_features_data_object.npy"))

        # load label
        if scenario == "disabled":
            label = np.load(os.path.join(data_path, "phone_watch_disabled_label_object.npy"))
        if scenario == "wheelchairs":
            label = np.load(os.path.join(data_path, "phone_watch_wheelchairs_label_object.npy"))
        if scenario == "six_activities":
            label = np.load(os.path.join(data_path, "phone_watch_six_activities_label_object.npy"))

        # choose sensors
        if num_of_sensors == 3:  # lacc, gyro, mag
            data_phone = data_phone[:, :, :9]
            data_watch = data_watch[:, :, :9]
            data_phone_features = data_phone_features[:, :9 * 11]
            data_watch_features = data_watch_features[:, :9 * 11]
        if num_of_sensors == 2:  # lacc, gyro
            data_phone = data_phone[:, :, :6]
            data_watch = data_watch[:, :, :6]
            data_phone_features = data_phone_features[:, :6 * 11]
            data_watch_features = data_watch_features[:, :6 * 11]
        if num_of_sensors == 1:  # lacc
            data_phone = data_phone[:, :, :3]
            data_watch = data_watch[:, :, :3]
            data_phone_features = data_phone_features[:, :3 * 11]
            data_watch_features = data_watch_features[:, :3 * 11]

        data = np.concatenate([data_phone, data_watch], axis=2)
        data_features = np.concatenate([data_phone_features, data_watch_features], axis=1)

    print("Data shape:", data.shape, label.shape, data_features.shape)

    # DL model
    if model_type == "MLP":
        dl.MLP(data, label, split_type=split)
    if model_type == "CNN":
        dl.CNN(data, label, split_type=split)
    if model_type == "LSTM":
        dl.LSTM(data, label, split_type=split)
    if model_type == "TR":
        dl.Transformer(data, label, split_type=split)
    # ML model
    if model_type == "DT":
        ml.DT(data_features, label, split_type=split)
    if model_type == "RF":
        ml.RandomForest(data_features, label, split_type=split)
    if model_type == "XGB":
        ml.XGBoost(data_features, label, split_type=split)
    if model_type == "SVM":
        ml.SVM(data_features, label, split_type=split)

