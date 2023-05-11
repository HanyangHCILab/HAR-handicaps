import os
import time

import numpy as np

import model_attention_based
import model_deep_learning
import model_extra
import model_machine_learning
from utils import *

# path = "C:\\Users\\user\\Documents\\2022_2\\HAR\\data\\4_data_concat"

CheckTimeStartTime = time.perf_counter()

result_acc = []

model_type = "multi_LSTM"  # "uni_DNN", "uni_CNN", "uni_LSTM", "multi_DNN", "multi_CNN", "multi_LSTM, DT, RF, SVM, NB"
isIndoorTrain = False
batch_size = 128
num_classes = 12
slicing_time = 5
epochs = 20
drop_out = 0

phone_watch = False # 얘는 계속 false 일듯. phone+watch 데이터 test 할 때 사용
valid = False # valid한 45명만 test 할 때 사용

isGpsIncluded = False
# num_of_sensors = 4  # 1:acc, 2:gyro, 3:mag, 4:grav
# scenario = "disabled" # 'disabled', 'wheelchairs', 'indoor', 'outdoor', 'all-6', 'all-12'

indoor = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 19, 21, 23, 25, 27, 29, 31, 33, 35, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79, 81, 83, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 113, 115]
outdoor = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24, 26, 28, 30, 32, 34, 37, 39, 41, 43, 45, 47, 49, 51, 53, 55, 57, 59, 62, 64, 66, 68, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116]

if __name__ == '__main__':
    for model_type in ["XGB"]: # "DT", "RF", "SVM", "multi_CNN", "multi_LSTM"
        for split in ['user-indep', 'random']: # 'random', 'user-indep'
            for scenario in ['all-6', 'disabled', 'wheelchairs']: # 'all-6', 'disabled', 'wheelchairs','indoor', 'outdoor','all-6', 'all-12'
                for num_of_sensors in [3, 2, 1]:

                    print(model_type, split, scenario, num_of_sensors)

                    ### 여기부터 들여쓰기 했음
                    if isGpsIncluded:
                        gps = "GPS"
                    else:
                        gps = "X"
                    dl = model_deep_learning.ModelDeepLearning(data_split=split, num_of_sensors=num_of_sensors,
                                                               is_gps_included=isGpsIncluded, scenario=scenario)
                    ml = model_machine_learning.ModelMachineLearning(data_split=split, num_of_sensors=num_of_sensors,
                                                                     is_gps_included=isGpsIncluded, scenario=scenario)
                    extra = model_extra.ModelExtra(data_split=split, num_of_sensors=num_of_sensors, scenario=scenario)
                    attention = model_attention_based.ModelAttentionBased(data_split=split,
                                                                          num_of_sensors=num_of_sensors,
                                                                          scenario=scenario)

                    # if phone+watch valid 45 data
                    if valid:
                        data_path = 'E:\\'

                        data_phone = np.load(
                            os.path.join(data_path, "phone_watch_npy\\5s_valid_phone_data_obj.npy"))  # col 확인 필요
                        data_watch = np.load(
                            os.path.join(data_path, "phone_watch_npy\\5s_valid_watch_data_obj.npy"))
                        data_phone_features = np.load(
                            os.path.join(data_path, "phone_watch_npy\\5s_valid_phone_feature_data_obj.npy"))
                        data_watch_features = np.load(
                            os.path.join(data_path, "phone_watch_npy\\5s_valid_watch_feature_data_obj.npy"))

                        if num_of_sensors == 3:  # acc, gyro, mag
                            data_phone = data_phone[:, :, :9]
                            data_watch = data_watch[:, :, :9]
                            data_phone_features = data_phone_features[:, :9*11]
                            data_watch_features = data_watch_features[:, :9*11]
                        if num_of_sensors == 2:  # acc, gyro
                            data_phone = data_phone[:, :, :6]
                            data_watch = data_watch[:, :, :6]
                            data_phone_features = data_phone_features[:, :6*11]
                            data_watch_features = data_watch_features[:, :6*11]
                        if num_of_sensors == 1:  # acc
                            data_phone = data_phone[:, :, :3]
                            data_watch = data_watch[:, :, :3]
                            data_phone_features = data_phone_features[:, :3*11]
                            data_watch_features = data_watch_features[:, :3*11]

                        if scenario == "all-6":
                            label = np.load(os.path.join(data_path, "phone_watch_npy\\5s_valid_label_obj.npy"))
                        if scenario == "disabled":
                            label = np.load(os.path.join(data_path, "phone_watch_npy\\5s_valid_disabled_label.npy"))
                        if scenario == "wheelchairs":
                            label = np.load(os.path.join(data_path, "phone_watch_npy\\5s_valid_wheelchairs_label.npy"))

                        data = np.concatenate([data_phone, data_watch], axis=2)
                        data_features = np.concatenate([data_phone_features, data_watch_features], axis=1)
                        print(data.shape, data_features.shape)

                    # if phone+watch 57~120 - data load & sensor variation
                    if phone_watch:
                        data_path = 'E:\\'

                        data_phone = np.load(
                            os.path.join(data_path, "phone_watch_npy\\x_phone_reorder_col.npy"))  # col 확인 필요
                        data_watch = np.load(
                            os.path.join(data_path, "phone_watch_npy\\x_watch_reorder_col.npy"))
                        data_phone_features = np.load(
                            os.path.join(data_path, "phone_watch_npy\\features_data_phone_object.npy"))
                        data_watch_features = np.load(
                            os.path.join(data_path, "phone_watch_npy\\features_data_watch_object.npy"))

                        if num_of_sensors == 3:  # acc, gyro, mag
                            data_phone = data_phone[:, :, :9]
                            data_watch = data_watch[:, :, :9]
                            data_phone_features = data_phone_features[:, :9*11]
                            data_watch_features = data_watch_features[:, :9*11]
                        if num_of_sensors == 2:  # acc, gyro
                            data_phone = data_phone[:, :, :6]
                            data_watch = data_watch[:, :, :6]
                            data_phone_features = data_phone_features[:, :6*11]
                            data_watch_features = data_watch_features[:, :6*11]
                        if num_of_sensors == 1:  # acc
                            data_phone = data_phone[:, :, :3]
                            data_watch = data_watch[:, :, :3]
                            data_phone_features = data_phone_features[:, :3*11]
                            data_watch_features = data_watch_features[:, :3*11]

                        # # phone + watch
                        # data = np.concatenate([data_phone, data_watch], axis=2)
                        # data_features = np.concatenate([data_phone_features, data_watch_features], axis=1)
                        # print(data.shape, data_features.shape)
                        # # watch only
                        # data = data_watch
                        # data_features = data_watch_features

                        # # not-valid
                        # if scenario == "all-6":
                        #     label = np.load(os.path.join(data_path, "phone_watch_npy\\y_phone_reorder.npy"))
                        # if scenario == "disabled":
                        #     label = np.load(os.path.join(data_path, "phone_watch_npy\\y_reorder_disabled.npy"))
                        # if scenario == "wheelchairs":
                        #     label = np.load(os.path.join(data_path, "phone_watch_npy\\y_reorder_wheelchairs.npy"))

                    # if only phone 1~120
                    if not valid and not phone_watch:
                        data, label = load_data(scenario=scenario) # raw data for DL
                        data_path = 'E:\\' # feature data for ML
                        data_features = np.load(os.path.join(data_path, "5s_features_data_object.npy")) # 5sec window
                        # data_features = np.load(os.path.join(data_path, "2s_features_data_object.npy")) # 2sec 50% window
                        # data_features = np.load("feature_sliced.npy") # test
                        # print(data_features.shape)

                        # # slice to 1~56
                        # data = data[:80640, :, :]
                        # label = label[:80640]
                        # data_features = data_features[:80640, :]

                        # # slice to 57~120
                        # data = data[80640:, :, :]
                        # label = label[80640:]
                        # data_features = data_features[80640:, :]

                        # x_train, x_test, y_train, y_test = split_data(split_type=split, data=data, label=label)

                        ### number of sensors
                        # if isGpsIncluded: # gps included
                            # x_train_gps = x_train[:, :, 12:]
                            # x_test_gps = x_test[:, :, 12:]
                        # if num_of_sensors == 4:  # acc, gyro, mag, gra
                        #     # x_train = x_train[:, :, :12]
                        #     # x_test = x_test[:, :, :12]
                        #     data_features = data_features[:, :12]
                        if num_of_sensors == 3:  # acc, gyro, mag
                            # x_train = x_train[:, :, :9]
                            # x_test = x_test[:, :, :9]
                            data = data[:, :, :9]
                            data_features = data_features[:, :9*11]
                        if num_of_sensors == 2:  # acc, gyro
                            # x_train = x_train[:, :, :6]
                            # x_test = x_test[:, :, :6]
                            data = data[:, :, :6]
                            data_features = data_features[:, :6*11]
                        if num_of_sensors == 1:  # acc
                            # x_train = x_train[:, :, :3]
                            # x_test = x_test[:, :, :3]
                            data_features = data_features[:, :3*11]
                            data = data[:, :, :3]
                        # if isGpsIncluded:  # gps included
                        #     x_train = np.concatenate([x_train, x_train_gps], axis=-1)
                        #     x_test = np.concatenate([x_test, x_test_gps], axis=-1)

                        # ### sensor variations # acc 1~3, gyro 3~6, mag 6~9, grav 9~12, linacc 12~15
                        # data_list = []
                        # if sensors['acc']: data_list.append(data[:,:,0:3])
                        # if sensors['gyro']: data_list.append(data[:,:,3:6])
                        # if sensors['mag']: data_list.append(data[:,:,6:9])
                        # if sensors['grav']: data_list.append(data[:,:,9:12])
                        # # if sensors['linacc']: data_list.append(data[:,:,12:15])
                        # data_new = np.concatenate(data_list, axis=2)
                        # print(data.shape)
                        # print(len(data_list))
                        # print(data_new.shape)
                        print(data.shape)

                    ### model
                    if model_type == "uni_DNN":
                        dl.run_uni_DNN(data, label, split_type=split)
                    if model_type == "uni_CNN":
                        dl.run_uni_CNN(data, label, split_type=split)
                    if model_type == "uni_LSTM":
                        dl.run_uni_LSTM(data, label, split_type=split)
                    if model_type == "multi_DNN":
                        dl.run_multi_DNN(data, label, split_type=split)
                    if model_type == "multi_CNN":
                        dl.run_multi_CNN(data, label, split_type=split)
                    if model_type == "multi_LSTM":
                        dl.run_multi_LSTM(data, label, split_type=split)
                    if model_type == "DT":
                        ml.DT(data_features, label, split_type=split)
                    if model_type == "RF":
                        ml.RandomForest(data_features, label, split_type=split)
                    if model_type == "SVM":
                        ml.SVM(data_features, label, split_type=split)
                    if model_type == "NB":
                        ml.NaiveBayes(data_features, label, split_type=split)
                    if model_type == "KNN":
                        ml.KNN(data_features, label, split_type=split)
                    if model_type == "LR":
                        ml.LogisticRegression(data_features, label, split_type=split)
                    if model_type == "XGB":
                        extra.XGBoost(data_features, label, split_type=split)
                    if model_type == "attention":
                        attention.run_attention(data, label, split_type=split)
                    # if model_type == "uni_DNN":
                    #     dl.run_uni_DNN(x_train, x_test, y_train, y_test)
                    # if model_type == "uni_CNN":
                    #     dl.run_uni_CNN(x_train, x_test, y_train, y_test)
                    # if model_type == "uni_LSTM":
                    #     dl.run_uni_LSTM(x_train, x_test, y_train, y_test)
                    # if model_type == "multi_DNN":
                    #     dl.run_multi_DNN(x_train, x_test, y_train, y_test)
                    # if model_type == "multi_CNN":
                    #     dl.run_multi_CNN(x_train, x_test, y_train, y_test)
                    # if model_type == "multi_LSTM":
                    #     dl.run_multi_LSTM(x_train, x_test, y_train, y_test)
                    # if model_type == "DT":
                    #     ml.DT(x_train, x_test, y_train, y_test)
                    # if model_type == "RF":
                    #     ml.RandomForest(x_train, x_test, y_train, y_test)
                    # if model_type == "SVM":
                    #     ml.SVM(x_train, x_test, y_train, y_test)
                    # if model_type == "NB":
                    #     ml.NaiveBayes(x_train, x_test, y_train, y_test)

