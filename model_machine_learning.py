"""
Author: Oh, Seungwoo
Description: Machine learning models (DL, RF, XGB, SVM)
Last modified: 2024.09.09.
"""

import time

import numpy as np
import xgboost
from sklearn import tree, svm, ensemble
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearnex import patch_sklearn

from utils import *

slicing_time = 5
patch_sklearn()

random_seed = 10


class ModelMachineLearning:
    def __init__(self, data_split, num_of_sensors, scenario, only_phone):
        """
        Args:
            - data split -> "random", "user_indep"
            - num_of_sensors -> 1:lacc, 2:gyro, 3:mag
            - scenario -> "disabled", "wheelchairs", "six"
            - only_phone = True, False
        """
        self.data_split = data_split
        self.num_of_sensors = num_of_sensors
        self.scenario = scenario

        if scenario == "disabled":
            self.num_classes = 3
            self.classes = ['still', 'able-bodied', 'disabled']
        elif scenario == "wheelchairs":
            self.num_classes = 4
            self.classes = ['still', 'walk', 'walk-aid', 'wheelchairs']
        elif scenario == "six_activities":
            self.num_classes = 6
            self.classes = ['still', 'walking', 'walker', 'crutches', 'manual', 'power']
        else:
            print("wrong scenario")

        if only_phone:  # 1~120 phone
            n = 172800 // 5
            self.groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + \
                          [3 for _ in range(n)] + [4 for _ in range(n)]
            self.device = 'only_phone'
        else:  # 61~120 phone+watch
            n = 86400 // 5
            self.groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + \
                          [3 for _ in range(n)] + [4 for _ in range(n)]
            self.device = 'phone_watch'
        print(self.scenario, self.classes, self.num_classes)

    # Decision Tree
    def DT(self, data, label, split_type):
        model_type = "Decision Tree"
        print(model_type)

        # data split
        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            split_result = logo.split(data, label, self.groups)
        else:
            return -1

        fold_idx = 0
        for train_index, test_index in split_result:
            # data
            x_train_features, x_test_features = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            x_train_features = np.float32(x_train_features)
            x_test_features = np.float32(x_test_features)
            x_train_features = np.nan_to_num(x_train_features)
            x_test_features = np.nan_to_num(x_test_features)

            # model
            model = tree.DecisionTreeClassifier(random_state=random_seed)
            model.fit(x_train_features, y_train)

            # results
            score = model.score(x_test_features, y_test)  # mean accuracy
            y_pred = model.predict(x_test_features)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            accuracy = accuracy_score(y_test, y_pred)
            fscore = f1_score(y_test, y_pred, average='macro')
            confusion = confusion_matrix(y_test, y_pred)

            f = create_result_file(scenario=self.scenario, device=self.device, model_type="DT", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)

            fold_idx += 1

    def RandomForest(self, data, label, split_type):
        model_type = "Random Forest"
        print(model_type)

        # data split
        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            split_result = logo.split(data, label, self.groups)
        else:
            return -1

        fold_idx = 0
        for train_index, test_index in split_result:
            # data
            x_train_features, x_test_features = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            x_train_features = np.float32(x_train_features)
            x_test_features = np.float32(x_test_features)
            x_train_features = np.nan_to_num(x_train_features)
            x_test_features = np.nan_to_num(x_test_features)

            # model
            model = ensemble.RandomForestClassifier(random_state=random_seed)
            model.fit(x_train_features, y_train)

            # results
            score = model.score(x_test_features, y_test)  # mean accuracy
            y_pred = model.predict(x_test_features)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            accuracy = accuracy_score(y_test, y_pred)
            fscore = f1_score(y_test, y_pred, average='macro')
            confusion = confusion_matrix(y_test, y_pred)

            f = create_result_file(scenario=self.scenario, device=self.device, model_type="RF", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)


            fold_idx += 1

    def XGBoost(self, data, label, split_type):
        model_type = "XGBoost"
        print(model_type)

        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            print(data.shape)
            print(label.shape)
            split_result = logo.split(data, label, self.groups)
        else:
            return -1

        fold_idx = 0

        # train test set
        for train_index, test_index in split_result:
            print('fold: ', fold_idx)
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]
            model = xgboost.XGBClassifier()
            model.fit(x_train, y_train, verbose=True)

            y_pred = model.predict(x_test)

            report = classification_report(y_test, y_pred, target_names=self.classes)
            accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
            fscore = f1_score(y_test, y_pred, average='macro')
            confusion = confusion_matrix(y_test, y_pred)

            f = create_result_file(scenario=self.scenario, device=self.device, model_type="XGBoost", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)

            fold_idx += 1

    def SVM(self, data, label, split_type):
        model_type = "SVM"
        print(model_type)

        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            split_result = logo.split(data, label, self.groups)
        else:
            return -1

        fold_idx = 0
        for train_index, test_index in split_result:
            x_train_features, x_test_features = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            # normalize
            x_train_mean = x_train_features.mean(axis=0)
            x_train_std = x_train_features.std(axis=0)
            x_train_norm = (x_train_features - x_train_mean) / x_train_std
            x_test_mean = x_test_features.mean(axis=0)
            x_test_std = x_test_features.std(axis=0)
            x_test_norm = (x_test_features - x_test_mean) / x_test_std
            # change datatype to float32
            x_train_norm = np.float32(x_train_norm)
            x_test_norm = np.float32(x_test_norm)
            x_train_norm = np.nan_to_num(x_train_norm)
            x_test_norm = np.nan_to_num(x_test_norm)
            # flatten y train
            y_train_flat = y_train.flatten()

            # best parameter is already found
            final_model = svm.SVC(C=10, kernel='rbf', gamma=0.01, random_state=random_seed)
            final_model.fit(x_train_norm, y_train_flat)

            score = final_model.score(x_test_norm, y_test)
            y_pred = final_model.predict(x_test_norm)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            confusion = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            fscore = f1_score(y_test, y_pred, average='macro')

            f = create_result_file(scenario=self.scenario, device=self.device, model_type="SVM", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)

            fold_idx += 1
