import time

import numpy as np
from numpy.fft import fft
from scipy.stats import skew, kurtosis, iqr
from sklearn import tree, svm, ensemble, linear_model, neighbors, naive_bayes
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearnex import patch_sklearn

from utils import *

slicing_time = 2
CheckTimeStartTime = time.perf_counter()
patch_sklearn()

# n = 176760//(50+48+50+48+50)
# groups = [0 for _ in range(int(50 * n) + 26)] + [1 for _ in range(int(48 * n) + 27)] + [2 for _ in range(int(50 * n) + 26)] + [3 for _ in range(int(48 * n) + 27)] + [4 for _ in range(int(50 * n) + 26)]
# n = 172800 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n)]
# n = 79200 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n + 1440)] # 1~56
# n = 93600 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n - 1440)] # 57~120
# n = 64800 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n)] # valid 45
n = 862560 // 5
groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n)] # 2 sec 50%

random_seed = 10

class ModelMachineLearning:
    def __init__(self, data_split, num_of_sensors, is_gps_included, scenario):
        """
        Args:
            - data split -> "random", "user_indep", "period_indep", "forward"
            - num_of_sensors -> 1:acc, 2:gyro, 3:mag, 4:grav
            - is_gps_included -> True / False
            - scenario -> "disabled", "wheelchairs", "indoor", "outdoor", "all-6", "all-12"
        """
        self.data_split = data_split
        self.num_of_sensors = num_of_sensors
        self.isGpsIncluded = is_gps_included
        self.scenario = scenario

        if scenario == "disabled":
            self.num_classes = 3
            self.classes = ['still', 'able-bodied', 'disabled']
        elif scenario == "wheelchairs":
            self.num_classes = 4
            self.classes = ['still', 'walk', 'walk-aid', 'wheelchairs']
        elif scenario == 'all-12':
            self.num_classes = 12
            self.classes = ['in-still', 'in-walking', 'in-manual', 'in-power', 'in-walker', 'in-crutches',
                            'out-still', 'out-walking', 'out-manual', 'out-power', 'out-walker', 'out-crutches']
        else:
            self.num_classes = 6
            self.classes = ['still', 'walking', 'manual', 'power', 'walker', 'crutches']

        print(self.scenario, self.classes, self.num_classes)


    # Decision Tree
    def DT(self, data, label, split_type):
        model_type = "Decision Tree"
        print(model_type)
        print(self.scenario)
        print(self.classes)
        print(self.num_of_sensors)

        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            print(data.shape)
            print(label.shape)
            split_result = logo.split(data, label, groups)
        else:
            return -1

        # print("x_train" + str(x_train.shape))
        #
        # x_train_features = np.load("features\\x_train_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        # x_test_features = np.load("features\\x_test_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        #
        # # split by number of sensors
        # x_train_features = split_feature(x_train_features, self.num_of_sensors)
        # x_test_features = split_feature(x_test_features, self.num_of_sensors)
        # print("x_train_features " + str(x_train_features.shape))

        fold_idx = 0
        for train_index, test_index in split_result:
            x_train_features, x_test_features = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            x_train_features = np.float32(x_train_features)
            x_test_features = np.float32(x_test_features)
            x_train_features = np.nan_to_num(x_train_features)
            x_test_features = np.nan_to_num(x_test_features)
            print("y_train " + str(y_train.shape))

            model = tree.DecisionTreeClassifier(random_state=random_seed)
            model.fit(x_train_features, y_train)

            # results
            score = model.score(x_test_features, y_test)  # mean accuracy
            y_pred = model.predict(x_test_features)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            accuracy = accuracy_score(y_test, y_pred)
            fscore = f1_score(y_test, y_pred, average='macro')
            confusion = confusion_matrix(y_test, y_pred)
            print(report)
            print(confusion)

            f = create_result_file(scenario=self.scenario, model_type="DT", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                                   slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)
            save_confusion_matrix_ML(y_test, y_pred, classes=self.classes, scenario=self.scenario, model_type="DT",
                                  split=self.data_split, num_of_sensors=self.num_of_sensors,
                                  is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)

            fold_idx += 1

    def SVM(self, data, label, split_type):
        model_type = "SVM"
        print(model_type)
        print(self.scenario)
        print(self.classes)
        print(self.num_of_sensors)

        # print("x_train " + str(x_train.shape))
        #
        # # load features
        # x_train_features = np.load("features\\x_train_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        # x_test_features = np.load("features\\x_test_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        # # split by number of sensors
        # x_train_features = split_feature(x_train_features, self.num_of_sensors)
        # x_test_features = split_feature(x_test_features, self.num_of_sensors)
        # print("x_train_features " + str(x_train_features.shape))


        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            print(label.shape)
            split_result = logo.split(data, label, groups)
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
            print("y_train " + str(y_train.shape))

            # # find the best parameter by grid search
            # params_grid = [{'kernel': ['lineaer', 'rbf', 'poly'], 'gamma': [0.001, 0.01, 0.1],
            #                 'C': [1, 10, 100]}]
            # model = GridSearchCV(svm.SVC(), params_grid, cv=5, verbose=3, n_jobs=4)
            # model.fit(x_train_features, y_train_flat)
            # #
            # # print('Best score for training data:', model.best_score_, "\n")
            # print('Best C:', model.best_estimator_.C, "\n")
            # print('Best Kernel:', model.best_estimator_.kernel, "\n")
            # print('Best Gamma:', model.best_estimator_.gamma, "\n")
            #
            # final_model = model.best_estimator_

            # if best parameter is already found
            # final_model = linear_model.SGDClassifier(random_state=random_seed)
            final_model = svm.SVC(C=10, kernel='rbf', gamma=0.1, random_state=random_seed)
            final_model.fit(x_train_norm, y_train_flat)

            score = final_model.score(x_test_norm, y_test)
            y_pred = final_model.predict(x_test_norm)
            # y_pred = np.argmax(y_pred)
            # y_test = np.argmax(y_test)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            confusion = confusion_matrix(y_test, y_pred)
            accuracy = accuracy_score(y_test, y_pred)
            fscore = f1_score(y_test, y_pred, average='macro')

            f = create_result_file(scenario=self.scenario, model_type="SVM", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                                   slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)
            save_confusion_matrix_ML(y_test, y_pred, classes=self.classes, scenario=self.scenario, model_type="SVM",
                                     split=self.data_split, num_of_sensors=self.num_of_sensors,
                                     is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)

            fold_idx += 1

            print('Performance time: ', time.perf_counter() - CheckTimeStartTime)

    def LogisticRegression(self, data, label, split_type):
        model_type = "Logistic Regression"
        print(model_type)
        print(self.scenario)
        print(self.classes)
        print(self.num_of_sensors)

        # print("x_train" + str(x_train.shape))
        #
        # x_train_features = np.load("features\\x_train_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        # x_test_features = np.load("features\\x_test_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")

        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            print(label.shape)
            split_result = logo.split(data, label, groups)
        else:
            return -1


        fold_idx = 0
        for train_index, test_index in split_result:
            x_train_features, x_test_features = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            # split by number of sensors
            x_train_features = split_feature(x_train_features, self.num_of_sensors)
            x_test_features = split_feature(x_test_features, self.num_of_sensors)
            print("x_train_features " + str(x_train_features.shape))

            x_train_features = np.float32(x_train_features)
            x_test_features = np.float32(x_test_features)
            x_train_features = np.nan_to_num(x_train_features)
            x_test_features = np.nan_to_num(x_test_features)

            model = linear_model.LogisticRegression(random_state=random_seed)
            print(model)
            model.fit(x_train_features, y_train)

            # results
            score = model.score(x_test_features, y_test)  # mean accuracy
            y_pred = model.predict(x_test_features)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            fscore = f1_score(y_test, y_pred, average='macro')
            accuracy = accuracy_score(y_test, y_pred)
            confusion = confusion_matrix(y_test, y_pred)
            print(report)
            print(confusion)

            f = create_result_file(scenario=self.scenario, model_type="Logistic", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                                   slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)
            save_confusion_matrix_ML(y_test, y_pred, classes=self.classes, scenario=self.scenario, model_type="Logistic",
                                  split=self.data_split, num_of_sensors=self.num_of_sensors,
                                  is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)
            fold_idx += 1

    def KNN(self, data, label, split_type):
        model_type = "K Nearest Neighbors"
        print(model_type)
        print(self.scenario)
        print(self.classes)
        print(self.num_of_sensors)

        # print("x_train" + str(x_train.shape))
        #
        # x_train_features = np.load("features\\x_train_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        # x_test_features = np.load("features\\x_test_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        #
        # # split by number of sensors
        # x_train_features = split_feature(x_train_features, self.num_of_sensors)
        # x_test_features = split_feature(x_test_features, self.num_of_sensors)

        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            print(label.shape)
            split_result = logo.split(data, label, groups)
        else:
            return -1

        fold_idx = 0
        for train_index, test_index in split_result:
            x_train_features, x_test_features = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            x_train_features = np.float32(x_train_features)
            x_test_features = np.float32(x_test_features)
            x_train_features = np.nan_to_num(x_train_features)
            x_test_features = np.nan_to_num(x_test_features)

            # find the best parameter by grid search
            params_grid = [{'n_neighbors': list(range(1, 5))}]
            model = GridSearchCV(neighbors.KNeighborsClassifier(), params_grid, cv=5)
            model.fit(x_train_features, y_train)

            print('Best score for training data:', model.best_score_, "\n")
            print('Best K:', model.best_estimator_.n_neighbors, "\n")

            final_model = model.best_estimator_

            # final_model = neighbors.KNeighborsClassifier()

            # results
            score = final_model.score(x_test_features, y_test)  # mean accuracy
            y_pred = final_model.predict(x_test_features)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            accuracy = accuracy_score(y_test, y_pred)
            fscore = f1_score(y_test, y_pred, average='macro')
            confusion = confusion_matrix(y_test, y_pred)
            print(report)
            print(confusion)

            f = create_result_file(scenario=self.scenario, model_type="KNN", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                                   slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)
            save_confusion_matrix_ML(y_test, y_pred, classes=self.classes, scenario=self.scenario, model_type="KNN",
                                  split=self.data_split, num_of_sensors=self.num_of_sensors,
                                  is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)

            fold_idx += 1

        print('Performance time: ', time.perf_counter() - CheckTimeStartTime)

    def RandomForest(self, data, label, split_type):
        model_type = "Random Forest"
        print(model_type)
        print(self.scenario)
        print(self.classes)
        print(self.num_of_sensors)

        # print("x_train" + str(x_train.shape))
        #
        # x_train_features = np.load("features\\x_train_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        # x_test_features = np.load("features\\x_test_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        # # split by number of sensors
        # x_train_features = split_feature(x_train_features, self.num_of_sensors)
        # x_test_features = split_feature(x_test_features, self.num_of_sensors)
        # print("x_train_features " + str(x_train_features.shape))

        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            print(label.shape)
            split_result = logo.split(data, label, groups)
        else:
            return -1


        fold_idx = 0
        for train_index, test_index in split_result:
            # print(label.shape, len(train_index), len(test_index))
            x_train_features, x_test_features = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            x_train_features = np.float32(x_train_features)
            x_test_features = np.float32(x_test_features)
            x_train_features = np.nan_to_num(x_train_features)
            x_test_features = np.nan_to_num(x_test_features)

            model = ensemble.RandomForestClassifier(random_state=random_seed)
            print(model)
            model.fit(x_train_features, y_train)

            # results
            score = model.score(x_test_features, y_test)  # mean accuracy
            y_pred = model.predict(x_test_features)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            accuracy = accuracy_score(y_test, y_pred)
            fscore = f1_score(y_test, y_pred, average='macro')
            confusion = confusion_matrix(y_test, y_pred)
            print(report)
            print(confusion)

            f = create_result_file(scenario=self.scenario, model_type="RF", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                                   slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)
            save_confusion_matrix_ML(y_test, y_pred, classes=self.classes, scenario=self.scenario, model_type="RF",
                                  split=self.data_split, num_of_sensors=self.num_of_sensors,
                                  is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)

            fold_idx += 1


    def NaiveBayes(self, data, label, split_type):
        model_type = "Naive Bayes"
        print(model_type)
        print(self.scenario)
        print(self.classes)
        print(self.num_of_sensors)
        #
        # print("x_train" + str(x_train.shape))
        #
        # # load features
        # x_train_features = np.load("features\\x_train_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        # x_test_features = np.load("features\\x_test_features_" + str(self.data_split) + "_" + str(self.scenario) + "_sensors.npy")
        # # split by number of sensors
        # x_train_features = split_feature(x_train_features, self.num_of_sensors)
        # x_test_features = split_feature(x_test_features, self.num_of_sensors)
        # print("x_train_features " + str(x_train_features.shape))

        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            print(label.shape)
            split_result = logo.split(data, label, groups)
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
            x_train_features = np.float32(x_train_features)
            x_test_features = np.float32(x_test_features)
            x_train_features = np.nan_to_num(x_train_features)
            x_test_features = np.nan_to_num(x_test_features)

            model = naive_bayes.GaussianNB()
            print(model)
            model.fit(x_train_features, y_train)

            # results
            score = model.score(x_test_features, y_test)  # mean accuracy
            y_pred = model.predict(x_test_features)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            accuracy = accuracy_score(y_test, y_pred)
            fscore = f1_score(y_test, y_pred, average='macro')
            confusion = confusion_matrix(y_test, y_pred)
            print(report)
            print(confusion)

            f = create_result_file(scenario=self.scenario, model_type="NB", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                                   slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)
            save_confusion_matrix_ML(y_test, y_pred, classes=self.classes, scenario=self.scenario, model_type="NB",
                                  split=self.data_split, num_of_sensors=self.num_of_sensors,
                                  is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)

            fold_idx += 1