import time
import numpy as np

from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearnex import patch_sklearn
from tensorflow.keras.utils import to_categorical

import transformers
import xgboost
from xgboost import XGBClassifier

from utils import *

slicing_time = 5
CheckTimeStartTime = time.perf_counter()
patch_sklearn()

# n = 176760//(50+48+50+48+50)
# groups = [0 for _ in range(int(50 * n) + 26)] + [1 for _ in range(int(48 * n) + 27)] + [2 for _ in range(int(50 * n) + 26)] + [3 for _ in range(int(48 * n) + 27)] + [4 for _ in range(int(50 * n) + 26)]
n = 172800 // 5
groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n)]
# n = 79200 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n + 1440)] # 1~56
# n = 93600 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n - 1440)] # 57~120
# n = 64800 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n)] # valid 45
# n = 862560 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n)]  # 2 sec 50%

random_seed = 10


class ModelExtra:
    def __init__(self, data_split, num_of_sensors, scenario):
        """
        Args:
            - data split -> "random", "user_indep", "period_indep", "forward"
            - num_of_sensors -> 1:acc, 2:gyro, 3:mag, 4:grav
            - is_gps_included -> True / False
            - scenario -> "disabled", "wheelchairs", "indoor", "outdoor", "all-6", "all-12"
        """
        self.data_split = data_split
        self.num_of_sensors = num_of_sensors
        self.scenario = scenario

        if scenario == "disabled":
            self.num_classes = 4
            self.classes = ['still', 'walking', 'disabled']
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

    def XGBoost(self, data, label, split_type):
        CheckTimeStartTime = time.perf_counter()

        model_type = "XGBoost"
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

        fold_idx = 0

        # train test set
        for train_index, test_index in split_result:
            print('fold: ', fold_idx)
            # if not fold_idx == 0:
            #     fold_idx += 1
            #     continue
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            # y_train = to_categorical(y_train, self.num_classes)
            # y_test = to_categorical(y_test, self.num_classes)

            # x_train_set = []
            # x_test_set = []
            # # x_index 0 - grv, 1 - acc, 2 - gyt, 3 - mag
            # for x_index in range(self.num_of_sensors):
            #     x_train_set.append(x_train[:, :, (x_index * 3):(x_index * 3) + 3])
            #     x_test_set.append(x_test[:, :, (x_index * 3):(x_index * 3) + 3])
            # x_train = np.concatenate(tuple(x_train_set), axis=-1)
            # x_test = np.concatenate(tuple(x_test_set), axis=-1)

            # # xgboost 모델 구현
            # dtrain = xgboost.DMatrix(data=x_train, label=y_train)
            # dtest = xgboost.DMatrix(data=x_test, label=y_test)

            # params = {
            #     "max_depth": 3,
            #     "eta": 0.1,
            #     "objective": "binary:logistic",
            #     "eval_metric": "logloss"
            # }

            # num_rounds = 20
            #
            # wlist = [(x_train, "train"), (x_test, "eval")]

            # xgb_model = xgboost.train(params=params, dtrain=dtrain, num_boost_round=num_rounds, evals=wlist)

            model = xgboost.XGBClassifier()
            model.fit(x_train, y_train, verbose=True)

            # model.get_booster()
            y_pred = model.predict(x_test)
            # y_pred = to_categorical(y_pred, self.num_classes)
            # y_pred = np.argmax(y_pred)

            # results
            # y_pred_probs = xgb_model.predict(dtest)
            # y_pred = [idx for idx, prob in enumerate(y_pred_probs) if prob == max(y_pred_probs)]

            print(time.perf_counter()-CheckTimeStartTime)
            print(y_test.shape, y_pred.shape)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
            fscore = f1_score(y_test, y_pred, average='macro')
            confusion = confusion_matrix(y_test, y_pred)
            # print(report)
            print(confusion)

            f = create_result_file(scenario=self.scenario, model_type="XGBoost", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, is_gps_included=False,
                                   slicing_time=slicing_time, fold=fold_idx)
            save_result_file_ML(f, classification_report=report, accuracy=accuracy, fscore=fscore, confusion=confusion)

            fold_idx += 1
