import time

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, LeaveOneGroupOut

from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from utils import *

# path = "C:\\Users\\user\\Documents\\2022_1\\CRC_DeepLearning\\4_data_spliced\\indoor_and_outdoor"

CheckTimeStartTime = time.perf_counter()

### hyperparameters of deep learning models
batch_size = 128
slicing_time = 2
epochs = 20
drop_out = 0

# n = 176760//(50+48+50+48+50)
# groups = [0 for _ in range(int(50 * n) + 26)] + [1 for _ in range(int(48 * n) + 27)] + [2 for _ in range(int(50 * n) + 26)] + [3 for _ in range(int(48 * n) + 27)] + [4 for _ in range(int(50 * n) + 26)]
# n = 172800 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n)] # 1~120
# n = 79200 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n + 1440)] # 1~56
# n = 93600 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n - 1440)] # 57~120
# n = 64800 // 5
# groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n)] # valid 45
n = 862560 // 5
groups = [0 for _ in range(n)] + [1 for _ in range(n)] + [2 for _ in range(n)] + [3 for _ in range(n)] + [4 for _ in range(n)] # 2 sec 50%


class ModelDeepLearning:
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
            # self.classes = ['in-still', 'in-walking', 'in-manual', 'in-power', 'in-walker', 'in-crutches',
            #                 'out-still', 'out-walking', 'out-manual', 'out-power', 'out-walker', 'out-crutches']
            self.classes = ['in_still', 'out_still', 'in_walking', 'out_walking', 'in_crutches', 'out_crutches',
                            'in_walker', 'out_walker', 'in_manual', 'out_manual', 'in_power', 'out_power']

        else:
            self.num_classes = 6
            self.classes = ['still', 'walking', 'walker', 'crutches', 'manual', 'power']

    def get_model_info(self):
        """
        :return: scenario, number of sensors, is GPS included
        """
        return self.scenario, self.num_of_sensors, self.isGpsIncluded

    # def run_uni_DNN(self, x_train, x_test, y_train, y_test):
    #     print("multi DNN")
    #
    #     y_train = to_categorical(y_train, self.num_classes)
    #     y_test = to_categorical(y_test, self.num_classes)
    #
    #     input_shape = (int(slicing_time * 60), 3 * 4)
    #
    #     input_ = Input(shape=input_shape)
    #
    #     x_layer = layers.Dense(32, activation="relu")(input_)
    #     x_layer = layers.Dense(64, activation="relu")(x_layer)
    #     x_layer = layers.Dense(128, activation="relu")(x_layer)
    #     x_layer = layers.Flatten()(x_layer)
    #
    #     out = layers.Dense(256, activation="relu")(x_layer)
    #     out = layers.Dropout(0.2)(out)
    #     output = layers.Dense(self.num_classes, activation='softmax')(out)
    #
    #     model = Model(inputs=input_, outputs=output)
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    #     history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    #     score = model.evaluate(x_test, y_test, verbose=1)
    #     y_pred = np.argmax(model.predict(x_test), axis=1)
    #     y_test = np.argmax(y_test, axis=1)
    #     report = classification_report(y_test, y_pred, target_names=self.classes)
    #     confusion = confusion_matrix(y_test, y_pred)
    #     # confusion = confusion / confusion.astype(float).sum(axis=1)
    #
    #     f = create_result_file(scenario=self.scenario, model_type="uniDNN", split=self.data_split,
    #                      num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
    #                      slicing_time=slicing_time)
    #     save_result_file(f, slicing_time=slicing_time, history=history, score=score, classification_report=report,
    #                      confusion=confusion)
    #     save_confusion_matrix(confusion, classes=self.classes, scenario=self.scenario, model_type="multiDNN",
    #                                 split=self.data_split, num_of_sensors=self.num_of_sensors,
    #                                 is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)
    #     save_model(model, scenario=self.scenario, model_type="uniDNN", split=self.data_split,
    #                      num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
    #                      slicing_time=slicing_time)

    # def run_uni_CNN(self, x_train, x_test, y_train, y_test):
    #     print("multi CNN")
    #
    #     y_train = to_categorical(y_train, self.num_classes)
    #     y_test = to_categorical(y_test, self.num_classes)
    #
    #     input_shape = (int(slicing_time * 60), 3 * 4)
    #
    #     input_ = Input(shape=input_shape)
    #
    #     x_layer = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='Same', activation="relu",
    #                             input_shape=input_shape)(input_)
    #     x_layer = layers.MaxPooling1D(pool_size=2, strides=2)(x_layer)
    #     x_layer = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='Same', activation='relu')(x_layer)
    #     x_layer = layers.MaxPooling1D(pool_size=2, strides=2)(x_layer)
    #     x_layer = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='Same', activation='relu')(x_layer)
    #     x_layer = layers.MaxPooling1D(pool_size=2, strides=2)(x_layer)
    #     x_layer = layers.Flatten()(x_layer)
    #
    #     out = layers.Dense(256, activation="relu")(x_layer)
    #     out = layers.Dropout(0.2)(out)
    #     output = layers.Dense(self.num_classes, activation='softmax')(out)
    #
    #     model = Model(inputs=input_, outputs=output)
    #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #
    #     history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    #     score = model.evaluate(x_test, y_test, verbose=1)
    #     y_pred = np.argmax(model.predict(x_test), axis=1)
    #     y_test = np.argmax(y_test, axis=1)
    #     report = classification_report(y_test, y_pred, target_names=self.classes)
    #     confusion = confusion_matrix(y_test, y_pred)
    #     # confusion = confusion / confusion.astype(float).sum(axis=1)
    #
    #     f = create_result_file(scenario=self.scenario, model_type="multiDNN", split=self.data_split,
    #                      num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
    #                      slicing_time=slicing_time)
    #     save_result_file(f, slicing_time=slicing_time, history=history, score=score, classification_report=report,
    #                      confusion=confusion)
    #     save_confusion_matrix(confusion, classes=self.classes, scenario=self.scenario, model_type="multiDNN",
    #                                 split=self.data_split, num_of_sensors=self.num_of_sensors,
    #                                 is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)
    #     save_model(model, scenario=self.scenario, model_type="multiDNN", split=self.data_split,
    #                      num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
    #                      slicing_time=slicing_time)

    def run_multi_LSTM(self, data, label, split_type):
        print("multi LSTM")
        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=10)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            split_result = logo.split(data, label, groups)
        else:
            return -1


        print(label)
        fold_idx = 0
        for train_index, test_index in split_result:
            if fold_idx == 0 or fold_idx == 1 or fold_idx == 2:
                fold_idx += 1
                continue
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

            x_train_set = []
            x_test_set = []

            # x_index 0 - grv, 1 - acc, 2 - gyt, 3 - mag
            for x_index in range(self.num_of_sensors):
                x_train_set.append(x_train[:, :, (x_index * 3):(x_index * 3) + 3])
                x_test_set.append(x_test[:, :, (x_index * 3):(x_index * 3) + 3])
            if self.isGpsIncluded:
                x_train_set.append(x_train[:, :, -2:])
                x_test_set.append(x_train[:, :, -2:])

            input_shape = (int(slicing_time * 60), 3)
            gps_input_shape = (int(slicing_time * 60), 2)

            input_ = []
            for _ in range(self.num_of_sensors):
                input_.append(Input(shape=input_shape))
            if self.isGpsIncluded:
                input_.append(Input(shape=gps_input_shape))

            x_layer = []
            for i in range(self.num_of_sensors + int(self.isGpsIncluded)):
                x_layer.append(layers.LSTM(128, return_sequences=True)(input_[i]))
                x_layer[i] = layers.LSTM(128)(x_layer[i])

            out = layers.concatenate(x_layer, axis=-1)
            out = layers.Dense(256, activation="relu")(out)
            out = layers.Dropout(0.2)(out)
            output = layers.Dense(self.num_classes, activation='softmax')(out)

            model = Model(inputs=input_, outputs=output)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(x_train_set, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            score = model.evaluate(x_test_set, y_test, verbose=1)
            y_pred = np.argmax(model.predict(x_test_set), axis=1)
            y_test = np.argmax(y_test, axis=1)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            confusion = confusion_matrix(y_test, y_pred)
            # confusion = confusion / confusion.astype(float).sum(axis=1)

            f = create_result_file(scenario=self.scenario, model_type="multiLSTM", split=self.data_split,
                             num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                             slicing_time=slicing_time, fold=fold_idx)
            save_result_file(f, slicing_time=slicing_time, history=history, score=score, classification_report=report,
                             confusion=confusion)
            save_confusion_matrix(confusion, classes=self.classes, scenario=self.scenario, model_type="multiLSTM",
                                        split=self.data_split, num_of_sensors=self.num_of_sensors,
                                        is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)
            save_model(model, scenario=self.scenario, model_type="multiLSTM", split=self.data_split,
                             num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                             slicing_time=slicing_time, fold_num=fold_idx)


            fold_idx += 1

    def run_multi_DNN(self, data, label, split_type):
        print("multi DNN")
        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=10)
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            split_result = logo.split(data, label, groups)
        else:
            return -1

        print(label)
        fold_idx = 0
        for train_index, test_index in split_result:
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

            x_train_set = []
            x_test_set = []

            # x_index 0 - grv, 1 - acc, 2 - gyt, 3 - mag
            for x_index in range(self.num_of_sensors):
                x_train_set.append(x_train[:, :, (x_index * 3):(x_index * 3) + 3])
                x_test_set.append(x_test[:, :, (x_index * 3):(x_index * 3) + 3])
            if self.isGpsIncluded:
                x_train_set.append(x_train[:, :, -2:])
                x_test_set.append(x_test[:, :, -2:])

            input_shape = (int(slicing_time * 60), 3)
            gps_input_shape = (int(slicing_time * 60), 2)

            input_ = []
            for _ in range(self.num_of_sensors):
                input_.append(Input(shape=input_shape))
            if self.isGpsIncluded:
                input_.append(Input(shape=gps_input_shape))

            x_layer = []
            for i in range(self.num_of_sensors + int(self.isGpsIncluded)):
                x_layer.append(layers.Dense(32, activation="relu")(input_[i]))
                x_layer[i] = layers.Dense(64, activation="relu")(x_layer[i])
                x_layer[i] = layers.Dense(128, activation="relu")(x_layer[i])
                x_layer[i] = layers.Flatten()(x_layer[i])

            out = layers.concatenate(x_layer, axis=-1)
            out = layers.Dense(256, activation="relu")(out)
            out = layers.Dropout(drop_out)(out)
            output = layers.Dense(self.num_classes, activation='softmax')(out)

            model = Model(inputs=input_, outputs=output)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            history = model.fit(x_train_set, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            score = model.evaluate(x_test_set, y_test, verbose=1)
            y_pred = np.argmax(model.predict(x_test_set), axis=1)
            y_test = np.argmax(y_test, axis=1)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            confusion = confusion_matrix(y_test, y_pred)
            # confusion = confusion / confusion.astype(float).sum(axis=1)

            f = create_result_file(scenario=self.scenario, model_type="multiDNN", split=self.data_split,
                             num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                             slicing_time=slicing_time, fold=fold_idx)
            save_result_file(f, slicing_time=slicing_time, history=history, score=score, classification_report=report,
                             confusion=confusion)
            save_confusion_matrix(confusion, classes=self.classes, scenario=self.scenario, model_type="multiDNN",
                                        split=self.data_split, num_of_sensors=self.num_of_sensors,
                                        is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)
            save_model(model, scenario=self.scenario, model_type="multiDNN", split=self.data_split,
                             num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                             slicing_time=slicing_time, fold_num=fold_idx)

            fold_idx += 1

    def run_multi_CNN(self, data, label, split_type):
        print("multi CNN")

        if split_type == 'random':
            kf = KFold(n_splits=5, shuffle=True, random_state=10)
            # print(kf.split(data))
            split_result = kf.split(data)
        elif split_type == 'user-indep':
            logo = LeaveOneGroupOut()
            split_result = logo.split(data, label, groups)
        else:
            return -1

        fold_idx = 0
        for train_index, test_index in split_result:
            if not fold_idx == 4:
                fold_idx += 1
                continue
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

            x_train_set = []
            x_test_set = []
            # x_index 0 - grv, 1 - acc, 2 - gyt, 3 - mag
            for x_index in range(self.num_of_sensors):
                x_train_set.append(x_train[:, :, (x_index * 3):(x_index * 3) + 3])
                x_test_set.append(x_test[:, :, (x_index * 3):(x_index * 3) + 3])
            if self.isGpsIncluded:
                x_train_set.append(x_train[:, :, -2:])
                x_test_set.append(x_train[:, :, -2:])

            input_shape = (int(slicing_time * 60), 3)
            gps_input_shape = (int(slicing_time * 60), 2)

            input_ = []
            for _ in range(self.num_of_sensors):
                input_.append(Input(shape=input_shape))
            if self.isGpsIncluded:
                input_.append(Input(shape=gps_input_shape))

            x_layer = []
            for i in range(self.num_of_sensors + int(self.isGpsIncluded)):
                x_layer.append(layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='Same', activation="relu",
                                             input_shape=input_shape)(input_[i]))
                x_layer[i] = layers.MaxPooling1D(pool_size=2, strides=2)(x_layer[i])
                x_layer[i] = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='Same', activation='relu')(
                    x_layer[i])
                x_layer[i] = layers.MaxPooling1D(pool_size=2, strides=2)(x_layer[i])
                x_layer[i] = layers.Conv1D(filters=128, kernel_size=3, strides=1, padding='Same', activation='relu')(
                    x_layer[i])
                x_layer[i] = layers.MaxPooling1D(pool_size=2, strides=2)(x_layer[i])
                x_layer[i] = layers.Flatten()(x_layer[i])

            out = layers.concatenate(x_layer, axis=-1)
            out = layers.Dense(256, activation="relu")(out)
            out = layers.Dropout(drop_out)(out)
            output = layers.Dense(self.num_classes, activation='softmax')(out)

            model = Model(inputs=input_, outputs=output)
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            # print(model.summary())

            history = model.fit(x_train_set, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
            score = model.evaluate(x_test_set, y_test, verbose=1)
            y_pred = np.argmax(model.predict(x_test_set), axis=1)
            y_test = np.argmax(y_test, axis=1)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            confusion = confusion_matrix(y_test, y_pred)
            # confusion = confusion / confusion.astype(float).sum(axis=1)

            f = create_result_file(scenario=self.scenario, model_type="multiCNN", split=self.data_split,
                             num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                             slicing_time=slicing_time, fold=fold_idx)
            save_result_file(f, slicing_time=slicing_time, history=history, score=score, classification_report=report,
                             confusion=confusion)
            save_confusion_matrix(confusion, classes=self.classes, scenario=self.scenario, model_type="multiCNN",
                                        split=self.data_split, num_of_sensors=self.num_of_sensors,
                                        is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)
            save_model(model, scenario=self.scenario, model_type="multiCNN", split=self.data_split,
                             num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
                             slicing_time=slicing_time, fold_num=fold_idx)


            fold_idx += 1


    # def run_uni_LSTM(self, data, label, split_type):
    #     print("uni LSTM")
    #
    #     if split_type == 'random':
    #         kf = KFold(n_splits=5, shuffle=True, random_state=10)
    #         split_result = kf.split(data)
    #     elif split_type == 'user-indep':
    #         logo = LeaveOneGroupOut()
    #         groups = [0 for _ in range(50)] + [1 for _ in range(48)] + [2 for _ in range(50)] + [3 for _ in range(48)] + [4 for _ in range(50)]
    #         split_result = logo.split(data, label, groups)
    #     else:
    #         return -1
    #
    #     fold_idx = 0
    #     for train_index, test_index in split_result:
    #         x_train, x_test = data[train_index], data[test_index]
    #         y_train, y_test = label[train_index], label[test_index]
    #
    #         input_shape = (int(slicing_time * 60), data.shape[2])
    #
    #         input_ = Input(shape=input_shape)
    #
    #         x_layer = layers.LSTM(128, return_sequences=True)(input_)
    #         x_layer = layers.LSTM(128)(x_layer)
    #
    #         out = layers.Dense(256, activation="relu")(x_layer)
    #         out = layers.Dropout(0.2)(out)
    #         output = layers.Dense(self.num_classes, activation='softmax')(out)
    #
    #         model = Model(inputs=input_, outputs=output)
    #         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #         print(model.summary())
    #         history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    #         score = model.evaluate(x_test, y_test, verbose=1)
    #         y_pred = np.argmax(model.predict(x_test), axis=1)
    #         y_test = np.argmax(y_test, axis=1)
    #         report = classification_report(y_test, y_pred, target_names=self.classes)
    #         confusion = confusion_matrix(y_test, y_pred)
    #         # confusion = confusion / confusion.astype(float).sum(axis=1)
    #
    #         f = create_result_file(scenario=self.scenario, model_type="multiLSTM", split=self.data_split,
    #                          num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
    #                          slicing_time=slicing_time, fold=fold_idx)
    #         save_result_file(f, slicing_time=slicing_time, history=history, score=score, classification_report=report,
    #                          confusion=confusion)
    #         save_confusion_matrix(confusion, classes=self.classes, scenario=self.scenario, model_type="multiLSTM",
    #                                     split=self.data_split, num_of_sensors=self.num_of_sensors,
    #                                     is_gps_included=self.isGpsIncluded, slicing_time=slicing_time)
    #         save_model(model, scenario=self.scenario, model_type="multiLSTM", split=self.data_split,
    #                          num_of_sensors=self.num_of_sensors, is_gps_included=self.isGpsIncluded,
    #                          slicing_time=slicing_time)
    #
    #
    #         fold_idx += 1