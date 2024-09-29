"""
Author: Oh, Seungwoo
Description: Deep learning models. (MLP, CNN, LSTM, Transformer)
Last modified: 2024.09.09.
"""

import time

import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold, LeaveOneGroupOut

from tensorflow import keras
from tensorflow.keras import Input
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

from utils import *

CheckTimeStartTime = time.perf_counter()

### hyperparameters of deep learning models
batch_size = 128
slicing_time = 5
epochs = 30
drop_out = 0.2

random_seed = 10

class ModelDeepLearning:
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

    def MLP(self, data, label, split_type):
        print("MLP")
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
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

            x_train_set = []
            x_test_set = []
            for x_index in range(self.num_of_sensors):
                x_train_set.append(x_train[:, :, (x_index * 3):(x_index * 3) + 3])
                x_test_set.append(x_test[:, :, (x_index * 3):(x_index * 3) + 3])

            input_shape = (int(slicing_time * 60), 3)

            input_ = []
            for _ in range(self.num_of_sensors):
                input_.append(Input(shape=input_shape))

            x_layer = []
            for i in range(self.num_of_sensors):
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

            f = create_result_file(scenario=self.scenario, device=self.device, model_type="MLP", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold=fold_idx)
            save_result_file_DL(f, slicing_time=slicing_time, score=score,
                                classification_report=report, confusion=confusion)
            save_model(model, scenario=self.scenario, device=self.device, model_type="MLP", split=self.data_split,
                       num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold_num=fold_idx)

            fold_idx += 1

    def CNN(self, data, label, split_type):
        print("CNN")

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
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

            x_train_set = []
            x_test_set = []
            for x_index in range(self.num_of_sensors):
                x_train_set.append(x_train[:, :, (x_index * 3):(x_index * 3) + 3])
                x_test_set.append(x_test[:, :, (x_index * 3):(x_index * 3) + 3])

            input_shape = (int(slicing_time * 60), 3)

            input_ = []
            for _ in range(self.num_of_sensors):
                input_.append(Input(shape=input_shape))

            x_layer = []
            for i in range(self.num_of_sensors):
                x_layer.append(layers.Conv1D(filters=16, kernel_size=3, strides=1, padding='Same', activation="relu",                   
                                             input_shape=input_shape)(input_[i]))
                x_layer[i] = layers.MaxPooling1D(pool_size=2, strides=2)(x_layer[i])
                x_layer[i] = layers.Conv1D(filters=32, kernel_size=3, strides=1, padding='Same', activation='relu')(                    
                    x_layer[i])
                x_layer[i] = layers.MaxPooling1D(pool_size=2, strides=2)(x_layer[i])
                x_layer[i] = layers.Conv1D(filters=64, kernel_size=3, strides=1, padding='Same', activation='relu')(                   
                    x_layer[i])
                x_layer[i] = layers.MaxPooling1D(pool_size=2, strides=2)(x_layer[i])
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

            f = create_result_file(scenario=self.scenario, device=self.device, model_type="CNN", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold=fold_idx)
            save_result_file_DL(f, slicing_time=slicing_time, score=score, classification_report=report,
                                confusion=confusion)
            save_model(model, scenario=self.scenario, device=self.device, model_type="CNN", split=self.data_split,
                       num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold_num=fold_idx)

            fold_idx += 1

    def LSTM(self, data, label, split_type):
        print("LSTM")
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
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

            x_train_set = []
            x_test_set = []

            for x_index in range(self.num_of_sensors):
                x_train_set.append(x_train[:, :, (x_index * 3):(x_index * 3) + 3])
                x_test_set.append(x_test[:, :, (x_index * 3):(x_index * 3) + 3])

            input_shape = (int(slicing_time * 60), 3)

            input_ = []
            for _ in range(self.num_of_sensors):
                input_.append(Input(shape=input_shape))

            x_layer = []
            for i in range(self.num_of_sensors):
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

            f = create_result_file(scenario=self.scenario, device=self.device, model_type="LSTM", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold=fold_idx)
            save_result_file_DL(f, slicing_time=slicing_time, score=score, classification_report=report,
                                confusion=confusion)
            save_model(model, scenario=self.scenario, device=self.device, model_type="LSTM", split=self.data_split,
                       num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold_num=fold_idx)

            fold_idx += 1

    def transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout=0):
        # Normalization and Attention
        x = layers.LayerNormalization(epsilon=1e-6)(inputs)
        x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
        res = x + inputs
        # Feed Forward Part
        x = layers.Conv1D(filters=ff_dim, kernel_size=3, padding='same', activation='relu')(res)          
        x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=3, padding='same')(x)                   

        return x + res

    def Transformer(self, data, label, split_type):
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
            x_train, x_test = data[train_index], data[test_index]
            y_train, y_test = label[train_index], label[test_index]

            y_train = to_categorical(y_train, self.num_classes)
            y_test = to_categorical(y_test, self.num_classes)

            input_shape = x_train.shape[1:]

            inputs = Input(shape=input_shape)  # (300, 9)
            x = inputs

            for _ in range(4):
                x = self.transformer_encoder(x, head_size=16, num_heads=3, ff_dim=3)

            x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
            x = layers.Dense(256, activation="relu")(x)                                                         
            x = layers.Dropout(drop_out)(x)
            outputs = layers.Dense(self.num_classes, activation="softmax")(x)

            model = Model(inputs, outputs)

            model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                metrics=["categorical_accuracy"])

            callbacks = [keras.callbacks.EarlyStopping(patience=50)]                 

            history = model.fit(x_train, y_train, validation_split=0.2, epochs=250, batch_size=batch_size,              
                                callbacks=callbacks)

            score = model.evaluate(x_test, y_test, verbose=1)

            y_pred = np.argmax(model.predict(x_test), axis=1)
            y_test = np.argmax(y_test, axis=1)
            report = classification_report(y_test, y_pred, target_names=self.classes)
            confusion = confusion_matrix(y_test, y_pred)
            confusion = confusion / confusion.astype(float).sum(axis=1)

            f = create_result_file(scenario=self.scenario, device=self.device, model_type="transformer", split=self.data_split,
                                   num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold=fold_idx)
            save_result_file_DL(f, slicing_time=slicing_time, score=score, classification_report=report,
                                confusion=confusion)
            save_model(model, scenario=self.scenario, device=self.device, model_type="transformer", split=self.data_split,
                       num_of_sensors=self.num_of_sensors, slicing_time=slicing_time, fold_num=fold_idx)

            fold_idx += 1