import os
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
from sklearnex import patch_sklearn
import tensorflow as tf
import keras
# from tensorflow import keras
from keras.utils import to_categorical
from keras import layers
from utils import *

data_path = "E:\\"

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

data_split = 'user-indep'
num_of_sensors = 3
scenario = "wheelchairs"
n_classes = 4
classes = ['still', 'walking', 'walking-aid', 'wheelchair']

random_seed = 10
slicing_time = 5
isGpsIncluded = False

# epochs = 200


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    res = x + inputs
    print(inputs.shape, x.shape, res.shape)
    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def build_model(
        input_shape,
        head_size,
        num_heads,
        ff_dim,
        num_transformer_blocks,
        mlp_unit,
        dropout=0,
        mlp_dropout=0,
):
    # Input
    inputs = keras.Input(shape=input_shape)  # (300, 9)
    x = inputs

    # Encoders
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Output
    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    x = layers.Dense(mlp_unit, activation="relu")(x)
    x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


# # load data - 2s, overlap 50%
# data = np.load(os.path.join(data_path, "2s_publish_data_object.npy"))
# if scenario == "all-6":
#     label = np.load(os.path.join(data_path, "2s_six_activities_label_object.npy"))
# if scenario == "disabled":
#     label = np.load(os.path.join(data_path, "2s_disabled_label_object.npy"))
# if scenario == "wheelchairs":
#     label = np.load(os.path.join(data_path, "2s_wheelchairs_label_object.npy"))

# load data - 5s, no overlap
data = np.load(os.path.join(data_path, "5s_publish_data_object.npy"))
if scenario == "all-6":
    label = np.load(os.path.join(data_path, "5s_six_activities_label_object.npy"))
if scenario == "disabled":
    label = np.load(os.path.join(data_path, "5s_disabled_label_object.npy"))
if scenario == "wheelchairs":
    label = np.load(os.path.join(data_path, "5s_wheelchairs_label_object.npy"))
# number of sensors
if num_of_sensors == 3:  # acc, gyro, mag
    data = data[:, :, :9]
if num_of_sensors == 2:  # acc, gyro
    data = data[:, :, :6]
if num_of_sensors == 1:  # acc
    data = data[:, :, :3]

# # load data - 5s, phone+watch
# data_phone = np.load(os.path.join(data_path, "phone_watch_npy\\5s_valid_phone_data_obj.npy"))
# data_watch = np.load(os.path.join(data_path, "phone_watch_npy\\5s_valid_watch_data_obj.npy"))
# if num_of_sensors == 3:  # acc, gyro, mag
#     data_phone = data_phone[:, :, :9]
#     data_watch = data_watch[:, :, :9]
# if num_of_sensors == 2:  # acc, gyro
#     data_phone = data_phone[:, :, :6]
#     data_watch = data_watch[:, :, :6]
# if num_of_sensors == 1:  # acc
#     data_phone = data_phone[:, :, :3]
#     data_watch = data_watch[:, :, :3]
# data = np.concatenate([data_phone, data_watch], axis=2)
# # number of sensors
# if scenario == "all-6":
#     label = np.load(os.path.join(data_path, "phone_watch_npy\\5s_valid_label_obj.npy"))
# if scenario == "disabled":
#     label = np.load(os.path.join(data_path, "phone_watch_npy\\5s_valid_disabled_label.npy"))
# if scenario == "wheelchairs":
#     label = np.load(os.path.join(data_path, "phone_watch_npy\\5s_valid_wheelchairs_label.npy"))

# # load data - 5s, 1/10 testing data
# data = np.load("5s_data_sliced.npy")
# label = np.load("5s_label_sliced.npy")
# # number of sensors
# if num_of_sensors == 3:  # acc, gyro, mag
#     data = data[:, :, :9]
# if num_of_sensors == 2:  # acc, gyro
#     data = data[:, :, :6]
# if num_of_sensors == 1:  # acc
#     data = data[:, :, :3]

# train test split
if data_split == 'random':
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)
    split_result = kf.split(data)
elif data_split == 'user-indep':
    logo = LeaveOneGroupOut()
    split_result = logo.split(data, label, groups)

# param tuning results
file_name = "transformer_parameter_tuning.txt"
file_path = os.path.join("C:\\Users\\user\\Documents\\2022_2\\HAR\\results", file_name)
f = open(file_path, 'w')

for head_size in [128, 256]:
    for num_heads in [4]:
        for ff_dim in [3]:
            for depth in [1,2,3,4]:
                head_size, num_heads, ff_dim, depth = (16,3,3,4)
                fold_idx = 0
                for train_index, test_index in split_result:
                    if not fold_idx == 4:
                        fold_idx += 1
                        continue

                    x_train, x_test = data[train_index], data[test_index]
                    y_train, y_test = label[train_index], label[test_index]

                    y_train = to_categorical(y_train, n_classes)
                    y_test = to_categorical(y_test, n_classes)
                    # data : (7188, 120, 9)

                    input_shape = x_train.shape[1:]
                    # input_shape = (120, 9)

                    model = build_model(
                        input_shape,
                        head_size=head_size,  # 32, 64, 128, 256
                        num_heads=num_heads,  # 2, 3, 4
                        ff_dim=ff_dim,  # 1, 3
                        num_transformer_blocks=depth,  # 1, 2, 3, 4
                        mlp_unit=128,
                        mlp_dropout=0.2,
                        # dropout=0.2,
                    )

                    model.compile(
                        loss="categorical_crossentropy",
                        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                        metrics=["categorical_accuracy"],
                    )
                    model.summary()

                    callbacks = [keras.callbacks.EarlyStopping(patience=50, restore_best_weights=True)]

                    history = model.fit(
                        x_train,
                        y_train,
                        validation_split=0.2,
                        epochs=300,
                        batch_size=64,
                        callbacks=callbacks,
                    )

                    score = model.evaluate(x_test, y_test, verbose=1)

                    y_pred = np.argmax(model.predict(x_test), axis=1)
                    y_test = np.argmax(y_test, axis=1)
                    report = classification_report(y_test, y_pred, target_names=classes)
                    confusion = confusion_matrix(y_test, y_pred)
                    confusion = confusion / confusion.astype(float).sum(axis=1)

                    # # param tuning save accuracy
                    # acc = score[1]
                    # txt = ''.join(['head_size:', str(head_size), '  num_heads:', str(num_heads), '  ff_dim:', str(ff_dim),
                    #       '  depth:', str(depth), '  fold: ', str(fold_idx), '     accuracy:', str(acc), '\n'])
                    # print(txt)
                    # f.write(txt)

                    print(history)
                    print(score)
                    print(report)
                    print(confusion)

                    f = create_result_file(scenario=scenario, model_type="transformer", split=data_split,
                                           num_of_sensors=num_of_sensors, is_gps_included=isGpsIncluded,
                                           slicing_time=slicing_time, fold=fold_idx)
                    save_result_file(f, slicing_time=slicing_time, history=history, score=score, classification_report=report,
                                     confusion=confusion)
                    # save_confusion_matrix(confusion, classes=classes, scenario=scenario, model_type="transformer",
                    #                       split=data_split, num_of_sensors=num_of_sensors,
                    #                       is_gps_included=isGpsIncluded, slicing_time=slicing_time)
                    # save_model(model, scenario=scenario, model_type="transformer", split=data_split,
                    #            num_of_sensors=num_of_sensors, is_gps_included=isGpsIncluded,
                    #            slicing_time=slicing_time, fold_num=fold_idx)
                    fold_idx += 1
f.close()

