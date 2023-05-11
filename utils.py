import os
import random

import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import ConfusionMatrixDisplay, plot_confusion_matrix
import matplotlib.pyplot as plt

path = "C:\\Users\\user\\Documents\\2022_2\\HAR"


# utils for saving files

def create_result_file(scenario, model_type, split, num_of_sensors, is_gps_included, slicing_time, fold):
    """
    :return: file pointer
    """
    if is_gps_included:
        gps = "GPS"
    else:
        gps = "X"

    file_name = str(num_of_sensors) + "_sensors_" + gps + "_" + scenario + "_" + str(
        slicing_time) + "sec_" + model_type + "_" + split + "_" + "result-" + str(fold) + ".txt"
    file_path = os.path.join(os.path.join(path, "results"), file_name)
    f = open(file_path, 'w')
    return f

def save_result_file_ML(f, classification_report, accuracy, fscore, confusion):
    f.write('Classification report : \n' + str(classification_report))
    f.write('Accuracy: ' + str(accuracy))
    f.write('Average F1-score: ' + str(fscore))
    f.write('\n\nConfusion matrix\n' + str(confusion))

    print('Classification report: ' + str(classification_report))
    f.close()

    return

def save_result_file(f, slicing_time, history, score, classification_report, confusion):
    """
    :param f: file pointer
    :param slicing_time: window size
    :param history: keras.models.Model.fit
    :param score: keras.models.Model.evaluate
    :param classification_report: sklearn.metrics.classification_report
    :param confusion: sklearn.metrics.confusion_matrix
    """

    f.write(str(slicing_time) + "sec_Result" + "\n")
    # f.write('Train loss: ' + str(history.history['loss']) + "\n")
    # f.write('Train accuracy: ' + str(history.history['accuracy']) + "\n")
    f.write('Test loss: ' + str(score[0]) + "\n")
    f.write('Test accuracy: ' + str(score[1]) + "\n")
    f.write('Classification report : \n' + str(classification_report))
    f.write('Confusion matrix\n' + str(confusion))

    # print('Train loss: ' + str(history.history['loss']))
    # print('Train acc: ' + str(history.history['accuracy']))
    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    print('Classification report: ' + str(classification_report))

    f.close()

def save_confusion_matrix_ML(y_true, y_pred, classes, scenario, model_type, split, num_of_sensors,
                          is_gps_included, slicing_time):
    if is_gps_included:
        cm_color = plt.cm.Blues
        gps = "GPS"
    else:
        cm_color = plt.cm.Reds
        gps = "X"

    ConfusionMatrixDisplay.from_predictions(y_true=y_true, y_pred=y_pred, cmap=cm_color)
    file_name = str(num_of_sensors) + "_sensors_" + gps + "_" + scenario + "_" + str(
        slicing_time) + "sec_" + model_type + "_" + split + ".png"
    file_path = os.path.join(os.path.join(path, "results"), file_name)
    plt.savefig(file_path)


def save_confusion_matrix(confusion_matrix, classes, scenario, model_type, split, num_of_sensors,
                          is_gps_included, slicing_time):
    if is_gps_included:
        cm_color = plt.cm.Blues
        gps = "GPS"
    else:
        cm_color = plt.cm.Reds
        gps = "X"

    if len(classes) == 3:
        fontsize = 18
    elif len(classes) == 12:
        fontsize = 8
    else:
        fontsize = 13

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap=cm_color)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(confusion_matrix.shape[1]),
           yticks=np.arange(confusion_matrix.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(j, i, format(confusion_matrix[i, j], '.2f'),
                    ha="center", va="center", fontsize=fontsize,
                    color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2. else "black")
    file_name = str(num_of_sensors) + "_sensors_" + gps + "_" + scenario + "_" + str(
        slicing_time) + "sec_" + model_type + "_" + split + ".png"
    file_path = os.path.join(os.path.join(path, "results"), file_name)
    plt.savefig(file_path)


def save_model(model, scenario, model_type, split, num_of_sensors, is_gps_included, slicing_time, fold_num):
    if is_gps_included:
        gps = "GPS"
    else:
        gps = "X"

    file_name = str(num_of_sensors) + "_sensors_" + gps + "_" + scenario + "_" + str(
        slicing_time) + "sec_" + model_type + "_" + split + "_" + "model-" + str(fold_num) + ".hdf5"
    model_path = os.path.join(os.path.join(path, "models"), file_name)
    model.save(model_path)


# utils for loading data

# def load_data(scenario, slicing_time=5):
#     """
#     "disabled", "wheelchairs", "indoor", "outdoor", "all-6", "all-12"
#
#     :param scenario: loads the right npy file for the given scenario
#     :param slicing_time: set as 5
#     :return: (data, label)
#     """
#     data_path = "C:\\Users\\user\\Documents\\2022_2\\HAR\\data\\4_data_concat"
#
#     data = np.load(os.path.join(data_path, "300_data_object.npy"))
#     # label = np.load(os.path.join(data_path, "publish_label_object.npy"))
#     if scenario == "all-6":
#         # label = np.load(os.path.join(data_path, str(slicing_time * 60) + "_label_object.npy"))
#         label = np.load(os.path.join(data_path, "300_label_object.npy"))
#     if scenario == "all-12":
#         # indoor_x = np.load(os.path.join(data_path, "indoor_x.npy"))
#         # indoor_y = np.load(os.path.join(data_path, "indoor_y.npy"))
#         # outdoor_x = np.load(os.path.join(data_path, "outdoor_x.npy"))
#         # outdoor_y = np.load(os.path.join(data_path, "outdoor_y.npy")) + 6
#         # data = np.concatenate([indoor_x, outdoor_x], axis=0)
#         # label = np.load(os.path.join(data_path, "5s_publish_label_object.npy"))
#         # label = np.concatenate([indoor_y, outdoor_y], axis=0)
#         pass
#     if scenario == "indoor":
#         # data = np.load(os.path.join(data_path, "indoor_x.npy"))
#         # label = np.load(os.path.join(data_path, "indoor_y.npy"))
#         pass
#     if scenario == "outdoor":
#         # data = np.load(os.path.join(data_path, "outdoor_x.npy"))
#         # label = np.load(os.path.join(data_path, "outdoor_y.npy"))
#         pass
#     if scenario == "disabled":
#         # data = np.load(os.path.join(data_path, "disabled_object.npy"))
#         # label = np.load(os.path.join(data_path, "disabled_label.npy"))
#         label = np.load(os.path.join(data_path, "disabled_label_object.npy"))
#     if scenario == "wheelchairs":
#         # data = np.load(os.path.join(data_path, "wheelchairs_object.npy"))
#         # label = np.load(os.path.join(data_path, "wheelchairs_label.npy"))
#         label = np.load(os.path.join(data_path, "wheelchairs_label_object.npy"))
#
#     return data, label

def load_data(scenario, slicing_time=5):
    """
    "disabled", "wheelchairs", "indoor", "outdoor", "all-6", "all-12"

    :param scenario: loads the right npy file for the given scenario
    :param slicing_time: set as 5
    :return: (data, label)
    """
    data_path = "E:\\"
    if scenario == 'test':
        # data = np.load(os.path.join(data_path, "data_object_0127.npy"))
        # label = np.load(os.path.join(data_path, "label_object_0127.npy"))
        data = np.load("data_sliced.npy")
        label = np.load("label_sliced.npy")

    # else:
    #     data = np.load(os.path.join(data_path, "2s_publish_data_object.npy"))
    #     if scenario == "all-6":
    #         label = np.load(os.path.join(data_path, "2s_six_activities_label_object.npy"))
    #     if scenario == "all-12":
    #         label = np.load(os.path.join(data_path, "2s_publish_label_object.npy"))
    #     if scenario == "disabled":
    #         label = np.load(os.path.join(data_path, "2s_disabled_label_object.npy"))
    #     if scenario == "wheelchairs":
    #         label = np.load(os.path.join(data_path, "2s_wheelchairs_label_object.npy"))
    else:
        data = np.load(os.path.join(data_path, "5s_publish_data_object.npy"))
        if scenario == "all-6":
            label = np.load(os.path.join(data_path, "5s_six_activities_label_object.npy"))
        if scenario == "all-12":
            label = np.load(os.path.join(data_path, "5s_publish_label_object.npy"))
        if scenario == "disabled":
            label = np.load(os.path.join(data_path, "5s_disabled_label_object.npy"))
        if scenario == "wheelchairs":
            label = np.load(os.path.join(data_path, "5s_wheelchairs_label_object.npy"))

    return data, label

def split_feature(feature_set, num_of_sensors):
    if num_of_sensors == 4:  # acc, gyro, mag, gra
        feature_set = feature_set[:, :12*9]
    if num_of_sensors == 3:  # acc, gyro, mag
        feature_set = feature_set[:, 3*9:12*9]
    if num_of_sensors == 2:  # acc, gyro
        feature_set = feature_set[:, 3*9:9*9]
    if num_of_sensors == 1:  # acc
        feature_set = feature_set[:, 3*9:6*9]

    return feature_set


def split_data(split_type, data, label, test_size=0.2, random_seed=10):
    """
    "random", "user-indep", "forward", "shufflesplit", "LOGO", "timeseries"

    :param split_type: splits the file for the given split type
    :return: (x_train, x_test, y_train, y_test)
    """
    if split_type == "random":
        # x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=test_size, random_state=random_seed)
        KFold(n_splits=5, shuffle=True, random_state=random_seed)

    if split_type == "user-indep":
        random.seed(random_seed)
        participant_list = list(range(117))
        participant_num = len(participant_list)
        random_list = random.sample(range(participant_num), int(participant_num * test_size))

        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        for i in participant_list:
            if i in random_list:
                x_test_list.append(data[720 * i: 720 * (i + 1), :, :])
                y_test_list.append(label[720 * i: 720 * (i + 1)])
            else:
                x_train_list.append(data[720 * i: 720 * (i + 1), :, :])
                y_train_list.append(label[720 * i: 720 * (i + 1)])

        x_train = np.concatenate(tuple(x_train_list), axis=0)
        x_test = np.concatenate(tuple(x_test_list), axis=0)
        y_train = np.concatenate(tuple(y_train_list), axis=0)
        y_test = np.concatenate(tuple(y_test_list), axis=0)

    if split_type == "forward":
        random.seed(random_seed)

        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        # participant_list = list(range(len(data) // 120))
        for i in range(117):
            data_user = data[120 * i: 120 * (i + 1), :, :]
            label_user = label[120 * i: 120 * (i + 1)]

            x_train, x_test, y_train, y_test = train_test_split(data_user, label_user, test_size=0.3,
                                                                random_state=random_seed)

            x_train_list.append(x_train)
            x_test_list.append(x_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)

        x_train = np.concatenate(tuple(x_train_list), axis=0)
        x_test = np.concatenate(tuple(x_test_list), axis=0)
        y_train = np.concatenate(tuple(y_train_list), axis=0)
        y_test = np.concatenate(tuple(y_test_list), axis=0)

    if split_type == "forward-2":
        random.seed(random_seed)

        x_train_list = []
        x_test_list = []
        y_train_list = []
        y_test_list = []

        participant_list = list(range(len(data) // 120))
        for i in participant_list:
            data_user = data[120 * i: 120 * (i + 1), :, :]
            label_user = label[120 * i: 120 * (i + 1)]

            x_train, x_test = np.split(data_user, [int(len(data_user)*0.7)])
            y_train, y_test = np.split(label_user, [int(len(data_user) * 0.7)])


            x_train_list.append(x_train)
            x_test_list.append(x_test)
            y_train_list.append(y_train)
            y_test_list.append(y_test)

        x_train = np.concatenate(tuple(x_train_list), axis=0)
        x_test = np.concatenate(tuple(x_test_list), axis=0)
        y_train = np.concatenate(tuple(y_train_list), axis=0)
        y_test = np.concatenate(tuple(y_test_list), axis=0)

    return x_train, x_test, y_train, y_test
