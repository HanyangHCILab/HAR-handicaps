"""
Author: Oh, Seungwoo
Description: Utilities for saving files.
Last modified: 2024.09.09.
"""

import os

path = "D:\\"


def create_result_file(scenario, device, model_type, split, num_of_sensors, slicing_time, fold):
    """
    :return: file pointer
    """
    file_name = str(num_of_sensors) + "_sensors_" + device + "_" + scenario + "_" + str(
        slicing_time) + "sec_" + model_type + "_" + split + "_" + "result-" + str(fold) + ".txt"
    file_path = os.path.join(os.path.join(path, "results"), file_name)
    if not os.path.exists(os.path.join(path, "results")):
            os.mkdir(os.path.join(path, "results"))
    f = open(file_path, 'w')

    return f


def save_result_file_ML(f, classification_report, accuracy, fscore, confusion):
    f.write('Classification report : \n' + str(classification_report) + '\n')
    f.write('Accuracy: ' + str(accuracy) + '\n')
    f.write('Average F1-score: ' + str(fscore) + '\n')
    f.write('\n\nConfusion matrix\n' + str(confusion))

    print('Classification report: ' + str(classification_report))

    f.close()

    return


def save_result_file_DL(f, slicing_time, score, classification_report, confusion):
    """
    :param f: file pointer
    :param slicing_time: window size
    :param score: keras.models.Model.evaluate
    :param classification_report: sklearn.metrics.classification_report
    :param confusion: sklearn.metrics.confusion_matrix
    """

    f.write(str(slicing_time) + "sec_Result" + "\n")
    f.write('Test loss: ' + str(score[0]) + "\n")
    f.write('Test accuracy: ' + str(score[1]) + "\n")
    f.write('Classification report : \n' + str(classification_report))
    f.write('Confusion matrix\n' + str(confusion))

    print('Test loss: ', score[0])
    print('Test accuracy: ', score[1])
    print('Classification report: ' + str(classification_report))

    f.close()

    return


def save_model(model, scenario, device, model_type, split, num_of_sensors, slicing_time, fold_num):
    file_name = str(num_of_sensors) + "_sensors_" + device + "_" + scenario + "_" + str(slicing_time) + "sec_" + \
                model_type + "_" + split + "_" + "model-" + str(fold_num) + ".hdf5"
    model_path = os.path.join(os.path.join(path, "models"), file_name)
    if not os.path.exists(os.path.join(path, "models")):
            os.mkdir(os.path.join(path, "models"))
    model.save(model_path)
