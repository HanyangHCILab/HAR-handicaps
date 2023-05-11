import os
import numpy as np
from numpy.lib.stride_tricks import as_strided
from utils_preprocessing import *

# (3) Reshape: 0.016 -> 300 window size (5sec)

path_resampled = "D:\\2023\\CRC\\CRC_data\\HAR\\data_publish_resampled"
# path_resampled = "C:\\Users\\user\\Documents\\2022_2\\HAR\\data\\data_publish_resampled"
path_concat = "D:\\2023\\CRC\\CRC_data\\HAR\\data_publish_concat"

window_size = 120
overlap = 60
exclude_list = ['5010', '5019', '5032', '5051', '5051-2', '5089', '5089-2', '5112', '5121', '5121-2']
mode_dict = {'still': 0, 'walking': 1, 'crutches': 2, 'walker': 3, 'manual': 4, 'electric': 5}

obj_list = []
label_list = []

path_resampled_list = os.listdir(path_resampled)
for folder_idx in path_resampled_list:
    if folder_idx in exclude_list:
        continue

    if not os.path.exists(os.path.join(path_concat, folder_idx)):
        os.mkdir(os.path.join(path_concat, folder_idx))

    # for 5 sec, no overlap window
    basic_object_npy = np.load(os.path.join(os.path.join(path_resampled, folder_idx), str(folder_idx) + "_0.016sec_data_object.npy"))
    basic_label_npy = np.load(os.path.join(os.path.join(path_resampled, folder_idx), str(folder_idx) + "_0.016sec_label_object.npy"))

    data_size, _current_window_size, col_size = basic_object_npy.shape

    # no overlap
    final_obj = np.reshape(basic_object_npy, (data_size // window_size, window_size, col_size))
    final_label = np.reshape(basic_label_npy, (data_size // window_size, window_size))

    if final_label.shape[1] == 1:
        final_label = final_label.reshape(data_size // window_size)

    np.save(os.path.join(os.path.join(path_resampled, folder_idx), str(folder_idx) + "_" + str(window_size) + "_data_object.npy"), final_obj)
    np.save(os.path.join(os.path.join(path_resampled, folder_idx), str(folder_idx) + "_" + str(window_size) + "_label_object.npy"), final_label)

    # # for 2 sec, 50% overlap window
    # participant_obj_list = []
    # participant_label_list = []
    # for location in ['indoor', 'outdoor']:
    #     for tm in ['still', 'walking', 'crutches', 'walker', 'manual', 'electric']:
    #         basic_object_npy = np.load(os.path.join(os.path.join(path_resampled, folder_idx),
    #                                                 str(folder_idx) + "_" + tm + "_" + location +
    #                                                 "_0.016sec_data_object.npy"))
    #         basic_label_npy = np.load(os.path.join(os.path.join(path_resampled, folder_idx),
    #                                                str(folder_idx) + "_" + tm + "_" + location +
    #                                                "_0.016sec_label_object.npy"))
    #
    #         data_size, _current_window_size, col_size = basic_object_npy.shape # (36000, 1, 12)
    #         print(basic_object_npy.shape)
    #         print(basic_object_npy)
    #
    #         # yes overlap
    #         size = (1, window_size, col_size) # (1, 120, 12)
    #         window_obj = []
    #         for i in range(data_size//overlap-1):
    #             window_slice = basic_object_npy[i*overlap: i*overlap+window_size, :, :].reshape(size)
    #             window_obj.append(window_slice)
    #
    #         _obj = np.concatenate(tuple(window_obj)) # (599, 120, 12)
    #         _label = np.full(data_size//overlap-1, mode_dict[tm])
    #
    #         participant_obj_list.append(_obj)
    #         participant_label_list.append(_label)
    #
    # participant_obj = np.concatenate(participant_obj_list) # (7188, 120, 12)
    # participant_label = np.concatenate(participant_label_list)
    #
    # np.save(os.path.join(os.path.join(path_resampled, folder_idx), str(folder_idx) + "_" + str(window_size) +
    #                      "_window_" + str(overlap) + "_overlap" + "_data_object.npy"), participant_obj)
    # np.save(os.path.join(os.path.join(path_resampled, folder_idx), str(folder_idx) + "_" + str(window_size) +
    #                      "_window_" + str(overlap) + "_overlap" + "_label_object.npy"), participant_label)
    #
    # print(folder_idx)
    #
    # obj_list.append(participant_obj)
    # label_list.append(participant_label)

final_obj = np.concatenate(tuple(obj_list), axis=0) # (862560, 120, 12)
final_label = np.concatenate(tuple(label_list), axis=0)
np.save(os.path.join(path_resampled, "data_object.npy"), final_obj)
np.save(os.path.join(path_resampled, "label_object.npy"), final_label)

    # file_list = os.listdir(os.path.join(path_resampled, folder_idx))
    # for file in file_list:
    #
    #     print(file)
        # obj, label = concat_objects(object_list=file)


    # _obj = np.load(os.path.join(os.path.join(path_resampled,folder_idx), "0.016sec_data_object.npy"))
    # _label = np.load(os.path.join(os.path.join(path_resampled,folder_idx), "0.016sec_label_object.npy"))
    #
    # final_obj, final_label = reshape_object(_obj, _label, window_size)
    #
    # np.save(os.path.join(path_resampled, str(window_size) + "_data_object.npy"), final_obj)
    # np.save(os.path.join(path_resampled, str(window_size) + "_label_object.npy"), final_label)
