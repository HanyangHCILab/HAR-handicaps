### change valid 45 people's data (phone + watch) to npy

import os
import pandas as pd
import numpy as np
from utils_preprocessing import *


window_size = 120
overlap = 60

modes = ['still', 'walking', 'crutches', 'walker', 'manual', 'motorized']
mode_dict = {'still': 0, 'walking': 1, 'crutches': 2, 'walker': 3, 'manual': 4, 'motorized': 5}
sensor_type = {'LAccX': float, 'LAccY': float, 'LAccZ': float,
               'gyroX': float, 'gyroY': float, 'gyroZ': float,
               'magX': float, 'magY': float, 'magZ': float}

path_valid = "D:\\2023\\CRC\\CRC_data\\HAR\\valid_phone_watch_data"

obj_list = []
label_list = []
folder_list = os.listdir(path_valid)
for folder in folder_list:  # 피험자 폴더별로 접근
    print('participant', folder)
    path_folder = os.path.join(path_valid, folder)
    path_phone = os.path.join(path_folder, 'phone')
    path_watch = os.path.join(path_folder, 'watch')

    for mode in modes:
        print('--', mode)

        # phone data
        path_phone_files = os.listdir(path_phone)
        for file in path_phone_files:
            if mode in file:
                csv_phone = pd.read_csv(os.path.join(path_phone, file), dtype=str)
        csv_phone.set_index('Time', inplace=True)
        for col, col_type in sensor_type.items():
            csv_phone[col] = csv_phone[col].astype(col_type)
        csv_phone_to_npy = csv_phone.to_numpy().reshape(36000, 1, 9)

        # 5s window
        resampled_csv_phone_to_npy = csv_phone_to_npy.reshape(120, 300, 9)
        data_size, _current_window_size, col_size = resampled_csv_phone_to_npy.shape
        # 2s 50% window
        #data_size, _current_window_size, col_size = csv_phone_to_npy.shape
        # size = (1, window_size, col_size)  # (1, 120, 9)
        # phone_window_obj = []
        # for i in range(data_size // overlap - 1):
        #     window_slice = csv_phone_to_npy[i * overlap: i * overlap + window_size, :, :].reshape(size)
        #     phone_window_obj.append(window_slice)
        #
        # phone_obj = np.concatenate(tuple(phone_window_obj))  # (599, 120, 9)
        # phone_participant_obj_list.append(phone_obj)

        # watch data
        path_watch_files = os.listdir(path_watch)
        for file in path_watch_files:
            if mode in file:
                csv_watch = pd.read_csv(os.path.join(path_watch, file), dtype=str)
        csv_watch.set_index('Time', inplace=True)
        for col, col_type in sensor_type.items():
            csv_watch[col] = csv_watch[col].astype(col_type)
        csv_watch_to_npy = csv_watch.to_numpy().reshape(36000, 1, 9)

        # 5s window
        resampled_csv_watch_to_npy = csv_watch_to_npy.reshape(120, 300, 9)

        # # 2s 50% window
        # data_size, _current_window_size, col_size = csv_watch_to_npy.shape
        # size = (1, window_size, col_size)  # (1, 120, 9)
        # watch_window_obj = []
        # for i in range(data_size // overlap - 1):
        #     window_slice = csv_watch_to_npy[i * overlap: i * overlap + window_size, :, :].reshape(size)
        #     watch_window_obj.append(window_slice)
        #
        # watch_obj = np.concatenate(tuple(watch_window_obj))  # (599, 120, 9)
        # # watch_participant_obj_list.append(phone_obj)

        # phone + watch
        resampled_csv_to_npy = np.concatenate([resampled_csv_phone_to_npy, resampled_csv_watch_to_npy], axis=-1)
        resampled_mode_type = np.full(data_size, mode_dict[mode])
        #resampled_csv_to_npy = np.concatenate([phone_obj, watch_obj], axis=-1)
        #resampled_mode_type = np.full((data_size // overlap - 1), mode_dict[mode])

        print(resampled_csv_to_npy.shape)
        print(resampled_mode_type.shape)
        obj_list.append(resampled_csv_to_npy)
        label_list.append(resampled_mode_type)

final_obj = np.concatenate(tuple(obj_list), axis=0)
final_label = np.concatenate(tuple(label_list), axis=0)
print(final_obj.shape)
# print(final_label.shape)
np.save(os.path.join(path_valid, '5s_valid_data_obj.npy'), final_obj)
np.save(os.path.join(path_valid, '5s_valid_label_obj.npy'), final_label)
