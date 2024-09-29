"""
Author: Oh, Seungwoo
Description: (2) Create .npy file for model learning
Last modified: 2024.09.09.
"""

import os
import numpy as np
import pandas as pd

path_npy = "D:\\data_npy"

window_size = 300  # 5 sec = 300 samples

modes_phone = ['still_phone_indoor', 'walking_phone_indoor', 'crutches_phone_indoor', 'walker_phone_indoor',
               'manual_phone_indoor', 'electric_phone_indoor', 'still_phone_outdoor', 'walking_phone_outdoor',
               'crutches_phone_outdoor', 'walker_phone_outdoor', 'manual_phone_outdoor', 'electric_phone_outdoor']
modes_watch = ['still_watch_indoor', 'walking_watch_indoor', 'crutches_watch_indoor', 'walker_watch_indoor',
               'manual_watch_indoor', 'electric_watch_indoor', 'still_watch_outdoor', 'walking_watch_outdoor',
               'crutches_watch_outdoor', 'walker_watch_outdoor', 'manual_watch_outdoor', 'electric_watch_outdoor']

modes = ['still', 'walking', 'crutches', 'walker', 'manual', 'electric']
mode_dict = {'still': 0, 'walking': 1, 'crutches': 2, 'walker': 3, 'manual': 4, 'electric': 5}
sensor = ['Time', 'LAccX', 'LAccY', 'LAccZ', 'GyrX', 'GyrY', 'GyrZ', 'MagX', 'MagY', 'MagZ']


def is_phone_only(participant_num):
    # phone only: 1-60 & phone and watch: 61-120
    return int(participant_num) < 61

obj_list = []
label_list = []
obj_phone_list = []
obj_watch_list = []
label_phone_watch_list = []

for participant_num in range(120): # 120 = number of participants
    participant_num = str(participant_num+1)
    print('\nparticipant', participant_num)

    path_folder = os.path.join(path_npy, participant_num)
    path_phone = os.path.join(path_folder, 'phone')
    path_watch = os.path.join(path_folder, 'watch')

    # for each mode & environment: 'still_indoor', ... , 'electric_outdoor'
    for idx in range(12):
        mode = modes[idx % 6]  # 'still', 'walking', ..., 'electric'

        # load phone npy: participant 1-120
        phone_npy = np.load(os.path.join(path_phone, str(participant_num) + '_' + modes_phone[idx] + '.npy'))
        samples = phone_npy.shape[0]
        resampled_phone_npy = phone_npy.reshape(
            (samples // window_size, window_size, len(sensor) - 1))  # (36000, 1, 9) -> (120, 300, 9)
        label_npy = np.full(samples // window_size, mode_dict[mode])

        obj_list.append(resampled_phone_npy)  # phone 1-120                                     
        
        label_list.append(label_npy)

        # load watch npy: participant 61-120
        if is_phone_only(participant_num):
            continue
        watch_npy = np.load(os.path.join(path_watch, str(participant_num) + '_' + modes_watch[idx] + '.npy'))
        samples = watch_npy.shape[0]
        resampled_watch_npy = watch_npy.reshape(
            (samples // window_size, window_size, len(sensor) - 1))  # (36000, 1, 9) -> (120, 300, 9)

        obj_phone_list.append(resampled_phone_npy)  # phone and watch 61-120                    
        obj_watch_list.append(resampled_watch_npy)                                              
        label_phone_watch_list.append(label_npy)

# phone only: participant 1-120
final_obj = np.concatenate(tuple(obj_list), axis=0)
np.save(os.path.join(path_npy, 'data_object.npy'), final_obj)

final_label = np.concatenate(tuple(label_list), axis=0)
np.save(os.path.join(path_npy, 'six_activities_label_object.npy'), final_label)

# phone and watch: participant 61-120
final_phone_obj = np.concatenate(tuple(obj_phone_list), axis=0)
final_watch_obj = np.concatenate(tuple(obj_watch_list), axis=0)
np.save(os.path.join(path_npy, 'phone_data_object.npy'), final_phone_obj)
np.save(os.path.join(path_npy, 'watch_data_object.npy'), final_watch_obj)

final_phone_watch_label = np.concatenate(tuple(label_phone_watch_list), axis=0)
np.save(os.path.join(path_npy, 'phone_watch_six_activities_label_object.npy'), final_phone_watch_label)
