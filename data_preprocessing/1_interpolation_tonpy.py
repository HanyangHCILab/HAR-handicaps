"""
Author: Oh, Seungwoo
Description: (1) Interpolate data at 60Hz and convert to .npy file
Last modified: 2024.09.09.
"""

import os
import numpy as np
import pandas as pd

path = "D:\\data_publish"
path_npy = "D:\\data_npy"

# time table for interpolation
time_table_second = [0.0167, 0.0334, 0.0501, 0.0668, 0.0835, 0.1002, 0.1169, 0.1336, 0.1503, 0.167, 0.1837, 0.2004,
                     0.2171, 0.2338, 0.2505, 0.2672, 0.2839, 0.3006, 0.3173, 0.334, 0.3507, 0.3674, 0.3841, 0.4008,
                     0.4175, 0.4342, 0.4509, 0.4676, 0.4843, 0.501, 0.5177, 0.5344, 0.5511, 0.5678, 0.5845, 0.6012,
                     0.6179, 0.6346, 0.6513, 0.668, 0.6847, 0.7014, 0.7181, 0.7348, 0.7515, 0.7682, 0.7849, 0.8016,
                     0.8183, 0.835, 0.8517, 0.8684, 0.8851, 0.9018, 0.9185, 0.9352, 0.9519, 0.9686, 0.9853, 1]
time_table = []
for i in range(0, 600):
    for j in range(0, 60):
        time_table.append(time_table_second[j] + i)


def is_phone_only(participant_num):
    # phone only: 1-60 & phone and watch: 61-120
    return int(participant_num) < 61


modes_phone = ['still_phone_indoor', 'walking_phone_indoor', 'crutches_phone_indoor', 'walker_phone_indoor',
               'manual_phone_indoor', 'electric_phone_indoor', 'still_phone_outdoor', 'walking_phone_outdoor',
               'crutches_phone_outdoor', 'walker_phone_outdoor', 'manual_phone_outdoor', 'electric_phone_outdoor']
modes_watch = ['still_watch_indoor', 'walking_watch_indoor', 'crutches_watch_indoor', 'walker_watch_indoor',
               'manual_watch_indoor', 'electric_watch_indoor', 'still_watch_outdoor', 'walking_watch_outdoor',
               'crutches_watch_outdoor', 'walker_watch_outdoor', 'manual_watch_outdoor', 'electric_watch_outdoor']

sensor = ['LAccX', 'LAccY', 'LAccZ', 'GyrX', 'GyrY', 'GyrZ', 'MagX', 'MagY', 'MagZ']

folder_list = os.listdir(path)
for folder in folder_list:
    participant_num = folder

    print(folder, 'resample and interpolation start!')
    path_folder = os.path.join(path, folder)
    path_phone = os.path.join(path_folder, 'phone')
    path_watch = os.path.join(path_folder, 'watch')

    for idx in range(12):  # 'still_indoor', ... , 'electric_outdoor'
        print('--', modes_phone[idx])

        # phone - interpolate
        phone_file = str(participant_num) + '_' + modes_phone[idx] + '.csv'
        csv_phone = pd.read_csv(os.path.join(path_phone, phone_file), dtype=str)

        csv_phone = csv_phone.astype(dtype=float)
        csv_phone.set_index('Time', inplace=True)
        csv_phone = csv_phone[sensor]  # use only LAcc, Gyr, and Mag

        new_csv_phone = pd.DataFrame({'Time': time_table})

        for i in range(len(sensor)):
            new_csv_phone.insert(i + 1, sensor[i], np.nan)
        new_csv_phone.insert(len(sensor) + 1, 'isNew', 'T')
        new_csv_phone = new_csv_phone.set_index('Time')

        csv_phone.insert(len(sensor), 'isNew', np.nan)

        new_csv_phone = pd.concat([csv_phone, new_csv_phone], sort=True)
        new_csv_phone.sort_values(by=['Time', 'isNew'], inplace=True)
        new_csv_phone.reset_index(inplace=True)
        new_csv_phone.drop_duplicates('Time', keep='last', inplace=True)
        new_csv_phone.set_index('Time', inplace=True)

        new_csv_phone[sensor] = new_csv_phone[sensor].interpolate(method='index')

        new_csv_phone.reset_index(inplace=True)
        new_csv_phone = new_csv_phone.loc[new_csv_phone['Time'].isin(time_table)]
        new_csv_phone.set_index('Time', inplace=True)
        new_csv_phone = new_csv_phone[sensor]

        # phone - save as npy
        path_save_phone = os.path.join(os.path.join(path_npy, participant_num), 'phone')
        if not os.path.exists(path_npy):
            os.mkdir(path_npy)
        if not os.path.exists(path_save_phone):
            os.mkdir(os.path.join(path_npy, participant_num))
            os.mkdir(path_save_phone)

        path_save_phone_csv = os.path.join(path_save_phone, str(participant_num) + '_' + modes_phone[idx] + '.npy')
        # new_csv_phone.to_csv(path_save_phone_csv, mode='w', index=True)
        phone_npy = new_csv_phone.to_numpy()
        np.save(path_save_phone_csv, phone_npy)

        # watch - interpolate
        if is_phone_only(participant_num):
            continue
        print('--', modes_watch[idx])

        watch_file = str(participant_num) + '_' + modes_watch[idx] + '.csv'
        csv_watch = pd.read_csv(os.path.join(path_watch, watch_file), dtype=str)

        csv_watch = csv_watch.astype(dtype=float)
        csv_watch.set_index('Time', inplace=True)
        csv_watch = csv_watch[sensor]

        new_csv_watch = pd.DataFrame({'Time': time_table})

        for i in range(len(sensor)):
            new_csv_watch.insert(i + 1, sensor[i], np.nan)
        new_csv_watch.insert(len(sensor) + 1, 'isNew', 'T')
        new_csv_watch = new_csv_watch.set_index('Time')

        csv_watch.insert(len(sensor), 'isNew', np.nan)

        new_csv_watch = pd.concat([csv_watch, new_csv_watch], sort=True)
        new_csv_watch.sort_values(by=['Time', 'isNew'], inplace=True)
        new_csv_watch.reset_index(inplace=True)
        new_csv_watch.drop_duplicates('Time', keep='last', inplace=True)
        new_csv_watch.set_index('Time', inplace=True)

        new_csv_watch[sensor] = new_csv_watch[sensor].interpolate(method='index')

        new_csv_watch.reset_index(inplace=True)
        new_csv_watch = new_csv_watch.loc[new_csv_watch['Time'].isin(time_table)]
        new_csv_watch.set_index('Time', inplace=True)
        new_csv_watch = new_csv_watch[sensor]

        # watch - save as npy
        path_save_watch = os.path.join(os.path.join(path_npy, participant_num), 'watch')
        if not os.path.exists(path_save_watch):
            os.mkdir(path_save_watch)

        path_save_watch_csv = os.path.join(path_save_watch, str(participant_num) + '_' + modes_watch[idx] + '.npy')
        # new_csv_watch.to_csv(path_save_watch_csv, mode='w', index=True)
        watch_npy = new_csv_watch.to_numpy()
        np.save(path_save_watch_csv, watch_npy)

