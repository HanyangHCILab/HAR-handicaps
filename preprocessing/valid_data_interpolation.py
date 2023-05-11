### match valid 45 people's data (phone + watch) and interpolate at 60Hz

import os
import pandas as pd
import numpy as np
from utils_preprocessing import *

path_raw = "D:\\2023\\CRC\\CRC_data\\HAR\\1_data_raw"
path_valid = "D:\\2023\\CRC\\CRC_data\\HAR\\valid_phone_watch_data"

invalid = ['5061', '5062', '5063', '5065', '5068', '5070', '5071', '5073', '5074', '5075', '5076', '5078', '5079',
           '5082', '5083', '5088', '5089', '5090', '5093', '5099', '5112', '5121']
phone_only = [str(i) for i in range(5001, 5061)]

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

modes_phone = ['still', 'walking', 'crutches', 'walker', 'manualChar', 'powerChar']
modes_watch = ['still', 'walking', 'crutches', 'walker', 'manual', 'motorized']

change_column_name = {'LinearAccX': 'LAccX', 'LinearAccY': 'LAccY', 'LinearAccZ': 'LAccZ',
                      'GyroX': 'gyroX', 'GyroY': 'gyroY', 'GyroZ': 'gyroZ',
                      'MagX': 'magX', 'MagY': 'magY', 'MagZ': 'magZ'}
phone_sensor_type = {'Time': float, 'LAccX': float, 'LAccY': float, 'LAccZ': float,
                     'gyroX': float, 'gyroY': float, 'gyroZ': float,
                     'magX': float, 'magY': float, 'magZ': float}
watch_sensor_type = {'Time': float, 'LAccX': float, 'LAccY': float, 'LAccZ': float,
                     'gyroX': float, 'gyroY': float, 'gyroZ': float,
                     'magX': float, 'magY': float, 'magZ': float}
sensor = ['LAccX', 'LAccY', 'LAccZ', 'gyroX', 'gyroY', 'gyroZ', 'magX', 'magY', 'magZ']

# TODO: 파일 불러오기
folder_list = os.listdir(path_raw)
for folder in folder_list:  # 피험자 폴더별로 접근
    participant_num = folder[:4]

    if participant_num in phone_only:
        continue
    if participant_num in invalid:
        continue

    print(folder, 'concat start!')
    path_folder = os.path.join(path_raw, folder)
    path_phone = os.path.join(path_folder, 'phone')
    path_watch = os.path.join(path_folder, 'watch')

    for idx in range(6):  # 각 mode에 대하여 접근
        print('--', modes_watch[idx])

        mode_phone = modes_phone[idx]
        mode_watch = modes_watch[idx]

        path_phone_files = os.listdir(path_phone)
        for file in path_phone_files:
            if 'SensorData.csv' in file and mode_phone in file:
                csv_phone = pd.read_csv(os.path.join(path_phone, file), dtype=str)

        path_watch_files = os.listdir(path_watch)
        for file in path_watch_files:
            if mode_watch in file:
                csv_watch = pd.read_csv(os.path.join(path_watch, file), dtype=str)

        # TODO: 시간 맞추기
        phone_min, phone_sec = int(csv_phone['Min'][0]), int(csv_phone['Sec'][0])
        watch_min, watch_sec = int(csv_watch['min'][0]), int(csv_watch['sec'][0])

        if phone_min == watch_min:
            start_min = phone_min
            start_sec = (max(phone_sec, watch_sec) + 1) % 60
            if start_sec == 0:
                start_min += 1
        else:
            start_min = max(phone_min, watch_min)
            start_sec = (min(phone_sec, watch_sec) + 1) % 60

        end_min = (start_min + 10) % 60
        end_sec = start_sec
        # if end_sec == 0:
        #     end_min += 1

        print(start_min, start_sec)
        print(end_min, end_sec)
        csv_phone_start_idx = csv_phone[csv_phone.Min == str(start_min)]
        csv_phone_start_idx = csv_phone_start_idx[csv_phone_start_idx.Sec == str(start_sec)]
        csv_watch_start_idx = csv_watch[csv_watch['min'] == str(start_min)]
        csv_watch_start_idx = csv_watch_start_idx[csv_watch_start_idx['sec'] == str(start_sec)]
        # print(csv_phone.index[0])
        # print(csv_watch.index[0])
        # exit()
        phone_start_idx = csv_phone_start_idx.index[0]
        watch_start_idx = csv_watch_start_idx.index[0]
        # print(phone_start_idx)
        # print(phone_start_idx, watch_start_idx)
        # exit()

        # print(end_min, end_sec)
        csv_phone_end_idx = csv_phone[csv_phone.Min == str(end_min)]
        csv_phone_end_idx = csv_phone_end_idx[csv_phone_end_idx.Sec == str(end_sec)]
        csv_watch_end_idx = csv_watch[csv_watch['min'] == str(end_min)]
        csv_watch_end_idx = csv_watch_end_idx[csv_watch_end_idx['sec'] == str(end_sec)]

        phone_end_idx = csv_phone_end_idx.index[-1]
        watch_end_idx = csv_watch_end_idx.index[-1]
        # print(phone_end_idx, watch_end_idx)
        # exit()

        csv_phone = csv_phone[phone_start_idx:phone_end_idx]
        csv_watch = csv_watch[watch_start_idx:watch_end_idx]
        # print(csv_phone, csv_watch)
        # exit()

        # TODO: interpolation
        ### phone
        csv_phone.rename(columns=change_column_name, inplace=True)
        for col, col_type in phone_sensor_type.items():
            csv_phone[col] = csv_phone[col].astype(col_type)
        csv_phone['Time'] = csv_phone['Time'] - csv_phone['Time'].iloc[0]
        # print(csv_phone)
        csv_phone.set_index('Time', inplace=True)
        csv_phone = csv_phone[sensor]

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

        # print(new_csv_phone)
        # exit()

        ### watch
        watch_time_col = []
        for i in range(len(csv_watch)):
            minute = ((int(csv_watch['min'].iloc[i]) + 60 - int(csv_watch['min'].iloc[0])) % 60) * 60
            second = (int(csv_watch['sec'].iloc[i]) + 60 - start_sec) % 60
            millis = float(csv_watch['millis'].iloc[i]) / 1000
            watch_time_col.append(minute + second + millis)
        # print(watch_time_col)

        csv_watch.insert(0, 'Time', watch_time_col)
        for col, col_type in watch_sensor_type.items():
            csv_watch[col] = csv_watch[col].astype(col_type)
        csv_watch.set_index('Time', inplace=True)
        csv_watch = csv_watch[sensor]
        # print(csv_watch)
        # exit()

        new_csv_watch = pd.DataFrame({'Time': time_table})

        for i in range(len(sensor)):
            new_csv_watch.insert(i + 1, sensor[i], np.nan)
        new_csv_watch.insert(len(sensor) + 1, 'isNew', 'T')
        new_csv_watch = new_csv_watch.set_index('Time')

        csv_watch.insert(len(sensor), 'isNew', np.nan)
        # print(csv_watch)

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

        # print(new_csv_watch)
        # exit()

        # TODO: save interpolated csv
        path_save_phone = os.path.join(os.path.join(path_valid, folder), 'phone')
        if not os.path.exists(path_save_phone):
            os.mkdir(os.path.join(path_valid, folder))
            os.mkdir(path_save_phone)
        path_save_watch = os.path.join(os.path.join(path_valid, folder), 'watch')
        if not os.path.exists(path_save_watch):
            os.mkdir(path_save_watch)

        path_save_phone_csv = os.path.join(path_save_phone, 'Resampled_' + modes_watch[idx] + '_phone.csv')
        new_csv_phone.to_csv(path_save_phone_csv, mode='w', index=True)
        path_save_watch_csv = os.path.join(path_save_watch, 'Resampled_' + modes_watch[idx] + '_watch.csv')
        new_csv_watch.to_csv(path_save_watch_csv, mode='w', index=True)
