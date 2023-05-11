import os
import pandas as pd
import numpy as np
from utils_preprocessing import *

# (1) Resample: change column names & cut off initial and terminal 30 seconds (no interpolation for publishing data)

path_raw = "D:\\2023\\CRC\\CRC_data\\HAR\\1_data_raw"
path_publish =  "D:\\2023\\CRC\\CRC_data\\HAR\\data_publish"
# path_resampled = "C:\\Users\\user\\Documents\\2022_2\\HAR\\data\\data_publish_resampled"
# path_raw = "E:\\1_data_raw"
# path_resampled = "E:\\3_data_resampled"

# sensors: LAc, Gyr, Mag, Gra, Acc, Ori, RotVec, GameRotVec, Pressure, Height, Light, Step, Proxi (total 31)
change_column_name = {'LinearAccX': 'LAccX', 'LinearAccY': 'LAccY', 'LinearAccZ': 'LAccZ',
                      'GyroX': 'GyrX', 'GyroY': 'GyrY', 'GyroZ': 'GyrZ',
                      'Orientation_0_Azimuth': 'Ori_Azimuth', 'Orientation_1_Pitch': 'Ori_Pitch', 'Orientation_2_Roll':
                      'Ori_Roll', 'Survay1': 'Position'}
sensor_data_type = ['Time', 'LAccX','LAccY','LAccZ', 'GyrX', 'GyrY', 'GyrZ', 'MagX', 'MagY', 'MagZ',
                    'GraX', 'GraY', 'GraZ', 'AccX', 'AccY', 'AccZ', 'Ori_Azimuth', 'Ori_Pitch', 'Ori_Roll', 'RotVec_0',
                    'RotVec_1', 'RotVec_2', 'RotVec_3', 'RotVec_4', 'Game_RotVec_0', 'Game_RotVec_1', 'Game_RotVec_2',
                    'Game_RotVec_3', 'Pressure', 'Height', 'Light', 'Step', 'Proxi']
exclude_list = ['5010', '5019', '5032', '5051','5080', '5051-2', '5089', '5089-2', '5090', '5090-2', '5112', '5121', '5121-2']

# subject_idx = 1

path_raw_list = os.listdir(path_raw)
file_count = 0  # for each subject, only 12 files should be created
subject_idx = 1  # new idx - excluding invalid participants
pre_idx = path_raw_list[0]
for folder_idx in path_raw_list:
    if folder_idx in exclude_list:
        continue
    if(subject_idx >7 ):
        continue
    if(pre_idx[:4] != folder_idx[:4]):
        pre_idx = folder_idx

        if file_count == 12:
            print(str(subject_idx), ' end!\n\n')
            subject_idx += 1
            file_count = 0
        else:
            file_count = 0
        
    print(folder_idx, " resample start!")

    file_path = os.path.join(path_raw, folder_idx)
    file_list = os.listdir(file_path)
    if len(file_list) == 2:
        file_path = os.path.join(file_path, 'phone')
        file_list = os.listdir(file_path)

    for file in file_list:
        # if "Confirm" in file:
        #     confirm_file_path = os.path.join(file_path, file)
        #     confirm_data = pd.read_csv(confirm_file_path)
        #
        #     if confirm_data['Survay3'][0] != "Yes":
        #         continue

        if not "SensorData" in file:
            continue

        old_csv = pd.read_csv(os.path.join(file_path, file), dtype=str)
        old_csv.rename(columns=change_column_name, inplace=True)  # change names of the column
        old_csv['Time'] = pd.to_numeric(old_csv['Time']) # change 'Time' column datatype to float

        time_range = (old_csv['Time'] > 30) & (old_csv['Time'] <= 630)

        new_csv = old_csv[sensor_data_type]
        new_csv = new_csv[time_range]
        new_csv_time = new_csv['Time'] - 30
        new_csv['Time'] = new_csv_time

        path_publish_csv = os.path.join(os.path.join(path_publish, str(subject_idx)),
                                          str(subject_idx) + "_" + find_tm_type(file) +
                                          "_" + find_location_type_folder(folder_idx) +
                                          "_" + find_position(old_csv) + ".csv")

        if not os.path.exists(os.path.join(path_publish, str(subject_idx))):
            os.mkdir(os.path.join(path_publish, str(subject_idx)))

        new_csv.to_csv(path_publish_csv, mode='w', index=False) # save resampled file
        print(str(subject_idx), find_tm_type(file), find_location_type_folder(folder_idx))
        file_count += 1

  
