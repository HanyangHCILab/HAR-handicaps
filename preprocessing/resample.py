import os
import pandas as pd
import numpy as np
from utils_preprocessing import *

# (2) Resample: choose specific columns (sensor AGM for recognition) & interpolate -> save as 0.016.npy

path_raw = "D:\\2023\\CRC\\CRC_data\\HAR\\1_data_raw"
path_resampled = "D:\\2023\\CRC\\CRC_data\\HAR\\data_publish_resampled"
# path_raw = "C:\\Users\\user\\Documents\\2022_2\\HAR\\data\\data_publish"
# path_resampled = "C:\\Users\\user\\Documents\\2022_2\\HAR\\data\\data_publish_resampled"
NUM_OF_PARTICIPATNS = 240 # indoor and outdoor condition is separated
DELETE_LIST = ['5010', '5019', '5032', '5032-2', '5051', '5051-2', '5089', '5089-2', '5112', '5112-2', '5121', '5121-2']

# time table for interpolation
time_table_second = [0.0167, 0.0334, 0.0501, 0.0668, 0.0835, 0.1002, 0.1169, 0.1336, 0.1503, 0.167, 0.1837, 0.2004,
                     0.2171, 0.2338, 0.2505, 0.2672, 0.2839, 0.3006, 0.3173, 0.334, 0.3507, 0.3674, 0.3841, 0.4008,
                     0.4175, 0.4342, 0.4509, 0.4676, 0.4843, 0.501, 0.5177, 0.5344, 0.5511, 0.5678, 0.5845, 0.6012,
                     0.6179, 0.6346, 0.6513, 0.668, 0.6847, 0.7014, 0.7181, 0.7348, 0.7515, 0.7682, 0.7849, 0.8016,
                     0.8183, 0.835, 0.8517, 0.8684, 0.8851, 0.9018, 0.9185, 0.9352, 0.9519, 0.9686, 0.9853, 1]

# data types
change_column_name = {'LinearAccX': 'LAccX', 'LinearAccY': 'LAccY', 'LinearAccZ': 'LAccZ',
                      'GyroX': 'GyrX', 'GyroY': 'GyrY', 'GyroZ': 'GyrZ'}
resampled_type = {'Time': float, 'LAccX': float, 'LAccY': float, 'LAccZ': float,
                  'GyrX': float, 'GyrY': float, 'GyrZ': float,
                  'MagX': float, 'MagY': float, 'MagZ': float}
                  # 'GraX': float, 'GraY': float, 'GraZ': float}
sensor_data_type = ['LAccX','LAccY','LAccZ', 'GyrX', 'GyrY', 'GyrZ', 'MagX', 'MagY', 'MagZ'] #, 'GraX', 'GraY', 'GraZ']
mode_dict = {'still': 0, 'walking': 1, 'crutches': 2, 'walker': 3, 'manual': 4, 'electric': 5}
tm_loc_dict = {'still_in': 0, 'still_out': 1, 'walking_in': 2, 'walking_out': 3, 'crutches_in': 4, 'crutches_out': 5,
               'walker_in': 6, 'walker_out': 7, 'manual_in': 8, 'manual_out': 9, 'electric_in': 10, 'electric_out': 11}

# interpolation
time_table = []
for i in range(0, 600):
    for j in range(0, 60):
        time_table.append(time_table_second[j] + i)

count = 0
path_raw_list = os.listdir(path_raw)
for folder_idx in path_raw_list: # 1, 2, 3, ...
    if folder_idx in DELETE_LIST:
        continue
    if (int(folder_idx) >7):
        continue
    obj_list = []
    label_list = []

    file_list = os.listdir(os.path.join(path_raw, folder_idx))
    for file in file_list: # 1_crutches_indoor_pocket.csv, ...

        if ".txt" in file:
            continue

        old_csv = pd.read_csv(os.path.join(os.path.join(path_raw, folder_idx), file), dtype=str)
        old_csv.rename(columns=change_column_name, inplace=True)  # change names of the column

        for col, col_type in resampled_type.items():
            old_csv[col] = old_csv[col].astype(col_type) # change data types
        old_csv.set_index("Time", inplace=True) # set 'Time' column as index

        new_csv = pd.DataFrame({'Time': time_table})

        for i in range(len(sensor_data_type)):
            new_csv.insert(i+1, sensor_data_type[i], np.nan)
        new_csv.insert(len(sensor_data_type)+1, 'isNew', 'T')
        new_csv = new_csv.set_index('Time')

        old_csv.insert(len(sensor_data_type), 'isNew', np.nan)

        new_csv = pd.concat([old_csv, new_csv], sort=True)

        new_csv.sort_values(by=['Time', 'isNew'], inplace=True)

        new_csv.reset_index(inplace=True)
        new_csv.drop_duplicates('Time', keep='last', inplace=True)
        new_csv.set_index('Time', inplace=True)

        new_csv[sensor_data_type] = new_csv[sensor_data_type].interpolate(method='index')

        new_csv.reset_index(inplace=True)
        new_csv = new_csv.loc[new_csv['Time'].isin(time_table)]
        new_csv.set_index('Time', inplace=True)

        new_csv = new_csv[sensor_data_type]

        # print(new_csv)

        # path_resampled_csv = os.path.join(os.path.join(path_resampled, folder_idx),
        #                                   folder_idx + "_" + find_tm_type(file) + "_SensorData_Resample.csv")
        path_resampled_csv = os.path.join(os.path.join(path_resampled, folder_idx), file)

        if not os.path.exists(os.path.join(path_resampled, folder_idx)):
            os.mkdir(os.path.join(path_resampled, folder_idx))

        new_csv.to_csv(path_resampled_csv, index='Time')

        csv_to_npy = new_csv.to_numpy()
        mode_type = mode_dict[find_tm_type(file)] # 6 class label
        # mode_type = tm_loc_dict[find_tm_location_type(file)] # 12 class label
        print(file)

        resampled_csv_to_npy = csv_to_npy.reshape(36000, 1, 9)
        resampled_mode_type = np.full((36000), mode_type)

        # # for 2 sec, 50% overlap window
        # np.save(os.path.join(os.path.join(path_resampled, folder_idx), folder_idx + "_" + find_tm_type(file) + "_" +
        #                       find_location_type(file) + "_0.016sec_data_object.npy"), resampled_csv_to_npy)
        # np.save(os.path.join(os.path.join(path_resampled, folder_idx), folder_idx + "_" + find_tm_type(file) + "_" +
        #                      find_location_type(file) + "_0.016sec_label_object.npy"), resampled_mode_type)

        # for 5 sec, no overlap window
        obj_list.append(resampled_csv_to_npy)
        label_list.append(resampled_mode_type)

    # for 5 sec, no overlap window
    # print(label_list)
    final_obj = np.concatenate(tuple(obj_list), axis=0)
    final_label = np.concatenate(tuple(label_list), axis=0)
    np.save(os.path.join(os.path.join(path_resampled, folder_idx),folder_idx + "_0.016sec_data_object.npy"), final_obj)
    np.save(os.path.join(os.path.join(path_resampled, folder_idx),folder_idx + "_0.016sec_label_object.npy"), final_label)

    count += 1
    if count == NUM_OF_PARTICIPATNS:
        exit()

    print("Files in  " + folder_idx + " are created!")


print("Resampling finished!")