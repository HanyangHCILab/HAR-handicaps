import os
import pandas as pd

# (0) Data cleaning: find error in raw file

# path = "C:\\Users\\user\\Documents\\2022_1\\CRC_DeepLearning\\1_data_raw"
path = "E:\\1_data_raw"

# COUNT_FILES = True
CONFIRM = False
SENSOR_DATA_LENGTH = False
DELETE_LIST = ['5010', '5019', '5032', '5089', '5090', '5090-2', '5112', '5121']
NUM_OF_PARTICIPATNS = 248 # indoor + outdoor
CHECK_WATCH = False

count = 0
# check the number of the files
folders = os.listdir(path)
for folder in folders:

    if folder in DELETE_LIST:
        continue

    folder_path = os.path.join(path, folder)
    files = os.listdir(folder_path)

    if len(files) > 2:  # phone only
        if CHECK_WATCH:
            continue
        else:
            pass
    else:  # phone + watch
        if CHECK_WATCH:
            folder_path = os.path.join(path, folder + '\\watch')
        else:
            folder_path = os.path.join(path, folder + '\\phone')
        files = os.listdir(folder_path)

    # if COUNT_FILES:
    #     if len(files) != 24 and len(files) != 30:
    #         print(folder)

    modes = {'walking': False, 'crutches': False, 'walker': False, 'manual': False, 'power': False, 'still': False}
    for file in files:
        # Check confirm file with "No"
        if CONFIRM:
            if file[-11:-4] == "Confirm":
                file_path = os.path.join(folder_path, file)
                data = pd.read_csv(file_path)
                if data['Survay3'][0] != "Yes":
                    print("Confirm: ", str(folder))

        # Check sensor data with inefficient number of lines - this only works for phone data
        if SENSOR_DATA_LENGTH:  # 5002 is not error data
            if file[-14:-4] == "SensorData":
                file_path = os.path.join(folder_path, file)
                data = pd.read_csv(file_path)
                if len(data) != 39600:
                    print("SensorData: ", str(folder), file[-25:-15])

        # check label errors   # 5002 is not error data
        if 'walking' in file:
            modes['walking'] = True
        elif 'crutches' in file:
            modes['crutches'] = True
        elif 'walker' in file:
            modes['walker'] = True
        elif 'manual' in file:
            modes['manual'] = True
        elif 'power' in file:
            modes['power'] = True
        elif 'still' in file:
            modes['still'] = True
    if False in modes.values():
        print(modes)
        print('mode error in ', folder)

    count += 1
    # print("participant num " + str(folder))
    if count == NUM_OF_PARTICIPATNS:
        exit()
