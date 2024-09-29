"""
Author: Oh, Seungwoo
Description: (3) Create label numpy files for each scenario
Last modified: 2024.09.09.
"""

import os
import numpy as np

path = "D:\\data_npy"

change_to_disabled = False
change_to_wheelchairs = True

is_phone_only = False  # True: participant 1-120, False: participant 61-120


# mode_dict = {'still': 0, 'walking': 1, 'crutches': 2, 'walker': 3, 'manual': 4, 'electric': 5}

# still / walking / disabled
# index 2,3,4,5 -> 2
def to_disabled(data):
    data[data == 3] = 2
    data[data == 4] = 2
    data[data == 5] = 2
    return data


# still / walking / walking-aid / wheelchairs
# index 1,2,3 -> 1 / 4,5 -> 2
def to_wheelchairs(data):
    data[data == 3] = 2
    data[data == 4] = 3
    data[data == 5] = 3
    return data


if is_phone_only:
    six_activities = 'six_activities_label_object.npy'
    disabled = 'disabled_label_object.npy'                              
    wheelchairs = 'wheelchairs_label_object.npy'                        
else:
    six_activities = 'phone_watch_six_activities_label_object.npy'
    disabled = 'phone_watch_disabled_label_object.npy'                  
    wheelchairs = 'phone_watch_wheelchairs_label_object.npy'            

data_six_activities = np.load(os.path.join(path, six_activities))
print(np.unique(data_six_activities, return_counts=True))

if change_to_disabled:
    data_disabled = to_disabled(data_six_activities)
    np.save(os.path.join(path, disabled), data_disabled)
    print(np.unique(data_disabled, return_counts=True))

if change_to_wheelchairs:
    data_wheelchairs = to_wheelchairs(data_six_activities)
    np.save(os.path.join(path, wheelchairs), data_wheelchairs)
    print(np.unique(data_wheelchairs, return_counts=True))
