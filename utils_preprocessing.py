# indoor_list = ['5001', '5002', '5003', '5004', '5005', '5006', '5007', '5008', '5009', '5010', '5011', '5012', '5013', '5014',
#           '5015', '5016', '5017', '5018', '5019', '5020', '5021', '5022', '5023', '5024', '5025', '5026', '5027', '5028',
#           '5029', '5030', '5031', '5032', '5033', '5034', '5035', '5036', '5037', '5038', '5039', '5040', '5041', '5042',
#           '5043', '5044', '5045', '5046', '5047', '5048', '5049', '5050', '5051', '5052', '5053', '5054', '5055', '5056',
#           '5057', '5058', '5059', '5060', '5061-2', '5062-2', '5063-2', '5064', '5065', '5066', '5067-2', '5068',
#           '5069-2', '5070', '5071-2', '5072-2', '5073', '5074-2', '5075-2', '5076-2', '5077-2', '5078-2', '5079', '5080',
#           '5081', '5082-2', '5083', '5084', '5085', '5086-2', '5087', '5088', '5089', '5090', '5091-2', '5092-2',
#           '5093-2', '5094-2', '5095-2', '5096', '5097', '5098-2', '5099-2', '5100', '5101-2', '5102-2', '5103-2',
#           '5104-2', '5105-2', '5106', '5107', '5108','5109', '5110', '5111', '5112', '5113', '5114', '5115', '5116-2',
#           '5117-2', '5118-2', '5119-2', '5120-2', '5121-2', '5122-2', '5123-2', '5124-2', '5125-2', '5126-2', '5127']
# outdoor_list = ['5001-2', '5002-2', '5003-2', '5004-2', '5005-2', '5006-2', '5007-2', '5008-2', '5009-2', '5010-2', '5011-2',
#            '5012-2', '5013-2', '5014-2', '5015-2', '5016-2', '5017-2', '5018-2', '5019-2', '5020-2', '5021-2', '5022-2',
#            '5023-2', '5024-2', '5025-2', '5026-2', '5027-2', '5028-2', '5029-2', '5030-2', '5031-2', '5032-2', '5033-2',
#            '5034-2', '5035-2', '5036-2', '5037-2', '5038-2', '5039-2', '5040-2', '5041-2', '5042-2', '5043-2', '5044-2',
#            '5045-2', '5046-2', '5047-2', '5048-2', '5049-2', '5050-2', '5051-2', '5052-2', '5053-2', '5054-2', '5055-2',
#            '5056-2', '5057-2', '5058-2', '5059-2', '5060-2', '5061', '5062', '5063', '5064-2', '5065-2', '5066-2',
#            '5067', '5068-2', '5069', '5070-2', '5071', '5072', '5073-2', '5074', '5075', '5076', '5077', '5078',
#            '5079-2', '5080-2', '5081-2', '5082', '5083-2', '5084-2', '5085-2', '5086', '5087-2', '5088-2', '5089-2',
#            '5090-2', '5091', '5092', '5093', '5094', '5095', '5096-2', '5097-2', '5098', '5099', '5100-2', '5101',
#            '5102', '5103', '5104', '5105', '5106-2', '5107-2', '5108-2', '5109-2', '5110-2', '5111-2', '5112-2',
#            '5113-2', '5114-2', '5115-2', '5116', '5117', '5118', '5119', '5120', '5121', '5122', '5123', '5124', '5125',
#            '5126', '5127-2']
exclude_list = ['5010', '5010-2', '5019', '5019-2', '5032', '5032-2', '5051', '5051-2', '5089', '5089-2', '5112',
                '5112-2', '5121', '5121-2']

def find_tm_type(file_name):
    TransportationType = ['crutches', 'walker', 'manualChar', 'powerChar', 'still', 'walking']
    TransportationType_watch = {'walking': 'walking', 'crutches': 'crutches', 'still': 'still', 'manual': 'manual',
                                'motorized': 'electric', 'walker': 'walker'}
    TransportationType_new = {'Walk_': 'walking', 'Crutches': 'crutches', 'Electric wheelchair': 'electric',
                              'Walker': 'walker', 'Still': 'still', 'Manual wheelchair': 'manual'}
    TransportationType_publish = {'crutches': 'crutches', 'walker': 'walker', 'manualChar': 'manual',
                                  'powerChar': 'electric', 'still': 'still', 'walking': 'walking'}
    TransportationType_publish_resampled = ['crutches', 'walker', 'manual', 'electric', 'still', 'walking']
    # for tm_type, tm_type_publish in TransportationType_publish.items():
    #     if tm_type_publish in file_name:
    #         return tm_type
    for tm_type_watch, tm_type in TransportationType_watch.items():
        if tm_type_watch in file_name:
            return tm_type
    for tm_type in TransportationType_publish_resampled:
        if tm_type in file_name:
            return tm_type
    for tm_type in TransportationType:
        if tm_type in file_name:
            return TransportationType_publish[tm_type]
    for tm_type_new, tm_type in TransportationType_new.items():
        if tm_type_new in file_name:
            return tm_type
    return -1

def find_location_type(file_name):
    LocationType = ['indoor', 'outdoor']
    LocationType_new = {'Indoor': 'indoor', 'Outdoor': 'outdoor'}

    for location_type in LocationType:
        if location_type in file_name:
            return location_type
    for location_type_new, location_type in LocationType_new.items():
        if location_type_new in file_name:
            return location_type
    return -1

def find_location_type_folder(folder_name):
    # if folder_name in indoor_list:
    #     return "indoor"
    # elif folder_name in outdoor_list:
    #     return "outdoor"
    if len(folder_name) == 4:
        return "indoor"
    elif len(folder_name) == 6:
        return "outdoor"
    elif folder_name in exclude_list:
        return "exclude"

def find_tm_location_type(folder_name):
    tm_location = ['still_in', 'still_out', 'walking_in', 'walking_out', 'crutches_in', 'crutches_out', 'walker_in',
                   'walker_out', 'manual_in', 'manual_out', 'electric_in', 'electric_out']

    for tl in tm_location:
        if tl in folder_name:
            return tl

    return -1

def find_position(df):
    positionType = {'Etc': 'others', 'Pocket': 'pocket', 'Hand': 'hand', 'Bag': 'bag'}
    positionType_new = ['others', 'pocket', 'hand', 'bag']

    position = df['Position'][0]
    position_type = positionType[position]
    return position_type


import numpy as np

def concat_objects(object_list, label_list):
    final_obj = np.concatenate(tuple(object_list), axis=0)
    final_label = np.concatenate(tuple(label_list), axis=0)

    print("Concat is done!!")

    return final_obj, final_label

def reshape_object(basic_object_npy, basic_label_npy, window_size):
    data_size, _current_window_size, col_size = basic_object_npy.shape

    final_obj = np.reshape(basic_object_npy, (data_size // window_size, window_size, col_size))
    final_label = np.reshape(basic_label_npy, (data_size // window_size, window_size))

    final_label = np.unique(final_label, axis=1)

    if final_label.shape[1] == 1:
        final_label = final_label.reshape(data_size // window_size)
    else:
        print("error")
        return None, None

    return final_obj, final_label
