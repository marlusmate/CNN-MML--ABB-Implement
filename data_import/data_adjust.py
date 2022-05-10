import shutil
import json
import numpy as np
import pandas as pd

from data_import.data_import import read_json, load_json


def get_exp_date(data_point):
    if len(data_point[0]) == 1:
        file_name = data_point
    else:
        file_name = data_point[0]
    exp_dates = ["02-08", "02-09", "02-22", "02-25", "03-09"]
    exp_date = [date for date in exp_dates if date in file_name]

    return exp_date[0]


def get_exp_nr(data_point):
    if len(data_point[1]) == 1:
        file_name = data_point
    else:
        file_name = data_point[1]
        exp_numbers = [f"exp0{i}" for i in np.arange(0,8,1)]
        exp_nr = [date for date in exp_numbers if date in file_name]

        return exp_nr[0]


def add_key_content_value(json_dict, key_one, key_two, key_content, key_three="value"):
    json_dict.update({key_one: {key_two: {key_three: key_content}}})

    return json_dict


def is_gfl_setpoint(gasflow, gfl_setpoints=None):
    if gfl_setpoints is None:
        gfl_setpoints = [10, 15, 20, 25, 30, 35, 40, 50]
    elif gfl_setpoints.shape == ():
        gfl_setpoints = [gfl_setpoints, gfl_setpoints]

    if any([int(gasflow) in np.arange(setpoint - 2, setpoint + 2) for setpoint in gfl_setpoints]):
        return True
    else:
        return False


def get_flow_regime(proc_date, proc_exp, proc_gfl, proc_rpm, data_point, exp_log=None):

    # exp columns: Datum; ExpNr; rpm; gasflow; w-gly; temp; Fr; Fl; Dichte; Visk; Re; ofs-luft; Label; in-use
    if exp_log is None:
        exp_log = pd.read_csv("experiment_log.csv", delimiter=";")
        exp_log["Datum"] = [date.replace('_', '-') for date in exp_log["Datum"]]

    # Get all Rows of Setpoints with "proc_date" as date of experiment
    exp_day = exp_log[exp_log["Datum"] == proc_date[-5:]]
    exp_sp = exp_day[exp_day["ExpNr"] == "exp" + proc_exp]
    # Get all Setpoints of certain experiment number
    exp_gfl = pd.DataFrame([exp_sp.iloc[idx, :] for idx in np.arange(len(exp_sp)) if
                         is_gfl_setpoint(proc_gfl, exp_sp["gasflow"][exp_sp.index[idx]])], columns=exp_sp.columns)  # Get all Setpoints with matching gfl
    if not len(exp_gfl) == 0:
        exp_rpm = [exp_gfl.iloc[idx, :] for idx in np.arange(len(exp_gfl)) if
                   exp_gfl["rpm"][exp_gfl.index[idx]] == proc_rpm]  # Get Setpoint with also matching RPM
    else:
        exp_rpm = []

    if len(exp_rpm) == 1:
        if exp_rpm[0][-1] == 1:
            label = int(exp_rpm[0][-2])
        else:
            print("Experiment not in use, ", data_point)
            label = 3
    else:

        print("problematic file, could not determine label from ""experimantal_log.csv""", data_point[1])
        label = 3

    return label


def set_flow_regime(proc_dict, label):
    proc_dict["flow_regime"]["data"]["value"] = label

    return proc_dict


def change_flow_regime(data_point, label_mat):
    """
    This Changes the label of the data-point, according to label_mat.
    Overwrites old json-file
    :param data_point: single data point (see function get_data_points_list()).
    :param label_mat: contains matrix of set-points and associated labels [gasflow, rpm, label].
    :return: None(, overwritten label in json-file)
    """
    file = load_json(data_point)
    prc_data = read_json(data_point)
    if prc_data is None:
        return
    for i in np.arange(len(label_mat)):
        if label_mat[i][1] == prc_data[0] and int(prc_data[1]) in np.arange(label_mat[i][0] - 2, label_mat[i][0] + 2):
            file["flow_regime"]["data"]["value"] = label_mat[i][2]
            out_file = open(data_point[1], 'w+')
            json.dump(file, out_file, indent=4)
        else:
            continue
    return


def change_data_dir(data_point, target_dir):
    """
    Moves data_point (image, metadata(json)) to target_dir
    :param data_point:
    :param target_dir:
    :return:
    """
    shutil.move(data_point[0], target_dir)
    shutil.move(data_point[1], target_dir)
    print(data_point, "moved to ", target_dir)
    return


def sort_data_points(data_points, set_points, num):
    """
    This returns two lists of data points, the original list is divided by
    checking whether param value is located within setpoints.
    Loads each json file (default params, see function read_json() )
    :param data_points: original list of data points
    :param set_points: Dict with np.arrange of setpoints (dict.values())
    :param num: param from json file for decision criterium
    :return: Two np.arrays of the data_points
    """
    data_points_setpoint = []
    data_points_trans = []
    for data_point in data_points:
        prc = read_json(data_point)
        if prc is None:
            continue
        if not any(int(prc[num]) in set_points[i] for i in set_points):
            data_points_trans.append(data_point)
            continue

        data_points_setpoint.append(data_point)

    data_points_setpoint = np.array(data_points_setpoint)
    data_points_trans = np.array(data_points_trans)

    return data_points_setpoint, data_points_trans
