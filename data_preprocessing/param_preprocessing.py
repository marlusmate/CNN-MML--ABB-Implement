# Import
import os
import random
import json
#import numpy as np
import pandas as pd
from data_import.data_import import get_data_points_list, load_json
from data_import.data_adjust import add_key_content_value, get_exp_date, is_gfl_setpoint, get_flow_regime, get_exp_nr \
    , set_flow_regime

"""
This Scripts contains all preprocessing Steps of the Process Data before Training the Model.
In an ideal Situation this has to be executed only once.
Returns preprocessed adjusted Metadata-File, saved in two new target dirs (Training,Eval & Final Testing).
Random SEED und split RATIO must be the same as in scrip "param_preprocessing"
"""

# Directories
source_dir = 'C:/Users/Markus Esser/PycharmProjects/MML-Beleg/data/example_pictures'
target_dir_test = 'C:/Users/Markus Esser/PycharmProjects/MML-Beleg/data/example_pictures-test'
target_dir_train = 'C:/Users/Markus Esser/PycharmProjects/MML-Beleg/data/example_pictures-train'
target_dir_trans = 'C:/Users/Markus Esser/PycharmProjects/MML-Beleg/data/example_pictures-trans'
target_dir_troub = 'C:/Users/Markus Esser/PycharmProjects/MML-Beleg/data/example_pictures-troub'
exp_log_dir = "../data_import/experiment_log.csv"

# Set random Seed for reproducibility
random.seed(73)

# Split Final Test Set
split_ratio = [0.9, 0.1]

# Get Experiment Log
exp_log = pd.read_csv(exp_log_dir, delimiter=";")

# Get data_points, shuffeld list
data_points = get_data_points_list(source_dir)
shuffled_data_points = random.sample(data_points, len(data_points))
dataset_len = len(shuffled_data_points)

# Training-, Validation Set
no_train_points = int(split_ratio[0] / sum(split_ratio) * dataset_len)
data_points_train = shuffled_data_points[:no_train_points]

# Final Test Set
no_test_points = int(split_ratio[1] / sum(split_ratio) * dataset_len)
data_points_test = shuffled_data_points[no_train_points: no_train_points + no_test_points]
print("Anzahl Daten-Punkte:\n---------------------------\n ")
print("Training, Validation: ", no_train_points, " von ", dataset_len, "(", no_train_points / dataset_len, "%)")
print("Finales Testen: ", no_test_points, " von ", dataset_len, "(", no_test_points / dataset_len, "%)")

# Loop trough data points
for data_points, target_dir in zip([data_points_test, data_points_train],
                                   [target_dir_test, target_dir_train]):
    for data_point in data_points:
        # adjust path
        img_name = os.path.split(data_point[0])[1]
        file_name = img_name.replace("_camera_frame", "")
        file_name = file_name.replace("png", "json")
        file_source = os.path.join(source_dir, file_name)

        # Adjust Json File Content (Add Data, Sort transition points out, Change Label)
        file_raw = load_json(file_source)

        # Add Date, [Camera Position, Reactor Geometry, Substance Properties (Glycerol Share), Exp_nr, In Use]

        # Date
        exp_date = get_exp_date(data_point)
        file_added = add_key_content_value(file_raw, key_one="datetime", key_two="date",
                                           key_content="2022-" + exp_date)
        # Exp Number
        exp_nr = get_exp_nr(data_point)
        file_added = add_key_content_value(file_raw, key_one="experiment", key_two="number",
                                           key_content=exp_nr[-2:])

        # Reactor Geometry





        # Check for Transition Point, define target path
        gasflow_point = file_added["gas_flow_rate"]["data"]["value"]
        setpoint_bool = is_gfl_setpoint(gasflow_point)
        if setpoint_bool:
            proc_gfl = file_added["gas_flow_rate"]["data"]["value"]
            proc_rpm = file_added["stirrer_rotational_speed"]["data"]["value"]
            proc_exp = file_added["experiment"]["number"]["value"]
            proc_date = file_added["datetime"]["date"]["value"]

            exp_log["Datum"] = [date.replace('_', '-') for date in exp_log["Datum"]]
            label = get_flow_regime(proc_date, proc_exp, proc_gfl, proc_rpm, data_point, exp_log)
            if label == 3:
                file_path = os.path.join(target_dir_troub, file_name)
                file_path = target_dir_troub + '/' + file_name
            else:
                file_path = os.path.join(target_dir, file_name)
                file_path = target_dir + '/' + file_name
        else:
            label = 3
            file_path = os.path.join(target_dir_trans, file_name)
            file_path = target_dir_trans + '/' + file_name
            # Set Distintion between bad data and transition points (btw trans points still own wrong rpm values)

        # Add Label, according to experiment (Date,) Process Params (and Substance Properties)
        file_labeled = set_flow_regime(file_added, label)

        # Save Json to new dir
        with open(file_path, 'w') as fp:
            json.dump(file_labeled, fp, indent=4)
