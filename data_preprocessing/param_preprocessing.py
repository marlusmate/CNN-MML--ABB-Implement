# Import
import os
import random
import json
from tensorflow.keras.preprocessing.image import array_to_img
import pandas as pd
from data_import.data_import import get_exp_data_points_list, load_json, read_image, preprocess_image
from data_import.data_adjust import add_key_content_value, get_exp_date, is_gfl_setpoint, get_flow_regime, get_exp_nr \
    , set_flow_regime, get_gly_value, get_file_usage

"""
This Scripts contains all preprocessing Steps of the Process Data before Training the Model.
In an ideal Situation this has to be executed only once.
Returns preprocessed adjusted Metadata-File, saved in two new target dirs (Training,Eval & Final Testing).
Random SEED und split RATIO must be the same as in scrip "param_preprocessing"
"""

# Directories
source_dir = 'C:/Users/Markus Esser/mml-data/01_raw_data'
target_dir_test = 'C:/Users/Markus Esser/mml-data/example_pictures-test'
target_dir_train = 'C:/Users/Markus Esser/mml-data/example_pictures-train'
target_dir_trans = 'C:/Users/Markus Esser/mml-data/example_pictures-trans'
target_dir_troub = 'C:/Users/Markus Esser/mml-data/example_pictures-troub'
exp_log_dir = "../data_import/experiment_log.csv"

# Set random Seed for reproducibility
random.seed(73)

# Split Final Test Set
split_ratio = [0.9, 0.1]

# Get Experiment Log
exp_log = pd.read_csv(exp_log_dir, delimiter=";")

# Get data_points, shuffeld list
data_points = get_exp_data_points_list(source_dir)
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
        file_source = os.path.split(data_point[0])[0] +'/'+file_name

        # Adjust Json File Content (Add Data, Sort transition points out, Change Label)
        file_raw = load_json(file_source)

        # Get Date and Experiment Number from dir name
        exp_date = get_exp_date(data_point)
        exp_nr = get_exp_nr(data_point)

        # Add Date, [Camera Position, Reactor Geometry, Substance Properties (Glycerol Share), Exp_nr, In Use]

        # Date
        file_added = add_key_content_value(file_raw, key_one="datetime", key_two="date",
                                           key_content="2022-" + exp_date)
        # Exp Number
        file_added = add_key_content_value(file_added, key_one="experiment", key_two="number",
                                           key_content=exp_nr[-2:])
        # Reactor Geometry

        # Substance Properties
        gly_val = get_gly_value(file_added)
        file_added = add_key_content_value(file_added, key_one="Substance", key_two="Glycerol",
                                           key_content=float(gly_val))

        # In-Use
        in_use = get_file_usage(file_added)
        file_added = add_key_content_value(file_added, key_one="in-usage", key_two="File",
                                           key_content=float(in_use))

        # Check for Transition Point, define target path
        if in_use == 3:
            # file_path = os.path.join(target_dir_troub, file_name)
            file_path = target_dir_troub + '/' + file_name
            label = 3
        else:
            gasflow_point = file_added["gas_flow_rate"]["data"]["value"]
        setpoint_bool = is_gfl_setpoint(gasflow_point)
        if setpoint_bool and in_use == 1:
            proc_gfl = file_added["gas_flow_rate"]["data"]["value"]
            proc_rpm = file_added["stirrer_rotational_speed"]["data"]["value"]
            proc_exp = file_added["experiment"]["number"]["value"]
            proc_date = file_added["datetime"]["date"]["value"]

            exp_log["Datum"] = [date.replace('_', '-') for date in exp_log["Datum"]]
            label = get_flow_regime(proc_date, proc_exp, proc_gfl, proc_rpm, data_point, exp_log)
            if label == 3:
                #file_path = os.path.join(target_dir_troub, file_name)
                file_path = target_dir_troub + '/' + file_name
            else:
                #file_path = os.path.join(target_dir, file_name)
                file_path = target_dir + '/' + file_name
        elif in_use==4:
            if gasflow_point < 1:
                file_path = target_dir_trans + '/' + file_name
            label = 4

            # Set Distintion between bad data and transition points

        # Add Label, according to experiment (Date,) Process Params (and Substance Properties) (from experiment_log.csv)
        file_labeled = set_flow_regime(file_added, label)

        # Preprocess Image
        image_raw = read_image(data_point[0])
        if image_raw is None:
            continue
        img_preprocessed = array_to_img(preprocess_image(image_raw, data_point[0]))
        image_path = os.path.split(file_path)[0] + '/' + img_name

        # Save Json to new dir
        with open(file_path, 'w') as fp:
            json.dump(file_labeled, fp, indent=4)

        # Save Image to new dir
        img_preprocessed.save(image_path)

