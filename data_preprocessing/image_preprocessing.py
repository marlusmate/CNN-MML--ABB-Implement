# Import
import os
import random
from tensorflow.keras.utils import save_img
from data_import.data_import import get_data_points_list, read_image, preprocess_image, read_json

"""
This Scripts contains all preprocessing Steps of the Images Data before Training the Model.
In an ideal Situation this has to be executed only once.
Returns preprocessed Image saved in two new target dirs (Training,Eval & Final Testing).
Random SEED und split RATIO must be the same as in scrip "param_preprocessing"
"""

# Directories
source_dir = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes_me/02_data'
target_dir_test = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes_me/04_test_data'
target_dir_train = '/mnt/0A60B2CB60B2BD2F/Datasets/bioreactor_flow_regimes_me/'

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
print("Training, Validation: ", no_train_points, " von ", dataset_len, "(", no_train_points/dataset_len, "%)")
print("Finales Testen: ", no_test_points, " von ", dataset_len, "(", no_test_points/dataset_len, "%)")

# Loop trough data points
for data_points, target_dir in zip([data_points_test, data_points_train],
                                   [target_dir_test, target_dir_train]):
    for data_point in data_points:
        # adjust path
        img_name = data_point[0].split('/')[-1]
        img_source = os.path.join(source_dir, img_name)
        img_path = os.path.join(target_dir, img_name)

        # Preprocess Image
        image_raw = read_image(img_source)
        if image_raw is None:
            continue
        img_preprocessed = preprocess_image(image_raw, img_source)

        # Save Preprocessed Image
        save_img(path=img_path, x=img_preprocessed)
        print("Preprocessed and Saved Image ", img_name, " to ", img_path)
