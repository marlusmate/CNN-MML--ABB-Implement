import os
import tensorflow as tf
import matplotlib.pyplot as plt

from data_import.data_import import get_data_points_list, read_image, read_json, read_label, preprocess_image
from tensorflow.keras.models import load_model
from grad_cam import make_gradcam_heatmap, save_and_display_gradcam

# Adapted from https://keras.io/examples/vision/grad_cam/

# Model Name
model_name = 'Final-HybridFusion2-CnnOutput'

# Paths
source_dir = 'C:/Users/Markus Esser/PycharmProjects/MML-Beleg/data/example_pictures'
source_ref = 'C:/Users/Markus Esser/PycharmProjects/MML-Beleg/data/example_pictures-cropped'
model_path = f'../training/results/{model_name}/trained_model'

# Dataset parameters
param_list = ["stirrer_rotational_speed", "gas_flow_rate", "temperature", "fill_level"]
no_classes = 3
split_ratio = [0.7, 0.1, 0.2]
output_proc_shape = (len(param_list),)
output_img_shape = (128, 128, 1)

# Get Data Points
data_points = get_data_points_list(source_dir)

# Load Model
model = load_model(model_path)
model.summary()


# Define Final Cnn-layer-name
last_conv_layer_name = "conv3MMMLP1"

for data_point in data_points[:2]:
    image_path = data_point[0]
    file_path = data_point[1]
    image_name = os.path.split(image_path)[1]

    img = read_image(image_path)
    file = read_json(file_path)
    label = read_label(file_path, no_classes)

    img_preprocessed = preprocess_image(image_file=img, name_file=image_path, output_image_shape=output_img_shape)
    img_x = tf.convert_to_tensor([img_preprocessed])
    file_x = tf.convert_to_tensor([file])

    prediction = model.predict((img_x, file_x))
    # prediction = [Final-Prediction, EarlyFusion-Prediction, ImageFeature-Prediction, Param-Prediction]

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap((img_x, file_x), model, last_conv_layer_name)
    plt.matshow(heatmap)

    # Create Superimposed Visualization
    img_ref = read_image(source_ref + '/' + image_name)
    img_superimposed = save_and_display_gradcam(img_ref, heatmap)
    plt.imshow(img_superimposed)

plt.show()


