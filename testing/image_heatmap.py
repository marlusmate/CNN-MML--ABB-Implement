import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from data_import.data_import import get_data_points_list, read_image, read_json, read_label
from tensorflow.keras.models import load_model
from grad_cam import make_gradcam_heatmap

# Adapted from https://keras.io/examples/vision/grad_cam/

# Model Name
model_name = 'Final-HybridFusion2-CnnOutput'

# Paths
source_dir = 'C:/Users/Markus Esser/PycharmProjects/MML-Beleg/data/example_pictures'
source_ref = 'C:/Users/Markus Esser/PycharmProjects/MML-Beleg/data/example_pictures-cropped'
model_path = f'../training/results/{model_name}/trained_model'

# Get Data Points
data_points = get_data_points_list(source_dir)

# Load Model
opt = Adam()
model = load_model(model_path)
model.summary()

#model.compile(loss=["categorical_crossentropy","categorical_crossentropy","categorical_crossentropy","categorical_crossentropy"], loss_weights=[0.7, 0.1, 0.1, 0.1], metrics=["accuracy"])

# Define Final Cnn-layer-name
last_conv_layer_name = "momo"

for data_point in data_points:
    image_path = data_point[0]
    file_path = data_point[1]

    img = read_image(image_path)
    file = read_json(file_path)
    label = read_label((file_path))

    img_x = tf.convert_to_tensor([img])
    file_x = tf.convert_to_tensor([file])

    pred = model.predict((img_x, file_x))

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap((img_x, file_x), model, last_conv_layer_name)



