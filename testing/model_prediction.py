import tensorflow as tf
import json
import random
import pickle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from data_import.data_import import data_generator


# Model Name
model_type = "Final-HybridFusion2-CnnOutput"

# Paths
model_path = '../training/results'
model_name = f'/results/{model_type}/trained_model'
model_checkpoint_path = f'../training/results/{model_type}' + '/checkpoints/checkpoint-0010.ckpt'
data_list = '../data/data-points-test.pickle'

# Dataset Parameters
param_list = ["stirrer_rotational_speed", "gas_flow_rate", "temperature", "fill_level"]
output_proc_shape = (len(param_list),)
batch_size = 15
output_img_shape = (128, 128, 1)
no_classes = 3
no_epochs = 1

# Load Data Points for Testing
with open(data_list, 'rb') as file:
    # Call load method to deserialze
    data_points = pickle.load(file)
shuffled_data_points = random.sample(data_points, len(data_points))

# Get Distribution
lb_test = []
for data_point in data_points:
    with open(data_point[1]) as f:
        json_content = json.load(f)
        lb_test.append(json_content["flow_regime"]["data"]["value"])
print("Test Instanzen:\n-----------------------")
print("Klasse 0: ", lb_test.count(0), "\nKlasse 1: ", lb_test.count(1), "\nKlasse 2: ", lb_test.count(2))

# Data Generator
output_signature = ((tf.TensorSpec(shape=output_img_shape, dtype=tf.float32),
                    tf.TensorSpec(shape=output_proc_shape, dtype=tf.float32)),
                     (tf.TensorSpec(shape=(no_classes), dtype=tf.bool),
                      tf.TensorSpec(shape=(no_classes), dtype=tf.bool),
                    tf.TensorSpec(shape=(no_classes), dtype=tf.bool),
                    tf.TensorSpec(shape=(no_classes), dtype=tf.bool)))

data_gen_test = data_generator(data_points, repeats=no_epochs, no_classes=3, param_list=param_list)
dataset_test = tf.data.Dataset.from_generator(lambda: data_gen_test, output_signature=output_signature)
dataset_test_batched = dataset_test.batch(batch_size)

# Build Model
opt = Adam()
model = load_model(f'../training/results/{model_type}/trained_model')
# model.compile(loss=["categorical_crossentropy","categorical_crossentropy","categorical_crossentropy","categorical_crossentropy"], loss_weights=[0.7, 0.1, 0.1, 0.1], optimizer=opt, metrics=["accuracy"])
model.summary()

# Save Predictions
pred = model.predict(dataset_test_batched, batch_size=batch_size)
with open(f"Final-y_pred_{model_type}.pickle", 'wb') as f:
    pickle.dump(pred, f)
with open(f"Final-y_true_{model_type}.pickle", 'wb') as f:
    pickle.dump(lb_test, f)

print("Vorhersagen(", len(pred), ") abgespeichert - y_pred.json")

