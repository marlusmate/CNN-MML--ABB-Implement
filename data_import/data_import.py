import os
import tensorflow as tf
import json


def get_data_points_list(source_dir, number_points='all'):
    """
    This function iterative over the given folder and look for all data points (image file and metadata file) and
    returns the list with full file names to those files.
    :param source_dir: directories where data points are to search
    :param number_points: specifies whether all data points (use tag 'all') must be or only limited number. The later
    helps to test the training pipeline with a small amount of data.
    :return: List of full file names including full paths to files to each data points. MUST be SHUFFLED
    """
    image_file = []
    metadata_file = []
    for file in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, file)) and file.endswith('.png'):
            filename_image = os.path.join(source_dir, file)
            filename = os.path.splitext(file)[0][:-13]
            filename_metadata = os.path.join(source_dir, filename + '.json')
            if os.path.isfile(filename_metadata):
                image_file.append(filename_image)
                metadata_file.append(filename_metadata)
    if number_points == 'all':
        return list(zip(image_file, metadata_file))
    else:
        return list(zip(image_file, metadata_file))[0:number_points]


def load_json(data_point):
    """
    This loads a json file as a dictionary every time it is called.
    :param data_point: data point (see function get_data_points_list())
    :return: json file as dict
    """
    if len(data_point[1]) == 1:
        json_file = data_point
    else:
        json_file = data_point[1]
    with open(json_file) as f:
        json_content = json.load(f)

    return json_content


def read_json(data_point, param_list=None):
    """
    This returns the process data of a json file (from a dict) as a list (See function load_json()).
    :param data_point: data point (see function get_data_points())
    :param param_list: list of parameters to be extracted
    :return: list of parameter values, in the same order as param_list (no labels)
    """
    json_content = load_json(data_point)
    if param_list is None:
        param_list = ["stirrer_rotational_speed", "gas_flow_rate", "temperature", "fill_level"]
    if not all(["value" in json_content[param]["data"] for param in param_list]):
        return None

    params = [json_content[param]["data"]["value"] for param in param_list]

    return params


def read_image(file):
    """
    This function read an image and returns it as a tensorflow tensor.
    :param file: Full path to the image file
    :return: Image as a tensor or None if the image cannot be read.
    """
    try:
        image_file = tf.io.read_file(file)
        image_data = tf.image.decode_image(image_file)
        return image_data
    except Exception:
        print(f'Image {file} could not be read')
        return None


def read_label(file, no_classes):
    """
    This function reads the label from the metadata file and encodes it according to the one-hot scheme.
    :param file: Full path to the metadata file
    :param no_classes: Total number of all possible classes to allow one-hot encoding
    :return: One-hot encoded label as a tensor
    """
    one_hot_encoder = tf.one_hot(range(no_classes), no_classes)
    with open(file) as f:
        json_content = json.load(f)
        label_int = json_content["flow_regime"]["data"]["value"]
        label = one_hot_encoder[label_int]
        return label


def preprocess_image(image_file, name_file, output_image_shape=(128, 128, 1), cropping=True, crop_box=None):
    """
    This functions contains all preprocess steps of the image data. Including: Grayscale Conversion, Cropping,
    Resize and Normalization.
    :param image_file: Tensor or Array of Image
    :param name_file:  string of Name of Image File (contains experiment date for cropping purpose)
    :param output_image_shape: List or Tuple with Target Size for preprocessed Images
    :param cropping: Boolean, for deactivation of cropping step
    :param crop_box: List or Tuple with Points /Coordinates for cropping
    (see function tf.image.crop_to_bounding_box() doc)
    :return: Image as a tensor
    """
    # Convert to Gray Scale
    if image_file.shape[2] != 1:
        image_grayscaled = tf.image.rgb_to_grayscale(image_file)
    else:
        image_grayscaled = image_file
    print("Shape image_grayscaled: ", image_grayscaled.shape, "; Type: ", type(image_grayscaled))

    # Crop Box, Size
    if cropping is True:
        if crop_box is None:
            crop_matrix = {"02-08": [400, 0, 1165, 2447], "02-09": [225, 0, 1481, 2447], "02-22": [440, 0, 1162, 2447],
                           "02-25": [455, 0, 1149, 2447], "03-09": [370, 0, 1310, 2447]}
            exp_date = [date for date in crop_matrix.keys() if date in name_file]
            crop_points = crop_matrix[exp_date[0]]
        else:
            crop_points = [crop_box[0], 0, crop_box[1], image_grayscaled.shape[0]]

        # Crop and
        image_cropped = tf.image.crop_to_bounding_box(image_grayscaled, crop_points[0], crop_points[1], crop_points[2],
                                                      crop_points[3])
    else:
        image_cropped = image_grayscaled

    # Resize
    final_image_size = list(output_image_shape)[0:2]
    image_resized = tf.image.resize(image_cropped, final_image_size, method='bicubic')

    # Normalize Image
    image_normed = tf.math.divide(image_resized,
                                  tf.constant(255, shape=tf.shape(image_resized).numpy(), dtype="float32"))

    return image_normed


def data_generator(list_data_points, repeats, no_classes, param_list=None):
    """
    Dataset Generator for Model In- & Output. Preprocessing is NOT build in (see function preprocess_image()),
    therefore should be done bevor training, evaluation.
    :param list_data_points: (SHUFFLED) List of tuples of data_points (image_path, json_path).
    :param repeats: Number of Epochs
    :param no_classes: Number of classes
    :param param_list: List of strings with to be extracted Process Parameters
    :return: Yields tuple, must match defined output_signature
    """
    for repeat in range(repeats):
        for data_point in list_data_points:
            image_file = data_point[0]
            label_file = data_point[1]

            image_preprocessed = read_image(image_file)
            params_preprocessed = read_json(label_file, param_list=param_list)  # actually not preprocessed, just loaded

            if image_preprocessed is None:
                print("Parameter File ", label_file, "missing, skipping data point")
                continue
            if params_preprocessed is None:
                print("Image File ", image_file, "missing, skipping data point")
                continue

            label_data = read_label(label_file, no_classes)
            proc_data = tf.convert_to_tensor(params_preprocessed)

            yield (image_preprocessed, proc_data), (label_data, label_data, label_data, label_data)
