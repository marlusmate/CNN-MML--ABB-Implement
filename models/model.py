from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Dropout
from tensorflow.keras import Model
from tensorflow import float32


class mmlmodel:  # HybridFusion2
    @staticmethod
    def build_image_extraction(input_shape, classes):
        # Input
        input_image = Input(shape=input_shape, name="InputImageMMMLP1")

        # First complex of convolutional layers (conv, activation, pooling)
        x = Conv2D(filters=20, kernel_size=(7, 7), padding="same", name="conv1MMMLP1")(input_image)
        x = Activation("relu", name="relu1ImgMMMLP1")(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool1MMMLP1")(x)

        # Second complex of convolutional layers (conv, activation, pooling)
        x = Conv2D(filters=15, kernel_size=(5, 5), padding="same", name="conv2MMMLP1")(x)
        x = Activation("relu", name="relu2ImgMMMLP1")(x)
        second_last_layer = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool2MMMLP1")(x)

        # Third complex of convolutional layers (conv, activation, pooling)
        last_conv = Conv2D(filters=15, kernel_size=(5, 5), padding="same", name="conv3MMMLP1")(x)
        x = Activation("relu", name="relu3ImgMMMLP1")(last_conv)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool3MMMLP1")(x)

        # First complex of dense layer (flatten, dense)
        x = Flatten(name="flattenIamgeMMMLP1")(x)
        erl_features = Dense(20, name="dense1ImgMMMLP1")(x)

        # Higher Level Features
        output_erl_feauters_cnn = Activation("relu", name="relu4_lt_cnn")(erl_features)
        """
        # Possible MLP
        x = Dense(11, name="dense2ImgMMMLP1")(x)
        x = Activation("relu", name="relu5_lt_cnn")(x)
        lt_features = Dense(15, name="dense3ImgMMMLP1")(x)

        # Output Layer (activation)
        output_erl_features = Activation("relu", name="relu1_erl_cnn_MMMlP1")(erl_features)
        output_lt_features = Activation("relu", name="relu6_lt_cnn")(lt_features)
        """
        # Aux Output
        #x = Dense(11,activation="relu", name="denseAuxImgMMMLP1")(output_erl_feauters_cnn)
        #output_aux_label = Dense(classes, activation="sigmoid", name="relu_out_lt_cnn")(x)

        return Model(inputs=input_image, outputs=[erl_features,output_erl_feauters_cnn, second_last_layer],
                     name="ModelImageExtract")

    @staticmethod
    def build_params_extraction(input_shape, classes):
        # Input
        input_params = Input(shape=input_shape, name="InputProcessMMMLP1")

        # First complex of fully connected Dense Layers (dense, activation)
        x = Dense(20, name="dense1ParamMMMLP1")(input_params)
        x = Activation("relu", name="relu1ParamMMMlP1")(x)

        # Second complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="dense2ParamMMMLP1")(x)
        x = Activation("relu", name="relu2ParamMMlP1")(x)

        # Third Complex of fully connected Dense Layers
        erl_features_mlp = Dense(15, name="erl_features_mlpMMMLP1")(x)
        output_erl_features_mlp = Activation("relu", name="erl_output_mlp")(erl_features_mlp)
        """
        # Second complex of fully connected Dense Layers (dense, activation)
        x = Dense(7, name="dense4ParamMMMLP1")(output_erl_features_mlp)
        x = Activation("relu", name="relu4ParamMMlP1")(x)

        # Second complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="dense5ParamMMMLP1")(x)
        output_lt_features_mlp = Activation("relu", name="relu5ParamMMlP1")(x)
        """
        # Aux Label Output
        #output_aux_label = Dense(classes, activation="sigmoid", name="aux_label_mlp")(output_erl_features_mlp)

        return Model(inputs=input_params, outputs=[erl_features_mlp, output_erl_features_mlp],
                     name="ModelParamExtract")

    @staticmethod
    def build(input_shape_image, input_shape_params, no_classes):
        # Input
        image_in = Input(shape=input_shape_image, name="FrontInputImageMMMLP1")
        params_in = Input(shape=input_shape_params, name="FrontInputParamsMMMLP1")

        # Build Feature extraction blocks (Image, Process Parameters)
        cnn1 = mmlmodel.build_image_extraction(input_shape_image, no_classes)
        param1 = mmlmodel.build_params_extraction(input_shape_params, no_classes)

        # Output Extract-Models
        image_features = cnn1(image_in)
        param_features = param1(params_in)

        # >EARLY Fusion Features
        x = image_features[2]
        # Third complex of convolutional layers (conv, activation, pooling)
        last_conv = Conv2D(filters=15, kernel_size=(5, 5), padding="same", name="conv3MMMLP1")(x)
        x = Activation("relu", name="relu3ImgMMMLP1")(last_conv)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="maxpool3MMMLP1")(x)

        # First complex of dense layer (flatten, dense)
        x = Flatten(name="flattenIamgeMMMLP1")(x)
        image_erl_features = Dense(20, name="dense1ImgMMMLP1")(x)
        # Higher Level Features
        output_erl_feauters_cnn = Activation("relu", name="relu4_lt_cnn")(image_erl_features)

        param_erl_features = param_features[0]

        # First complex of fully connected Layers (concatenate)
        merged_erl = Concatenate()([image_erl_features, param_erl_features])
        #merged_erl_activated = Activation("relu")(merged_erl)

        # Frist complex of fully connected Layers (dense, activation
        x = Dense(15, name="dense1MergedMMMLP1")(merged_erl)
        x = Activation("relu", name="relu1MergedMMMlP1")(x)
        x = Dropout(rate=0.2)(x)

        # Second complex of fully connected Dense Layers (dense, activation)
        erl_features_mml  = Dense(15, name="dense2MergedMMMLP1")(x)
        output_erl_features_mml = Activation("relu", name="relu2MergedMMlP1")(erl_features_mml)

        # >LATE Fusion Features
        image_lt_features = output_erl_feauters_cnn
        param_lt_features = Activation("relu")(param_features[0])
        # First complex of fully connected Layers (concatenate)
        merged_lt = Concatenate()([image_lt_features, output_erl_features_mml, param_lt_features])

        # Frist complex of fully connected Layers (dense, activation
        x = Dense(15, name="dense3MergedMMMLP1")(merged_lt)
        x = Activation("relu", name="relu3MergedMMMlP1")(x)
        x = Dropout(rate=0.2)(x)

        # Second complex of fully connected Dense Layers (dense, activation)
        x = Dense(15, name="dense4MergedMMMLP1")(x)
        x = Activation("relu", name="relu4MergedMMlP1")(x)
        #x = Dropout(rate=0.2)(x)

        # Output
        output_final = Dense(no_classes, activation="sigmoid", name="Output_lt_label_dMMMLP1")(x)
        output_aux_erl = Dense(no_classes, activation="sigmoid", name="Output_erl_label_dMMMLP1")(output_erl_features_mml)
        output_aux_img = Dense(no_classes, activation="sigmoid", name="aux_label_cnn")(output_erl_feauters_cnn)
        output_aux_param = Dense(no_classes, activation="sigmoid", name="aux_label_mlp")(param_features[1])

        return Model(inputs=(image_in, params_in),
                     outputs=[output_final, output_aux_erl, output_aux_img, output_aux_param], name="ModelMultimodal")

