import tensorflow as tf
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.layers import Input, Flatten, Dense, Conv1D, Activation, add, AveragePooling1D, Dropout, Permute, concatenate, GRU
from tensorflow.python.layers.normalization import BatchNormalization
from tensorflow.python.keras import Model
from tensorflow.python.keras.optimizer_v2 import rmsprop

from tensorflow.python.keras.utils.vis_utils import plot_model

# GitHub - keunwoochoi/kapre: kapre: Keras Audio Preprocessors
# https://kapre.readthedocs.io/en/latest/
from kapre.time_frequency import Spectrogram
from kapre.utils import Normalization2D

import numpy as np

np.random.seed(3)


def spectrogram_layer(input_x):
    l2_lambda = .001
    n_dft = 64
    n_hop = 64

    x = Permute((2, 1))(input_x)
    x = Spectrogram(n_dft=n_dft, n_hop=n_hop, image_data_format='channels_last', return_decibel_spectrogram=True)(x)
    x = Normalization2D(str_axis='batch')(x)

    x = Flatten()(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)

    return x


def one_channel_resnet(input_shape, num_filters=64, num_res_blocks=2, cnn_per_res=3, kernel_sizes=(8, 5, 3), max_filters=128, pool_size=3, pool_stride_size=2):
    my_input = Input(shape=input_shape)

    for i in np.arange(num_res_blocks):
        if i == 0:
            block_input = my_input
            x = BatchNormalization()(block_input)
        else:
            block_input = x

        for j in np.arange(cnn_per_res):
            x = Conv1D(num_filters, kernel_sizes[j], padding='same')(x)
            x = BatchNormalization()(x)
            if j < cnn_per_res - 1:
                x = Activation('relu')(x)

        is_expand_channels = not (input_shape[0] == num_filters)

        if is_expand_channels:
            res_conn = Conv1D(num_filters, 1, padding='same')(block_input)
            res_conn = BatchNormalization()(res_conn)
        else:
            res_conn = BatchNormalization()(block_input)

        x = add([res_conn, x])
        x = Activation('relu')(x)

        if i < 5:
            x = AveragePooling1D(pool_size=pool_size, strides=pool_stride_size)(x)

        num_filters = 2 * num_filters
        if max_filters < num_filters:
            num_filters = max_filters

    return my_input, x


def build_bp_model(input_shape, num_channels, print_model_summary=False, plot_model_architecture=False):
    inputs = []
    l2_lambda = .001
    channel_outputs = []
    num_filters = 32
    for i in np.arange(num_channels):
        channel_resnet_input, channel_resnet_out = one_channel_resnet(
            input_shape, num_filters=num_filters, num_res_blocks=4, cnn_per_res=3, kernel_sizes=[8, 5, 5, 3],
            max_filters=64, pool_size=2, pool_stride_size=1)
        channel_outputs.append(channel_resnet_out)
        inputs.append(channel_resnet_input)

    spectral_outputs = []
    for x in inputs:
        spectro_x = spectrogram_layer(x)
        spectral_outputs.append(spectro_x)

    x = concatenate(channel_outputs, axis=-1)
    x = BatchNormalization()(x)
    x = GRU(65)(x)
    x = BatchNormalization()(x)

    s = concatenate(spectral_outputs, axis=-1)
    s = BatchNormalization()(s)

    x = concatenate([s, x])

    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(0.25)(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(0.25)(x)

    output = Dense(2, activation="relu")(x)
    model = Model(inputs=inputs, outputs=output, name="RunKicker_BP_model")
    optimizer = rmsprop.RMSprop(learning_rate=.0001, decay=.0001)
    loss_fn = tf.keras.losses.MeanSquaredError()
    mse_metric = tf.keras.metrics.MeanSquaredError()
    # acc_metric = tf.keras.metrics.Accuracy()

    model.compile(optimizer=optimizer, loss=loss_fn, metrics=[mse_metric])

    if print_model_summary:
        print(model.summary())

    if plot_model_architecture:
        plot_model(model=model, to_file='lstm_model.png', show_shapes=True, show_layer_names=True)

    return model
