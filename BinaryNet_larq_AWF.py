import os
import numpy as np

import larq as lq
import tensorflow.python.ops.variables
from tensorflow.keras.optimizers import Adam
from AWF.utility import LoadDataNoDefCW

from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.keras.layers import BatchNormalization
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint

from NetworkParameters import NetworkParameters
from tensorflow.python.keras.layers import Activation
import tensorflow as tf
import matplotlib.pyplot as plt

from yiyan_util import get_model, calculate_uncertainty, batch_run
np.random.seed(1337)  # for reproducibility

############################
# Parameters

modelDirectory = os.getcwd()

parameters = NetworkParameters(modelDirectory)
parameters.nb_epochs = 30
parameters.batch_size = 128
parameters.lr = 0.00005
parameters.batch_scale_factor = 8
parameters.decay = 0.0001
use_thermo_encoding='False' # do not require therometer encoding
fisrt_layer_binary=True
dense_layer_quantized=False
name_prefix = "AWF_larq_"


parameters.binarisation_type = 'XNORNet'

parameters.lr *= parameters.batch_scale_factor
parameters.batch_size *= parameters.batch_scale_factor

print('Learning rate is: %f' % parameters.lr)
print('Batch size is: %d' % parameters.batch_size)


def pre_process(use_thermo_encoding):
    optimiser = Adam(learning_rate=parameters.lr, decay=parameters.decay)

    ############################
    # Data
    X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataNoDefCW()

    X_train = X_train.reshape(X_train.shape[0], 50, -1)
    X_valid = X_valid.reshape(X_valid.shape[0], 50, -1)
    X_test = X_test.reshape(X_test.shape[0], 50, -1)

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    y_valid = np.squeeze(y_valid)

    if len(X_train.shape) < 4:
        X_train = np.expand_dims(X_train, -1)
        X_test = np.expand_dims(X_test, -1)
        X_valid = np.expand_dims(X_valid, -1)

    input_shape = X_train.shape[1:]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_valid = X_valid.astype('float32')
    y_train = y_train.astype('int32')
    y_test = y_test.astype('int32')
    y_valid = y_valid.astype('int32')

    nb_classes = int(y_train.max() + 1)

    y_test_cat = np_utils.to_categorical(y_test, nb_classes + 1)
    y_train_cat = np_utils.to_categorical(y_train, nb_classes + 1)
    y_valid_cat = np_utils.to_categorical(y_valid, nb_classes + 1)

    return X_train, y_train_cat, X_test, y_test_cat, X_valid, y_valid_cat, \
                                        input_shape, nb_classes, parameters, optimiser

############################
# test run
# X_train, y_train_cat, X_test, y_test_cat, X_valid, y_valid_cat, input_shape, nb_classes, parameters, optimiser =\
# pre_process(use_thermo_encoding)

# calculate_uncertainty(
#                     X_train, y_train_cat, X_test, y_test_cat, X_valid, y_valid_cat,
#                     input_shape, nb_classes, parameters, optimiser, 
#                     fisrt_layer_binary=fisrt_layer_binary, dense_layer_quantized=dense_layer_quantized,
#                     name=name_prefix + str(use_thermo_encoding) + "_" + str(fisrt_layer_binary) + "_" + str(dense_layer_quantized),
#                     round_num=1)


# batch test run
batch_run(pre_process,name_prefix, basic_combs=True, test_first_layer_encoding=False, 
    customize_configs=[])

# show_info(model, "images/AWF/", 'quant_conv2d', print_result=False)
# show_info(model, "images/AWF/", 'quant_conv2d_1', binarize_weights=True)
# show_info(model, "images/AWF/", 'quant_conv2d_2', binarize_weights=True)
# show_info(model, "images/AWF/", 'flatten', result_type=('output'), print_result=False)
# show_info(model, "images/AWF/", 'quant_dense')
