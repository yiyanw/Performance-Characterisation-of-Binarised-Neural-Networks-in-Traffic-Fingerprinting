import os
import numpy as np
import larq as lq
from IoT.utility import LoadDataIot, LoadDataIotNumeric, LoadDataIotNumeric_v2, LoadDataIotNumeric_v3, LoadDataIotNumeric_v4, LoadDataIotBinary

from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Activation

from NetworkParameters import NetworkParameters
import tensorflow as tf
import matplotlib.pyplot as plt
from thermoencoder import ThermoEncoder  # standard therometer encoding 
from TheromEncoder import StandardTheromEncoder as STE, CustomizedTheromEncoder as CTE
from yiyan_util import get_model, calculate_uncertainty, batch_run, bit_to_int
np.random.seed(1337)  # for reproducibility

############################
# Parameters

modelDirectory = os.getcwd()

parameters = NetworkParameters(modelDirectory)
parameters.nb_epochs = 100
parameters.batch_size = 128
parameters.lr = 0.00005
parameters.batch_scale_factor = 8
parameters.decay = 0.0001
use_thermo_encoding='False' # do not require therometer encoding
fisrt_layer_binary=True
dense_layer_quantized=False
name_prefix = "IOT_larq_"
#
parameters.binarisation_type = 'XNORNet'

parameters.lr *= parameters.batch_scale_factor
parameters.batch_size *= parameters.batch_scale_factor

print('Learning rate is: %f' % parameters.lr)
print('Batch size is: %d' % parameters.batch_size)


############################
# Data

def pre_process(use_thermo_encoding):
    optimiser = Adam(learning_rate=parameters.lr, decay=parameters.decay)
    # bit_to_int_test = False
    # X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataIot()
    # X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataIotNumeric_v3()
    # X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataIotBinary()
    X_train, y_train, X_valid, y_valid, X_test, y_test = LoadDataIotNumeric_v4()
    if use_thermo_encoding == 'standard':
        # temp_data = np.concatenate((X_train, X_valid, X_test))
        # max_arr = [ np.max(temp_data) for i in range(len(temp_data[0])) ]
        # te = ThermoEncoder()
        # te.fit(np.concatenate((temp_data, np.array([max_arr]))))
        temp_data = np.reshape(np.concatenate((X_train, X_valid, X_test)),(-1))
        te = STE(encoding_len=16)
        te.fit(temp_data)
        channel_num = 8
    elif use_thermo_encoding == 'customized':
        temp_data = np.reshape(np.concatenate((X_train, X_valid, X_test)),(-1))
        te = CTE(debug_mode=True, encoding_len=16)
        te.fit(temp_data)
        channel_num = len(te.thresholds)

    if use_thermo_encoding == 'standard' or use_thermo_encoding == 'customized':
        X_train = te.transform(X_train)
        X_train = np.reshape(X_train, (len(X_train), -1, channel_num))
        X_train = np.transpose(X_train, [0, 2, 1])
        X_train = np.reshape(X_train, [len(X_train), -1, channel_num])

        X_valid = te.transform(X_valid)
        X_valid = np.reshape(X_valid, (len(X_valid), -1, channel_num))
        X_valid = np.transpose(X_valid, [0, 2, 1])
        X_valid = np.reshape(X_valid, [len(X_valid), -1, channel_num])

        X_test = te.transform(X_test)
        X_test = np.reshape(X_test, (len(X_test), -1, channel_num))
        X_test = np.transpose(X_test, [0, 2, 1])
        X_test = np.reshape(X_test, [len(X_test), -1, channel_num])
    # thermo encoding end

    else:
        X_train = X_train.reshape(X_train.shape[0], -1, 1)
        X_valid = X_valid.reshape(X_valid.shape[0], -1, 1)
        X_test = X_test.reshape(X_test.shape[0], -1, 1)


    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)
    y_valid = np.squeeze(y_valid)

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


# batch test run
batch_run(pre_process,name_prefix, basic_combs=True, test_first_layer_encoding=True, 
    customize_configs=[['customized', True, False], ['standard', True, False]])