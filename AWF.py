import os
import numpy as np
# from tensorflow.python.keras.optimizers import Adam
from keras.optimizers import Adam

from datasets.AWF.utility import load_data_awf_binary
from tensorflow.python.keras.utils import np_utils
from common_util import batch_run, create_saved_models_dir

np.random.seed(1337)  # for reproducibility

############################
# Parameters

modelDirectory = os.getcwd()

nb_epochs = 30
batch_size = 128
lr = 0.00005
batch_scale_factor = 8
decay = 0.0001
use_thermo_encoding='False' # do not require therometer encoding
fisrt_layer_binary=True
dense_layer_quantized=False
name_prefix = "AWF_larq_"

lr *= batch_scale_factor
batch_size *= batch_scale_factor

print('Learning rate is: %f' % lr)
print('Batch size is: %d' % batch_size)

create_saved_models_dir()

def pre_process(use_thermo_encoding):
    optimiser = Adam(learning_rate=lr, decay=decay)

    ############################
    # Data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_awf_binary()

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

    return X_train, y_train_cat, X_test, y_test_cat, X_valid, y_valid_cat, input_shape, nb_classes, \
           batch_size, nb_epochs, optimiser


# batch test run
batch_run(pre_process,name_prefix, basic_combs=True, test_first_layer_encoding=False, 
    customize_configs=[])