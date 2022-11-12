import os
import numpy as np
from keras.optimizers import Adam
from datasets.DF.utility import load_data_df_binary
from tensorflow.python.keras.utils import np_utils
from TheromEncoder import StandardTheromEncoder as STE, CustomizedTheromEncoder as CTE
from common_util import batch_run, bit_to_int, create_saved_models_dir

np.random.seed(1337)  # for reproducibility

############################
# Parameters

modelDirectory = os.getcwd()
nb_epochs = 30
batch_size = 128
lr = 0.00005
batch_scale_factor = 8
decay = 0.0001
use_thermo_encoding = 'False'  # do not require therometer encoding
fisrt_layer_binary = False
dense_layer_quantized = False
name_prefix = "DF_larq_"

lr *= batch_scale_factor
batch_size *= batch_scale_factor

print('Learning rate is: %f' % lr)
print('Batch size is: %d' % batch_size)

create_saved_models_dir()

def pre_process(use_thermo_encoding):
    optimiser = Adam(learning_rate=lr, decay=decay)
    bit_to_int_test = True
    ############################
    # Data
    X_train, y_train, X_valid, y_valid, X_test, y_test = load_data_df_binary()

    if bit_to_int_test and (use_thermo_encoding == 'standard' or use_thermo_encoding == 'customized'):
        X_train = bit_to_int(X_train, 8)
        X_test = bit_to_int(X_test, 8)
        X_valid = bit_to_int(X_valid, 8)

    if use_thermo_encoding == 'standard':
        temp_data = np.reshape(np.concatenate((X_train, X_valid, X_test)), (-1))
        te = STE()
        te.fit(temp_data)
        channel_num = 8
    elif use_thermo_encoding == 'customized':
        temp_data = np.reshape(np.concatenate((X_train, X_valid, X_test)), (-1))
        te = CTE(True)
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

    print("X_test")
    print(X_test)

    return X_train, y_train_cat, X_test, y_test_cat, X_valid, y_valid_cat, input_shape, nb_classes, \
           batch_size, nb_epochs, optimiser


# batch test run
batch_run(pre_process, name_prefix, basic_combs=True, test_first_layer_encoding=False)
