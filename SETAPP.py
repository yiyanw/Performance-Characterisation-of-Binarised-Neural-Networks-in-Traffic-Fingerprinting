import os
import numpy as np

# from tensorflow.keras.optimizers import Adam
from keras.optimizers import Adam
from datasets.SETAPP.utility import LoadData
from tensorflow.python.keras.utils import np_utils
from TheromEncoder import StandardTheromEncoder as STE, CustomizedTheromEncoder as CTE
from common_util import batch_run

np.random.seed(1337)  # for reproducibility

############################
# Parameters

modelDirectory = os.getcwd()
nb_epochs = 1000
batch_size = 128
lr = 0.00005
batch_scale_factor = 8
decay = 0.0001
use_thermo_encoding = 'False'
fisrt_layer_binary = False
dense_layer_quantized = True
name_prefix = "SETAPP_larq_"
#
parameters.binarisation_type = 'XNORNet'

lr *= batch_scale_factor
batch_size *= batch_scale_factor

print('Learning rate is: %f' % lr)
print('Batch size is: %d' % batch_size)


def pre_process(use_thermo_encoding):
    optimiser = Adam(learning_rate=lr, decay=decay)

    X_train, y_train, X_valid, y_valid, X_test, y_test = LoadData()

    channel_num = 8
    te = None
    if use_thermo_encoding == 'standard':
        temp_data = np.reshape(np.concatenate((X_train, X_valid, X_test)), (-1))
        te = STE(encoding_len=8)
        te.fit(temp_data)
        channel_num = 8
    elif use_thermo_encoding == 'customized':
        temp_data = np.reshape(np.concatenate((X_train, X_valid, X_test)), (-1))
        te = CTE()
        te.fit(temp_data)
        channel_num = len(te.thresholds)

    if use_thermo_encoding == 'standard' or use_thermo_encoding == 'customized':
        # thermo encoding start
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

    return X_train, y_train_cat, X_test, y_test_cat, X_valid, y_valid_cat, input_shape, nb_classes, \
           batch_size, nb_epochs, optimiser


# batch test run
batch_run(pre_process, name_prefix, True, True, [])
