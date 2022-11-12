import _pickle as pickle
import os

import numpy as np

CUR_PATH = os.path.split(os.path.realpath(__file__))[0]
print(CUR_PATH)


# Load data for non-defended dataset for CW setting
def load_data_df_binary():
    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = os.path.join(CUR_PATH, "dataset")

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(os.path.join(dataset_dir, 'X_train_NoDef.pkl').replace('\\', '/'), 'rb') as handle:
        X_train = np.array(pickle.load(handle, encoding='latin1'))
    with open(os.path.join(dataset_dir, 'y_train_NoDef.pkl').replace('\\', '/'), 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(os.path.join(dataset_dir, 'X_valid_NoDef.pkl').replace('\\', '/'), 'rb') as handle:
        X_valid = np.array(pickle.load(handle, encoding='latin1'))
    with open(os.path.join(dataset_dir, 'y_valid_NoDef.pkl').replace('\\', '/'), 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(os.path.join(dataset_dir, 'X_test_NoDef.pkl').replace('\\', '/'), 'rb') as handle:
        X_test = np.array(pickle.load(handle, encoding='latin1'))
    with open(os.path.join(dataset_dir, 'y_test_NoDef.pkl').replace('\\', '/'), 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test