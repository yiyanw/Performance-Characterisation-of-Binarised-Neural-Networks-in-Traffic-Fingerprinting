import _pickle as pickle
import os
import numpy as np

CUR_PATH = os.path.split(os.path.realpath(__file__))[0]
print(CUR_PATH)

dataset_dir = os.path.join(CUR_PATH, "dataset")


# Load data for non-defended dataset for CW setting
def load_data_setapp_numeric():
    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data
    with open(os.path.join(dataset_dir, 'video_X_train.pkl').replace('\\', '/'), 'rb') as handle:
        X_train = np.array(pickle.load(handle, encoding='latin1'))
    with open(os.path.join(dataset_dir, 'video_y_train.pkl').replace('\\', '/'), 'rb') as handle:
        y_train = np.array(pickle.load(handle))

    # Load validation data
    with open(os.path.join(dataset_dir, 'video_X_valid.pkl').replace('\\', '/'), 'rb') as handle:
        X_valid = np.array(pickle.load(handle, encoding='latin1'))
    with open(os.path.join(dataset_dir, 'video_y_valid.pkl').replace('\\', '/'), 'rb') as handle:
        y_valid = np.array(pickle.load(handle))

    # Load testing data
    with open(os.path.join(dataset_dir, 'video_X_test.pkl').replace('\\', '/'), 'rb') as handle:
        X_test = np.array(pickle.load(handle, encoding='latin1'))
    with open(os.path.join(dataset_dir, 'video_y_test.pkl').replace('\\', '/'), 'rb') as handle:
        y_test = np.array(pickle.load(handle))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
