import os
import numpy as np

CUR_PATH = os.path.split(os.path.realpath(__file__))[0]
print(CUR_PATH)


# # Load data for non-defended dataset for CW setting
def load_data_dc_numeric():
    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = os.path.join(CUR_PATH, "dataset")

    # Load training data
    X_train = np.load(os.path.join(dataset_dir, 'X_train.npy').replace('\\', '/'))
    y_train = np.load(os.path.join(dataset_dir, 'y_train.npy').replace('\\', '/'))

    # Load testing data
    X_test = np.load(os.path.join(dataset_dir, 'X_test.npy').replace('\\', '/'))
    y_test = np.load(os.path.join(dataset_dir, 'y_test.npy').replace('\\', '/'))

    # Load testing data
    X_valid = np.load(os.path.join(dataset_dir, 'X_valid.npy').replace('\\', '/'))
    y_valid = np.load(os.path.join(dataset_dir, 'y_valid.npy').replace('\\', '/'))

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)
    print("X: Validating data's shape : ", X_valid.shape)
    print("y: Validating data's shape : ", y_valid.shape)

    return X_train, y_train, X_test, y_test, X_valid, y_valid
