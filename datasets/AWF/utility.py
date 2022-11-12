import os
import pickle
import numpy as np
from numpy import load
import zipfile
from sklearn.utils import shuffle

NB_CLASSES = 200
CUR_PATH = os.path.split(os.path.realpath(__file__))[0]
print(CUR_PATH)


def LoadDataNoDefCW():
    print(os.path.join(CUR_PATH, "dataset", "dataset.zip").replace('\\', '/'))
    try:
        with zipfile.ZipFile(os.path.join(CUR_PATH, "dataset", "dataset.zip").replace('\\', '/'),
                             'r') as zip_ref:
            zip_ref.extractall()
    except:
        print("no zip file")
        # skip

    print("Loading non-defended dataset for closed-world scenario")
    # Point to the directory storing data
    dataset_dir = os.path.join(CUR_PATH, "dataset")

    # X represents a sequence of traffic directions
    # y represents a sequence of corresponding label (website's label)

    # Load training data

    X_train = load(os.path.join(dataset_dir, 'X_train.npy').replace('\\', '/'))
    X_train = X_train[:, 0:1500]
    y_train = load(os.path.join(dataset_dir, 'y_train.npy').replace('\\', '/'))

    # Load validation data
    X_valid = load(os.path.join(dataset_dir, 'X_valid.npy').replace('\\', '/'))
    X_valid = X_valid[:, 0:1500]
    y_valid = load(os.path.join(dataset_dir, 'y_valid.npy').replace('\\', '/'))
    X_valid, y_valid = shuffle(X_valid, y_valid)

    # Load testing data
    X_test = load(os.path.join(dataset_dir, 'X_test.npy').replace('\\', '/'))
    X_test = X_test[:, 0:1500]
    y_test = load(os.path.join(dataset_dir, 'y_test.npy').replace('\\', '/'))
    X_test, y_test = shuffle(X_test, y_test)

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test
