import os
import pickle
import numpy as np
from numpy import load
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

NB_CLASSES = 200
CUR_PATH = os.path.split(os.path.realpath(__file__))[0]
print(CUR_PATH)


def convert_npz_to_dataset():
    tally = dict()
    classDict = dict()
    classSet = set()
    x = list()
    y = list()

    npzfile = np.load(os.path.join(CUR_PATH, "tor_200w_2500tr.npz").replace('\\', '/'), allow_pickle=True)
    data = npzfile["data"]
    labels = npzfile["labels"]
    npzfile.close()

    print(data)
    print(labels)
    for i in range(len(labels)):
        cur_x = data[i]
        cur_y = labels[i]
        if cur_y not in classDict.keys():
            tally[cur_y] = 0
            classDict[cur_y] = len(classSet)
            classSet.add(cur_y)
        tally[cur_y] += 1

        x.append(cur_x)
        y.append(classDict[cur_y])

    for i in range(len(x)):
        x[i] = x[i][:1500]
        if len(x[i]) < 1500:
            x[i] = np.concatenate((x[i], np.array([0 for i in range(1500 - len(x[i]))])))

    print(np.array(x).shape)

    os.mkdir(os.path.join(CUR_PATH, 'dataset').replace('\\', '/'))
    np.save(os.path.join(CUR_PATH, 'dataset', "x_data.npy").replace('\\', '/'), x)
    np.save(os.path.join(CUR_PATH, 'dataset', "y_data.npy").replace('\\', '/'), y)




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
    X_train = np.load(os.path.join(dataset_dir, "x_data.npy").replace('\\', '/'))
    y_train = np.load(os.path.join(dataset_dir, "y_data.npy").replace('\\', '/'))

    # partial use for this dataset
    X_train, _, y_train, _ = train_test_split(X_train, y_train, test_size=0.64, random_state=42)

    # split to train, validate, test sets
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.44, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


# convert_npz_to_dataset()
LoadDataNoDefCW()