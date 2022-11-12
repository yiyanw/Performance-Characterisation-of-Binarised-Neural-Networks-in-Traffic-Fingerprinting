import os
from pandas import read_csv
import numpy as np
from sklearn.model_selection import train_test_split

import zipfile

CUR_PATH = os.path.split(os.path.realpath(__file__))[0]
print(CUR_PATH)

dataset_dir = os.path.join(CUR_PATH, "dataset").replace('\\', '/')


dirNames = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
            '20']


def convert_csv_to_npy_binary():
    print("convert csv files to dataset")
    tally = dict()
    classDict = dict()
    classSet = set()
    x = list()
    y = list()

    max_len = 0
    for dir in dirNames:
        for fileName in os.listdir(os.path.join(CUR_PATH, 'csv', dir).replace('\\', '/')):
            className = fileName.split("__")[0]
            if className not in classDict.keys():
                tally[className] = 0
                classDict[className] = len(classSet)
                classSet.add(className)
            tally[className] += 1

            cur_sample = read_csv(os.path.join(CUR_PATH, 'csv', dir, fileName).replace('\\', '/'))
            new = cur_sample['direction']

            if max_len < len(new):
                max_len = len(new)

            x.append(new)
            y.append(classDict[className])

    print("max_len")
    print(max_len)
    # cut and pad x
    for i in range(len(x)):
        x[i] = x[i][:600]
        if len(x[i]) < 600:
            x[i] = np.concatenate((x[i], np.array([0 for i in range(600 - len(x[i]))])))

    print(np.array(x).shape)

    os.mkdir(os.path.join(CUR_PATH, 'dataset').replace('\\', '/'))
    np.save(os.path.join(dataset_dir, "x_data_all_v1.npy").replace('\\', '/'), x)
    np.save(os.path.join(dataset_dir, "y_data_all_v1.npy").replace('\\', '/'), y)


def load_data_iot_binary(class_num="all", version="v1"):

    if not os.path.exists(dataset_dir):
        if not os.path.exists(os.path.join(CUR_PATH, "csv").replace('\\', '/')):
            unzip_file()
        convert_csv_to_npy_binary()

    class_num_str = str(class_num)
    X_train = np.load(os.path.join(dataset_dir, "x_data_" + class_num_str + "_" + version + ".npy").replace('\\', '/'))
    y_train = np.load(os.path.join(dataset_dir, "y_data_" + class_num_str + "_" + version + ".npy").replace('\\', '/'))

    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.36, random_state=42)
    X_test, X_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size=0.444, random_state=42)

    print("Data dimensions:")
    print("X: Training data's shape : ", X_train.shape)
    print("y: Training data's shape : ", y_train.shape)
    print("X: Validation data's shape : ", X_valid.shape)
    print("y: Validation data's shape : ", y_valid.shape)
    print("X: Testing data's shape : ", X_test.shape)
    print("y: Testing data's shape : ", y_test.shape)

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def unzip_file():
    print("unzip csv_google_home_dataset.zip")
    file_name = os.path.join(CUR_PATH, 'csv_google_home_dataset.zip').replace('\\', '/')
    zip_file = zipfile.ZipFile(file_name)
    os.mkdir(os.path.join(CUR_PATH, "csv").replace("\\", '/'))
    for names in zip_file.namelist():
        zip_file.extract(names, "csv")
    zip_file.close()
