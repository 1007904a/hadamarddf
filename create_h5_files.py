import os
import csv
import time

import h5py
import numpy as np


file_base = f"{os.getcwd()}/models_df/"

n_bits = 8
n_bits_code = 1024

def create_h5_file(name="dbs.hdf5", labels=[], data=[]):
    _name = f'/media/eduardo/ADATA HV300/images_githup/{name}.hdf5'
    with h5py.File(f'{_name}', 'w') as f:
        f['labels'] = labels
        f['data'] = data    
    f.close()

def read_h5_file(name):
	with h5py.File(name, "r") as f:
		data = list(f['data'])
		labels = list(f['labels'])
	f.close()

	return data, labels

def main(file, rfile):
    #
    dtype_label = np.int32
    dtype_vector = np.float32

    #
    #print("Reading set-train values")
    FILE_PATH = '{}/corr_mtrx_{}_train/'.format(file_base, file)  # ../hadamard_index/

    #
    X_l = []
    with open(FILE_PATH + 'txt_classes.txt', 'r') as csv_file:
        data = csv.reader(csv_file)
        X_l = np.array([row[0] for row in data], dtype=dtype_label)

    #
    X = []
    lines_file = open(FILE_PATH + 'txt_vectors.txt', 'r')
    while True:
        line = lines_file.readline()
        if not line:
            break
        X.append(np.array(line.strip().split(','), dtype=np.float32))
        #import ipdb; ipdb.set_trace(); print("<=>")
        #line_ = line.strip().replace('.0', '').replace(',', '')
        #X.append([int(line_[i: i+n_bits], 2) for i in range(0, n_bits_code, n_bits)])

    #
    #print("Reading set-val values")
    FILE_PATH = '{}/corr_mtrx_{}_val/'.format(file_base, file)  # ../hadamard_index/

    Y_l = []
    with open(FILE_PATH + 'txt_classes.txt', 'r') as csv_file:
        data = csv.reader(csv_file)
        Y_l = np.array([row[0] for row in data], dtype=dtype_label)

    Y = []
    lines_file = open(FILE_PATH + 'txt_vectors.txt', 'r')
    while True:
        line = lines_file.readline()
        if not line:
            break
        Y.append(np.array(line.strip().split(','), dtype=np.float32))
        #import ipdb; ipdb.set_trace(); print("<=>")
        #line_ = line.strip().replace('.0', '').replace(',', '')
        #Y.append([int(line_[i: i+n_bits], 2) for i in range(0, n_bits_code, n_bits)])

    # train ... 
    name = f"{rfile}_train" #"imagenet_df_resnet101_train"
    #name = "imagenet_hdf_resnet101_train"
    create_h5_file(name=name, labels=X_l, data=X)
    # val ...
    name = f"{rfile}_val"   #"imagenet_df_resnet101_val"
    #name = "imagenet_hdf_resnet101_val"
    create_h5_file(name=name, labels=Y_l, data=Y)


if __name__ == "__main__":
    _dt = [["resnet_101_df_imagenet", "imagenet_df_resnet101"]]

    for _x, _y in _dt:
        print(f"{_x}")
        main(_x, _y)