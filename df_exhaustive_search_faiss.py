import time
import csv
import os

import faiss
import numpy as np

def read_files(txt_file, n_lines=False, ranges_=[-1, -1]):
    lines_file = open(txt_file, 'r')

    if n_lines:
        n_mtrix_ = 0
        while True:
            if not lines_file.readline():
                break
            n_mtrix_ += 1

        return n_mtrix_

    elif sum(ranges_) > 0:
        idx_mt = 0
        matrix_ = []
        while True:
            line = lines_file.readline()
            if not line:
                break
            else:
                if ranges_[0] <= idx_mt < ranges_[1]:
                    matrix_.append(np.array(line.strip().split(','), dtype=np.float32))
                elif ranges_[1] < idx_mt:
                    break
                idx_mt += 1

        matrix_ = np.array(matrix_)
        return matrix_

    else:
        matrix_ = []
        while True:
            line = lines_file.readline()
            if not line:
                break
            matrix_.append(np.array(line.strip().split(','), dtype=np.float32))

        matrix_ = np.array(matrix_)
        return matrix_    


def main_test(dir_):
    #file = "b{}_hd_imagenet".format(efb)
    file = dir_
    #file_base = ""
    file_base = ""

    #
    #print("Reading set-train values")
    FILE_PATH_OUT = 'index_files/{}gt_{}.txt'.format(file_base, file)
    os.system("rm -rf {}".format(FILE_PATH_OUT))

    # val dataset ...
    FILE_PATH = '{}corr_mtrx_{}_val/'.format(file_base, file)  # ../hadamard_index/
    matrix_val = read_files(FILE_PATH + 'txt_vectors.txt')

    # train dataset ... 
    k_neighbors = 1000
    n_k_blocks  =    4
    k_neighbors_block = int(k_neighbors / n_k_blocks)

    I = []
    D = []

    FILE_PATH = '{}corr_mtrx_{}_train/'.format(file_base, file)  # ../hadamard_index/
    n_mtrix_train = read_files(FILE_PATH + 'txt_vectors.txt', n_lines=True)

    n_blocks = int(n_mtrix_train / n_k_blocks)
    l_n_mtrix_train = np.arange(0, n_mtrix_train, n_blocks).repeat(2)
    l_n_mtrix_train[:-1] = l_n_mtrix_train[1:]
    l_n_mtrix_train[-1] = n_mtrix_train
    l_n_mtrix_train = l_n_mtrix_train.reshape((int(l_n_mtrix_train.shape[0]/2), 2))

    #
    idx_mt = 0
    for x, y in l_n_mtrix_train:
        matrix_train = read_files(FILE_PATH + 'txt_vectors.txt', ranges_=[x, y])

        D_, I_ = faiss.knn(matrix_val, matrix_train, k_neighbors_block)
        I_ = I_ + x

        if idx_mt == 0:
            D = D_
            I = I_

        else:
            D = np.hstack((D, D_))
            I = np.hstack((I, I_))

        idx_mt += 1
    
    # write file output ... 
    with open(FILE_PATH_OUT, 'a+') as csv_file:

        for i in range(matrix_val.shape[0]):
            idxs = np.argsort(D[i])
            D[i] = D[i][idxs]
            I[i] = I[i][idxs]

            csv_file.write("{}\n".format(", ".join(np.array(I[i], dtype=str).tolist())))
        
        
def test00():
    files = ["resnet_101_df_imagenet"]
    for file in files:
        main_test(file)


if __name__ == "__main__":
    test00()
