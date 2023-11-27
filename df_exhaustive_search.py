import numpy as np
from tqdm.notebook import tqdm as tq
import nmslib
import time
import csv
import os

from scipy.spatial import distance
from knn_library import graph_dst

model_knn = graph_dst()
model_knn.dst_type = 1
model_knn.dst_func = model_knn.get_dst_type()


def main(efb):
    #file = "b{}_hd_imagenet".format(efb)
    file = "vgg16_hd_imagenet"

    #
    dtype_label = np.uint16
    dtype_vector = np.float32

    #
    FILE_PATH = 'corr_mtrx_{}_train/'.format(file)

    X_l = []
    with open(FILE_PATH + 'txt_classes.txt', 'r') as csv_file:
        data = csv.reader(csv_file)
        X_l = np.array([row[0] for row in data], dtype=dtype_label)


    matrix_train = []
    lines_file = open(FILE_PATH + 'txt_vectors.txt', 'r')
    while True:
        line = lines_file.readline()
        if not line:
            break
        matrix_train.append(line.strip().replace(',', ' '))

    #
    FILE_PATH = 'corr_mtrx_{}_val/'.format(file)

    Y_l = []
    with open(FILE_PATH + 'txt_classes.txt', 'r') as csv_file:
        data = csv.reader(csv_file)
        Y_l = np.array([row[0] for row in data], dtype=dtype_label)

    matrix_val = []
    lines_file = open(FILE_PATH + 'txt_vectors.txt', 'r')
    while True:
        line = lines_file.readline()
        if not line:
            break
        matrix_val.append(line.strip().replace(',', ' '))

    #
    print('train instances: ', len(matrix_train))
    print('val instances: ', len(matrix_val))


    index = nmslib.init(method='hnsw', space='bit_hamming', data_type=nmslib.DataType.OBJECT_AS_STRING, dtype=nmslib.DistType.INT) 
    index.addDataPointBatch(matrix_train)


    # Set index parameters
    # These are the most important onese
    M = 16
    efC = 200

    num_threads = 2
    index_time_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post' : 2}


    #
    start = time.time()
    index.createIndex(index_time_params)
    end = time.time() 
    print('Index-time parameters', index_time_params)
    print('Indexing time = %f' % (end-start))


    #
    knn1 = 0
    knn5 = 0
    knn10 = 0

    votingknn5 = 0
    votingknn7 = 0
    votingknn9 = 0
    

    hsp1 = 0
    hsp_plus = 0
    topinf_av = []

    print("starting ... ")
    nbrs = index.knnQueryBatch(matrix_val, k=300, num_threads=4)

    for idx_u, nbr in enumerate(nbrs):
        #print("{}/{}".format(idx_u, len(matrix_val)))

        u = matrix_val[idx_u]
        idxs = nbr[0]
        X_ = np.array([matrix_train[i].split(' ') for i in idxs], dtype=dtype_label)
        X_l_ = X_l[idxs]

        # knn
        for i, knn_v in enumerate([1, 5, 10]):
            if Y_l[idx_u] in X_l_[:knn_v]:

                if   i == 0:
                    knn1 += 1
                elif i == 1:
                    knn5 += 1
                elif i == 2:
                    knn10 += 1

        #
        for i, knn_v in enumerate([5, 7, 9]):
            vals, idx = np.unique(X_l_[:knn_v], return_counts=True)
            idx = np.where(idx == idx.max())[0]

            if Y_l[idx_u] in vals[idx]:

                if   i == 0:
                    votingknn5 += 1
                elif i == 1:
                    votingknn7 += 1
                elif i == 2:
                    votingknn9 += 1

        # hsp        
        u_ = np.array(u.split(' '), dtype=dtype_label)

        X_w = list(map(lambda x: distance.hamming(u_, x), X_))
        X_w_idx = np.argsort(X_w)

        X_ = X_[X_w_idx]
        X_l_ = X_l_[X_w_idx]

        N = model_knn.get_hsp_items(u_, X_, X_l_)
        vals, counts = np.unique(N, return_counts=True)

        idx_counts = counts.argsort()
        vals_   =   vals[idx_counts]
        counts_ = counts[idx_counts]

        idxs = np.unique(counts_)[::-1]

        idx_c = np.where(counts_ == idxs[0])[0]
        n_vals_ = vals_[idx_c]

        if Y_l[idx_u] in n_vals_:
            hsp1 += 1

        if Y_l[idx_u] in N:
            hsp_plus += 1
            topinf_av.append(len(N))


    knn1 = knn1 / len(matrix_val) 
    knn5 = knn5  / len(matrix_val)
    knn10 = knn10 / len(matrix_val)
    votingknn5 = votingknn5 / len(matrix_val)
    votingknn7 = votingknn7 / len(matrix_val)
    votingknn9 = votingknn9 / len(matrix_val)

    hsp1 = hsp1 / len(matrix_val)
    hsp_plus = hsp_plus / len(matrix_val)
    topinf_av = np.array(topinf_av).sum() / len(topinf_av)

    os.system("knn1: {}, knn5: {}, knn10: {}, votingknn5: {}, votingknn7: {}, votingknn9: {}, hsp1: {}, hsp_plus: {}, topinf_av: {}".format(knn1,
                                                                                                                                            knn5,
                                                                                                                                            knn10,
                                                                                                                                            votingknn5,
                                                                                                                                            votingknn7,
                                                                                                                                            votingknn9,
                                                                                                                                            hsp1,
                                                                                                                                            hsp_plus,
                                                                                                                                            topinf_av ))

    print("===")
    print("knn1 {}".format(knn1))
    print("knn5 {}".format(knn5))
    print("knn10 {}".format(knn10))
    print("votingknn5 {}".format(votingknn5))
    print("votingknn7 {}".format(votingknn7))
    print("votingknn9 {}".format(votingknn9))
    print("hsp1 {}".format(hsp1))
    print("hsp_plus {}".format(hsp_plus))
    print("topinf_av {}".format(topinf_av))



if __name__ == "__main__":
    for ef in [0]:
        main(efb=ef)