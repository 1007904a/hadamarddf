import os
import csv
import time

import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


from knn_library import graph_dst

model_knn = graph_dst()
model_knn.dst_type = 1
model_knn.dst_func = model_knn.get_dst_type()


def main(_wb, file, sdir00, sub_dir):
    #
    dtype_label = np.uint32

    #
    FILE_PATH = f"data/corr_mtrx_00_{file}_train/"

    X_l = []
    with open(FILE_PATH + 'txt_classes.txt', 'r') as csv_file:
        data = csv.reader(csv_file)
        X_l = np.array([row[0] for row in data], dtype=dtype_label)

    X = []
    matrix_train = []
    lines_file = open(FILE_PATH + 'txt_vectors.txt', 'r')
    while True:
        line = lines_file.readline()
        if not line:
            break
        X.append(line.strip().replace('.0', ''))

    #
    FILE_PATH = f"data/corr_mtrx_00_{file}_val/"

    Y_l = []
    with open(FILE_PATH + 'txt_classes.txt', 'r') as csv_file:
        data = csv.reader(csv_file)
        Y_l = np.array([row[0] for row in data], dtype=dtype_label)


    Y = []
    matrix_val = []
    lines_file = open(FILE_PATH + 'txt_vectors.txt', 'r')
    while True:
        line = lines_file.readline()
        if not line:
            break
        Y.append(line.strip().replace('.0', ''))


    file_ans  = f"index_files/ans_vals.txt"

    for i in [0]: #range(64): #in [0, 1, 2, 3, 4, 5]:
        FILE_PATH = f"index_files/es_df_{i}0_{file}.txt"
        #FILE_PATH = f"index_files{_wb}/es_{i}_{file}.txt"
        #FILE_PATH = f"index_files/{sub_dir}/es_{i}_{file}.txt"
        #FILE_PATH = f"index_files{_wb}/es_{sdir00[0]}_{file}.txt"
        #FILE_PATH = f"index_files/resnet50_H1024_01/es_0{i}_{file}.txt"

        lines_file = open(FILE_PATH, 'r')
        #
        print(FILE_PATH)

        #
        idx_u = -1

        knn1  = 0
        knn5  = 0
        knn10 = 0

        lst_knn1  = []
        lst_knn5  = []
        lst_knn10 = []

        lst_n_class_knn1  = []
        lst_n_class_knn5  = []
        lst_n_class_knn10 = []
        lst_n_vals_knn1  = []
        lst_n_vals_knn5  = []
        lst_n_vals_knn10 = []

        votingknn3 = 0        
        votingknn5 = 0
        votingknn7 = 0
        votingknn9 = 0

        lst_votingknn3 = []        
        lst_votingknn5 = []
        lst_votingknn7 = []
        lst_votingknn9 = []

        hsp1 = 0
        hsp_plus = 0
        topinf_av = []

        while True:
            line = lines_file.readline()
            if not line:
                break
            
            idx_u += 1

            X_l_ = np.array(line.strip().split(','), dtype=dtype_label)
            X_l_ = X_l[X_l_]

            # knn
            for i, knn_v in enumerate([1, 5, 10]):
                _vals = X_l_[:knn_v]
                if Y_l[idx_u] in _vals:
                    _idxs, _counts = np.unique(_vals, return_counts=True)
                    _idx_val_class = np.where(_idxs == Y_l[idx_u])[0]

                    if   i == 0:
                        knn1 += 1
                        lst_knn1.append(Y_l[idx_u])
                        lst_n_class_knn1.append(len(_idxs))
                        lst_n_vals_knn1.append(len(_counts[_idx_val_class]))

                    elif i == 1:
                        knn5 += 1
                        lst_knn5.append(Y_l[idx_u])
                        lst_n_class_knn5.append(len(_idxs))
                        lst_n_vals_knn5.append(len(_counts[_idx_val_class]))
                            
                    elif i == 2:
                        knn10 += 1
                        lst_knn10.append(Y_l[idx_u])
                        lst_n_class_knn10.append(len(_idxs))
                        lst_n_vals_knn10.append(len(_counts[_idx_val_class]))

                else:
                    if   i == 0:
                        lst_knn1.append(X_l_[0])
                    elif i == 1:
                        lst_knn5.append(X_l_[0])
                    elif i == 2:
                        lst_knn10.append(X_l_[0])

            # knn-voting
            for i, knn_v in enumerate([3, 5, 7, 9]):
                vals, idx = np.unique(X_l_[:knn_v], return_counts=True)
                idx = np.where(idx == idx.max())[0]

                if Y_l[idx_u] in vals[idx]:
                    _idx_c = np.where(vals == Y_l[idx_u])[0]

                    if   i == 0:
                        votingknn3  += 1
                        lst_votingknn3.append(Y_l[idx_u])

                    elif i == 1:
                        votingknn5  += 1
                        lst_votingknn5.append(Y_l[idx_u])

                    elif i == 2:
                        votingknn7  += 1
                        lst_votingknn7.append(Y_l[idx_u])

                    elif i == 3:
                        votingknn9  += 1
                        lst_votingknn9.append(Y_l[idx_u])

                else:
                    if   i == 0:
                        lst_votingknn3.append(vals[0])
                    elif i == 1:
                        lst_votingknn5.append(vals[0])
                    elif i == 2:
                        lst_votingknn7.append(vals[0])
                    elif i == 3:
                        lst_votingknn9.append(vals[0])


            # hsp        
            X_l_ = np.array(line.strip().split(','), dtype=dtype_label)
            X_ = list(map(lambda x: X[x].strip().split(','), X_l_))
            X_ = np.array(X_, dtype=np.uint8)

            X_l_ = X_l[X_l_]

            u = np.array(Y[idx_u].split(','), dtype=dtype_label)
            N = model_knn.get_hsp_items(u, X_, X_l_)
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

            print(f"=> {idx_u} .. {hsp1/(idx_u+1)} .. {hsp1/(hsp_plus+1)}")


        Y_l_l = Y_l.shape[0]

        knn1 = knn1   / Y_l_l
        knn5 = knn5   / Y_l_l
        knn10 = knn10 / Y_l_l
        votingknn3 = votingknn3 / Y_l_l
        votingknn5 = votingknn5 / Y_l_l
        votingknn7 = votingknn7 / Y_l_l
        votingknn9 = votingknn9 / Y_l_l


        hsp1 = hsp1 / len(Y)
        hsp_plus = hsp_plus / len(Y)
        topinf_av = 0  #np.array(topinf_av).sum() / len(topinf_av)

        print(f"HSP: {hsp1}, HSP_P: {hsp_plus}, AVG: {topinf_av}")

        #""
        with open(file_ans, 'a+') as csv_file:
            csv_file.write("{}: knn1: {:.3f}, knn5: {:.3f}, knn10: {:.3f}, votingknn3: {:.3f}, votingknn5: {:.3f}, votingknn7: {:.3f}, votingknn9: {:.3f}, hsp1: {:.3f}, hsp_plus: {:.3f}, topinf_av: {:.3f}\n".format(FILE_PATH,
                                                                                                                                knn1,
                                                                                                                                knn5,
                                                                                                                                knn10,
                                                                                                                                votingknn3,
                                                                                                                                votingknn5,
                                                                                                                                votingknn7,
                                                                                                                                votingknn9,
                                                                                                                                hsp1,
                                                                                                                                hsp_plus,
                                                                                                                                topinf_av ))
        with open(file_ans, 'a+') as csv_file:
            # show data from library .... 
            for itm in [lst_knn1, lst_knn5, lst_knn10, lst_votingknn3, lst_votingknn5, lst_votingknn7, lst_votingknn9]:
                y_pred = np.array(itm, dtype=np.uint32)
                txt_ = 'Accuracy: {:.3f}, '.format   (accuracy_score(Y_l, y_pred))
                txt_ += 'Precision: {:.3f}, '.format (precision_score(Y_l, y_pred, average='micro'))
                txt_ += 'Recall: {:.3f}, '.format    (recall_score(Y_l, y_pred, average='micro'))
                txt_ += 'F1-score: {:.3f}\n'.format(f1_score(Y_l, y_pred, average='micro'))
                csv_file.write(txt_)

            csv_file.write("\n")
        csv_file.close()




if __name__ == "__main__":

    for itmx in ["resnet101_df_imagenet"]:
        main(_wb="", file=itmx, sdir00="", sub_dir="")
        #main(itmx, f"00_{itmy}", itmy)