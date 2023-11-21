import copy
#import hnswlib
import numpy as np
from datetime import datetime
#import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from scipy.linalg import hadamard
from scipy.spatial import distance
from sklearn.neighbors import NearestNeighbors

from numba import njit, jit, prange, objmode, types

from multiprocessing import Process, Queue, current_process, freeze_support
import concurrent.futures

@jit(nopython=True)
def get_hamming(u, v):
    sims = 0
    for i in prange(1024):
        if u[i] == v[i]:
            sims += 1
    return sims

class graph_dst():
    def __init__(self, X_train=[], Y_test=[], X_labels=[], Y_labels=[]):
        self.X = X_train     # train
        self.Y = Y_test      # test
        self.X_l = X_labels  # train
        self.Y_l = Y_labels  # test
        self.X_file = ""     # if we have to much data on test Array this var is needed 

        # hadamard codes 
        self.hadamard_codes = 0
        # max neig needed
        self.neighbors = 10
        # dist used
        # 0 -> euclidean
        # 1 -> hadamard
        self.dst_type = 1
        self.dst_func = self.get_dst_type()
        # top to retrieve data 
        self.top0 = 0
        self.top1 = 0
        self.top5 = 0
        self.top10 = 0
        self.topinf = 0

        self.voting5 = 0
        self.voting7 = 0
        self.voting9 = 0
    
    def init(self):

        self.Y_l_len = self.Y_l.shape[0]

        """
        knn -> kNN
        hsp0 -> hsp
        hsp1 -> clusters hsp
        hsp2 -> infinite
        """

        self.final_Y_knn   = np.ones((self.Y_l_len, 6), dtype=np.uint8) * -1
        self.final_Y_hsp0  = np.ones((self.Y_l_len, 3), dtype=np.uint8) * -1
        self.final_Y_hsp1  = np.ones((self.Y_l_len, 3), dtype=np.uint8) * -1
        self.final_Y_hsp2  = np.ones((self.Y_l_len, 3), dtype=np.uint8) * -1


        #self.final_Y_l_1  = np.ones(self.Y_l_len, dtype=np.uint8) * -1
        #self.final_Y_l_5  = np.ones(self.Y_l_len, dtype=np.uint8) * -1
        #self.final_Y_l_10 = np.ones(self.Y_l_len, dtype=np.uint8) * -1


    @staticmethod
    @jit(nopython=True)
    def data_kernel(u, database):

        lst = np.zeros(len(database))

        for i in prange(len(database)): 
            
            diff = 0
            for j in prange(1024):
                v = 1 if database[i][j] == "1" else 0
                if u[i] == v:
                    diff += 1

            lst[i] = diff

        print(lst)

        return lst

    """
    def call_ml_zz(self, data):
        return [data[0], np.count_nonzero(data[1] != np.array(data[2].split(','), dtype=np.uint8))]

    def data_kernel(self, u, database):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            offset = 1
            for idx in range(int(len(database) / offset) + 1):
                print("final: {}".format(offset * (idx + 1)))

                lst_final = [range(len(database))]

                itms = database[offset * idx: offset * (idx + 1)]
                itms = [[offset * idx + i, u, itm] for i, itm in enumerate(itms)]

                for itm, ans in zip(itms, executor.map(self.call_ml_zz, itms)):
                    lst_final[ans[0]] = ans[1]

        return lst_final
    """

    """
    """
    def get_dst_type(self):

        # dst 
        if self.dst_type == 0:
            return distance.euclidean

        # hamming dst 
        elif self.dst_type == 1:
            return distance.hamming

    """
    """
    def get_knn(self):

        self.top0 = 0
        self.top1 = 0
        self.top5 = 0
        self.top10 = 0
        self.topinf = 0

        self.voting5 = 0
        self.voting7 = 0
        self.voting9 = 0

        print("{}".format(self.dst_func))
        
        C = self.X

        d1 = np.ones(self.X_l.shape, dtype=np.uint8) * -1
        
        for idx_u, u in enumerate(self.Y):
            print("{}/{}".format(idx_u + 1, self.Y_l_len))

            # 
            now_ = datetime.now()

            """
            offset = 100000
            for idx in range(int(self.X_l.shape[0] / offset) + 1):
                print("final: {}".format(offset * (idx + 1)))
                #database = list(map(lambda x: np.array(x.split(','), dtype=np.uint8),  C[offset * idx: offset * (idx + 1)]))
                #database = np.array(database, dtype=np.uint8)
                d1 = self.data_kernel(u, C[offset * idx: offset * (idx + 1)])
            """
            d1 = self.data_kernel(u, C)

            end_ = datetime.now()
            print("datetime: {}".format(end_ - now_))

            import ipdb; ipdb.set_trace()

            #
            d1_ = d1.argsort()[:self.neighbors]

            idxs = self.X_l[d1_]

            # @1
            #self.final_Y_l_1[idx_u] = idxs[0]
            if self.Y_l[idx_u] == idxs[:1]:
                self.top1 += 1
                self.final_Y_knn[idx_u][0] = idxs[0]

            # @5
            #vals, idx = np.unique(idxs[:5], return_counts=True)
            #idx = np.where(idx == idx.max())[0][0]

            if self.Y_l[idx_u] in idxs[:5]:
                self.top5 += 1
            """
                self.final_Y_knn[idx_u][1] = self.Y_l[idx_u]
            else:
                self.final_Y_knn[idx_u][1] = vals[idx]
            """

            # @10
            #vals, idx = np.unique(idxs[:10], return_counts=True)
            #idx = np.where(idx == idx.max())[0][0]

            if self.Y_l[idx_u] in idxs[:10]:
                self.top10 += 1
            """
                self.final_Y_knn[idx_u][2] = self.Y_l[idx_u]
            else:
                self.final_Y_knn[idx_u][2] = vals[idx]
            """

            #
            for i, knn_v in enumerate([5, 7, 9]):
                vals, idx = np.unique(idxs[:knn_v], return_counts=True)
                idx = np.where(idx == idx.max())[0]

                if self.Y_l[idx_u] in vals[idx]:

                    if   i == 0:
                        self.voting5 += 1
                    elif i == 1:
                        self.voting7 += 1
                    elif i == 2:
                        self.voting9 += 1

            print("{} <acc> @1: {}, @5: {}, @10: {}, v5: {}, v7: {}, v9:{}".format(idx_u + 1,
                                                                                self.top1 / (idx_u + 1), 
                                                                                self.top5 / (idx_u + 1), 
                                                                                self.top10 / (idx_u + 1),
                                                                                self.voting5 / (idx_u + 1),
                                                                                self.voting7 / (idx_u + 1),
                                                                                self.voting9 / (idx_u + 1)))

        self.top1 /= self.Y_l_len
        self.top5 /= self.Y_l_len
        self.top10 /= self.Y_l_len

        self.voting5 /= self.Y_l_len
        self.voting7 /= self.Y_l_len
        self.voting9 /= self.Y_l_len

        #print("acc @1: {}, @5: {}, @10: {}".format(self.top1, self.top5, self.top10))


    """
    """
    def get_hsp(self, use_predata=False):

        print("{}".format(self.dst_func))

        self.top0 = 0
        self.top1 = 0
        self.top5 = 0
        self.top10 = 0
        self.topinf = 0
        self.topinf_av = []
        
        if not use_predata:
            pass
        else:
            # kNN - Sklearn
            #neigh = NearestNeighbors(n_neighbors=self.neighbors)
            #neigh.fit(self.X)

            # hnswlib
            num_elements, dim = self.X.shape

            model_p = hnswlib.Index(space='l2', dim=dim) 
            model_p.init_index(max_elements=num_elements, ef_construction=64, M=64)
            model_p.set_ef(64)
            model_p.set_num_threads(4) 

            labels_x = np.arange(num_elements)
            model_p.add_items(self.X, labels_x)

        #max_points = int(self.Y_l.shape[0] * 0.01)
        max_points = int(self.X.shape[0] * 0.01)

        for idx_u, u in enumerate(self.Y):

            print("{}/{}".format(idx_u + 1, self.Y_l_len))

            N = []

            C = copy.deepcopy(self.X)
            C_l = copy.deepcopy(self.X_l)

            #
            dts = list(map(lambda c: self.dst_func(u, c), C))
            dts = np.array(dts)

            idx_dts = dts.argsort()
            idx_dts = np.array(idx_dts)


            C = C[idx_dts][:max_points]
            C_l = C_l[idx_dts][:max_points]

            N = self.get_hsp_items(u, C, C_l)

            vals, counts = np.unique(N, return_counts=True)

            idx_counts = counts.argsort()
            vals_   =   vals[idx_counts]
            counts_ = counts[idx_counts]

            idxs = np.unique(counts_)[::-1]

            # heuristic ... 
            N_ = []
            N2_ = []
            N3_ = []

            c_ = 1
            for idx in idxs:

                idx_c = np.where(counts_ == idx)[0]
                n_vals_ = vals_[idx_c]

                n_ = np.abs(n_vals_ - self.Y_l[idx_u])
                n_ = n_vals_[n_.argsort()].tolist()

                if c_ <= 1:
                    for n in n_:
                        N_.append(n)

                if c_ <= 2:
                    for n in n_:
                        N2_.append(n)

                if c_ <= 3:
                    for n in n_:
                        N3_.append(n)

                c_ += 1

            #
            # HSP
            #
            idx_c = np.where(counts_ == idxs[0])[0]
            n_vals_ = vals_[idx_c]

            if self.Y_l[idx_u] in n_vals_:
                self.final_Y_hsp0[idx_u][0] = self.Y_l[idx_u]
                self.top0 += 1
            else:
                ilr = np.where(counts == counts.max())[0][0]
                self.final_Y_hsp0[idx_u][0] = vals[ilr]

            #
            # Inf
            #
            if self.Y_l[idx_u] in N:
                self.final_Y_hsp2[idx_u][0] = self.Y_l[idx_u]
                self.topinf += 1

                self.topinf_av.append(len(N))

            else:
                self.final_Y_hsp2[idx_u][0] = self.final_Y_hsp0[idx_u][0]

            #
            # 1, 2, 3 clusters
            #

            """
            if self.Y_l[idx_u] == N_[:1]:
                self.top1 += 1
            self.final_Y_hsp1[idx_u][0] = N_[0]
            """

            if self.Y_l[idx_u] in N2_:
                self.top5 += 1
                self.final_Y_hsp1[idx_u][1] = self.Y_l[idx_u]
            else:
                vals, idx = np.unique(N2_, return_counts=True)
                idx = np.where(idx == idx.max())[0][0]
                self.final_Y_hsp1[idx_u][1] = vals[idx]

            if self.Y_l[idx_u] in N3_:
                self.top10 += 1
                self.final_Y_hsp1[idx_u][2] = self.Y_l[idx_u]
            else:
                vals, idx = np.unique(N3_, return_counts=True)
                idx = np.where(idx == idx.max())[0][0]
                self.final_Y_hsp1[idx_u][2] = vals[idx]


            print("hsp: {}, hsp_inf:{}, cum1: {}, cum2: {}, cum3: {}".format(  self.top0   / (idx_u + 1), 
                                                                                            self.topinf / (idx_u + 1),                 
                                                                                            self.top1 / (idx_u + 1), 
                                                                                            self.top5 / (idx_u + 1), 
                                                                                            self.top10 / (idx_u + 1)))


        self.top0 /= self.Y_l_len
        self.top1 /= self.Y_l_len
        self.top5 /= self.Y_l_len
        self.top10 /= self.Y_l_len
        self.topinf /= self.Y_l_len

        self.topinf_av = np.array(self.topinf_av)
        self.topinf_av = self.topinf_av.sum() / len(self.topinf_av)

        """
        print("hsp: {}, hsp_inf:{}, cum1: {}, cum2: {}, cum3: {}".format(self.top0,
                                                                                      self.topinf,
                                                                                      self.top1,
                                                                                      self.top5, 
                                                                                      self.top10))
        """


    def get_hsp_items(self, u, C, C_l ):
        N = []
        N_ = []

        #print("{}".format(self.dst_func))
        #print("===\n===\n")
        #print("u: {}".format(u))
        #print("C: {}".format(C))
        #print("C_l: {}".format(C_l))

        while len(C) > 0:
            dts = list(map(lambda c: self.dst_func(u, c), C))
            dts = np.array(dts)

            idx_dts = dts.argsort()
            idx_dts = np.array(idx_dts)

            N.append(C_l[idx_dts[0]])
            N_.append(C[idx_dts[0]])

            v = C[idx_dts[0]]
            
            idx_c = np.arange(C.shape[0])
            idx_c = list(filter(lambda x: self.dst_func(C[x], u) <= self.dst_func(C[x], v), idx_c))
            idx_c = np.array(idx_c)

            if len(idx_c) > 0:
                if len(idx_c) == len(C):
                    for c_l in C_l:
                        N.append(c_l)
                    C = C[idx_c]
                    C_l = C_l[idx_c]
                    break

                else:
                    C = C[idx_c]
                    C_l = C_l[idx_c]
            
            else:
                C = []
                C_l = []

        #print("N: {}".format(N))
        #import ipdb; ipdb.set_trace()

        N = np.array(N, dtype=np.uint16)

        return N

        #vals, counts = np.unique(N, return_counts=True)

        #idx = np.where(counts == counts.max())[0][0]

        #return int(vals[idx]), N


    #
    #
    #
    def get_hadamard_index(self):

        print("{}".format(self.dst_func))

        codes_size = 128
        codes = hadamard(codes_size)

        idxs_x, idxs_y = np.where(codes == -1)
        
        for x, y in zip(idxs_x, idxs_y):
            codes[x][y] = 0
        
        C = codes
        C_l = np.arange(codes_size)

        for idx_u, u in enumerate(self.Y):
            print("{}/{}".format(idx_u + 1, self.Y_l_len))

            d1 = list(map(lambda x: self.dst_func(u, x), C))
            d1 = np.array(d1)

            d1_ = d1.argsort()[:self.neighbors]

            idxs = C_l[d1_]

            print("{}/{}".format(self.Y_l[idx_u], idxs))


            # @1
            self.final_Y_l_1[idx_u] = idxs[0]
            if self.Y_l[idx_u] == idxs[0]:
                self.top1 += 1

            # @5
            vals, idx = np.unique(idxs[:5], return_counts=True)
            idx = np.where(idx == idx.max())[0][0]

            if self.Y_l[idx_u] in idxs[:5]:
                self.top5 += 1
                self.final_Y_l_5[idx_u] = self.Y_l[idx_u]
            else:
                self.final_Y_l_5[idx_u] = vals[idx]

            # @10
            vals, idx = np.unique(idxs, return_counts=True)
            idx = np.where(idx == idx.max())[0][0]

            if self.Y_l[idx_u] in idxs[:10]:
                self.top10 += 1
                self.final_Y_l_10[idx_u] = self.Y_l[idx_u]
            else:
                self.final_Y_l_10[idx_u] = vals[idx]

        self.top1 /= self.Y_l_len
        self.top5 /= self.Y_l_len
        self.top10 /= self.Y_l_len

        print("acc @1: {}, @5: {}, @10: {}".format(self.top1, self.top5, self.top10))

    #
    # by using a index ... and error ... 
    #
    def get_hadamard_index_n(self, classes=100, codes_size=128):

        # ...
        n_times = np.log2(codes_size)
        n_times = int(n_times) + 1
        recall = np.zeros(n_times)
        p_final = np.zeros(n_times)

        # ... 
        print("{}".format(self.dst_func))

        codes = hadamard(codes_size)

        idxs_x, idxs_y = np.where(codes == -1)
        
        for x, y in zip(idxs_x, idxs_y):
            codes[x][y] = 0
        
        C = codes
        C_l = np.arange(codes_size)

        total_good = 0


        for c in range(classes):

            print(" class: {}/{}".format(c + 1, classes))

            precision = [[] for i in range(n_times)]

            idxs = np.where(self.Y_l == c)[0]

            Y = self.Y[idxs]

            for idx_u, u in enumerate(Y):
            
                #d1 = list(map(lambda x: self.dst_func(u, x), C))
                d1 = list(map(lambda x: self.dst_func(u, x) * codes_size, C))

                d1 = np.array(d1)

                if d1.min() <= (codes_size / 4 - 1):

                    total_good += 1

                    for k in range(n_times):

                        d1_ = np.where(d1 <= (codes_size / (2 ** k)))[0]

                        if c in C_l[d1_]:
                            recall[k] += 1
                            precision[k] = np.union1d(precision[k], C_l[d1_]).tolist()


            #precision = [1 if len(itm) <= 1 else 1/len(itm) for itm in precision]
            precision = [len(itm) for itm in precision]
            precision = np.array(precision)

            p_final += precision

        recall /= total_good
        print("recall: {}".format(recall))

        p_final /= classes
        print("precision: {}".format(p_final))

        #print("original: {}".format(self.Y_l_len))
        #print("final: {}".format(total_good))

        #self.top1 /= total_good
        #self.top5 /= total_good
        #self.top10 /= total_good
        #print("acc @1: {}, @5: {}, @10: {}".format(self.top1, self.top5, self.top10))


if __name__ == "__main__":
    pass 
