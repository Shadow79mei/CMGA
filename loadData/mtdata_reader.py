import numpy as np
import scipy.io as sio
import spectral as spy
import matplotlib.pyplot as plt
from collections import Counter



class DataReader():
    def __init__(self):
        self.data_cube1 = None
        self.data_cube2 = None
        self.g_truth = None
        self.multi_truth = None

    @property
    def cube1(self):
        return self.data_cube1

    @property
    def cube2(self):
        return self.data_cube2

    @property
    def truth(self):
        return self.g_truth.astype(np.int64)
    
    @property
    def multitruth(self):
        return self.multi_truth.astype(np.int64)

    @property
    def normal_cube1(self):
        return (self.data_cube1 - np.min(self.data_cube1)) / (np.max(self.data_cube1) - np.min(self.data_cube1))

    @property
    def normal_cube2(self):
        return (self.data_cube2 - np.min(self.data_cube2)) / (np.max(self.data_cube2) - np.min(self.data_cube2))

class Indian(DataReader):
    def __init__(self):
        super(Indian, self).__init__()

        data_mat = sio.loadmat('/home/miaorui/Datasets/Classify dataset/Indian_pines_corrected.mat')
        self.data_cube1 = data_mat['data'].astype(np.float32)
        gt_mat = sio.loadmat('/home/miaorui/Datasets/Classify dataset/Indian_pines_gt.mat')
        self.g_truth = gt_mat['groundT'].astype(np.float32)

class HongHu(DataReader):
    def __init__(self):
        super(HongHu, self).__init__()

        data_mat = sio.loadmat('/home/miaorui/Datasets/Classify dataset/WHU_Hi_HongHu.mat')
        self.data_cube1 = data_mat['WHU_Hi_HongHu'].astype(np.float32)
        gt_mat = sio.loadmat('/home/miaorui/Datasets/Classify dataset/WHU_Hi_HongHu_gt.mat')
        # print(gt_mat)
        self.g_truth = gt_mat['WHU_Hi_HongHu_gt'].astype(np.float32)

class Houston(DataReader):
    def __init__(self):
        super(Houston, self).__init__()

        data_mat = sio.loadmat('/home/miaorui/Datasets/Classify dataset/Houston2018/Houston.mat')
        self.data_cube1 = data_mat['Houston'].astype(np.float32)
        gt_mat = sio.loadmat('/home/miaorui/Datasets/Classify dataset/Houston2018/hu2018_gt.mat')
        self.g_truth = gt_mat['hu2018_gt'].astype(np.float32)

class Salinas(DataReader):
    def __init__(self):
        super(Salinas, self).__init__()

        data_mat = sio.loadmat('/home/miaorui/Datasets/Classify dataset/Salinas_corrected.mat')
        self.data_cube1 = data_mat['salinas_corrected'].astype(np.float32)
        gt_mat = sio.loadmat('/home/miaorui/Datasets/Classify dataset/Salinas_gt.mat')
        self.g_truth = gt_mat['salinas_gt'].astype(np.float32)

class Santa(DataReader):
    def __init__(self):
        super(Santa, self).__init__()
        raw_data_package1 = sio.loadmat(r"/home/miaorui/Datasets/datasets/barbara_2013.mat")
        data_cube1 = raw_data_package1["HypeRvieW"].astype(np.float32)
        self.data_cube1 = data_cube1
        data_package2 = sio.loadmat(r"/home/miaorui/Datasets/datasets/barbara_2014.mat")
        data_cube2 = data_package2["HypeRvieW"].astype(np.float32)
        self.data_cube2 = data_cube2
        truth = sio.loadmat(r"/home/miaorui/Datasets/datasets/barbara_gtChanges.mat")
        g_truth = truth["HypeRvieW"].astype(np.float32)
        g_truth = g_truth
        for i in range(g_truth.shape[0]):
            for j in range(g_truth.shape[1]):
                if g_truth[i, j] == 1:
                    g_truth[i, j] = 2
                elif g_truth[i, j] == 2:
                    g_truth[i, j] = 1
        self.g_truth = g_truth

if __name__ == "__main__":
    data = Indian().cube1
    data_gt = Indian().truth

    print(data.shape)
    print(data_gt.shape)