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

class River(DataReader):
    def __init__(self):
        super(River, self).__init__()
        raw_data_package1 = sio.loadmat(r"/home/miaorui/datasets/River_before.mat")
        self.data_cube1 = raw_data_package1["river_before"].astype(np.float32)

        #self.data_cube1,_ = apply_PCA(data_cube1)

        data_package2 = sio.loadmat(r"/home/miaorui/datasets/river_after.mat")
        self.data_cube2 = data_package2["river_after"].astype(np.float32)

        #self.data_cube2,_ = apply_PCA(data_cube2)

        truth = sio.loadmat(r"/home/miaorui/datasets/Rivergt.mat")
        self.g_truth = truth["gt"].astype(np.float32)

class Santa(DataReader):
    def __init__(self):
        super(Santa, self).__init__()
        raw_data_package1 = sio.loadmat(r"/home/miaorui/datasets/barbara_2013.mat")
        data_cube1 = raw_data_package1["HypeRvieW"].astype(np.float32)
        self.data_cube1 = data_cube1
        data_package2 = sio.loadmat(r"/home/miaorui/datasets/barbara_2014.mat")
        data_cube2 = data_package2["HypeRvieW"].astype(np.float32)
        self.data_cube2 = data_cube2
        truth = sio.loadmat(r"/home/miaorui/datasets/barbara_gtChanges.mat")
        g_truth = truth["HypeRvieW"].astype(np.float32)
        g_truth = g_truth
        for i in range(g_truth.shape[0]):
            for j in range(g_truth.shape[1]):
                if g_truth[i, j] == 1:
                    g_truth[i, j] = 2
                elif g_truth[i, j] == 2:
                    g_truth[i, j] = 1
        self.g_truth = g_truth

class China(DataReader):
    def __init__(self):
         super(China, self).__init__()
         raw_data_package1 = sio.loadmat(r"/home/miaorui/datasets/Farm1.mat")
         data_cube1 = raw_data_package1["imgh"].astype(np.float32)
         self.data_cube1 = data_cube1[0:420, :, :]
         data_package2 = sio.loadmat(r"/home/miaorui/datasets/Farm2.mat")
         # print(data_package2)
         data_cube2 = data_package2["imghl"].astype(np.float32)
         self.data_cube2 = data_cube2[0:420, :, :]
         truth = sio.loadmat(r"/home/miaorui/datasets/GTChina1.mat")
         g_truth = truth["label"].astype(np.float32)

         self.g_truth = g_truth[0:420, :]

         multi_truth = sio.loadmat(r"/home/miaorui/data/ChinaData/China_MultChange.mat")
         multi_g_truth = multi_truth["Mul_CMap"].astype(np.float32)
         multi_g_truth = multi_g_truth[0:420, :] + 1
         self.multi_truth = multi_g_truth

class Bay(DataReader):
    def __init__(self):
        super(Bay, self).__init__()
        raw_data_package1 = sio.loadmat(r"/home/miaorui/datasets/Bay_Area_2013.mat")
        self.data_cube1 = raw_data_package1["HypeRvieW"].astype(np.float32)
        data_package2 = sio.loadmat(r"/home/miaorui/datasets/Bay_Area_2015.mat")
        self.data_cube2 = data_package2["HypeRvieW"].astype(np.float32)
        truth = sio.loadmat(r"/home/miaorui/datasets/bayArea_gtChanges2.mat")
        g_truth = truth["HypeRvieW"].astype(np.float32)
        for i in range(g_truth.shape[0]):
            for j in range(g_truth.shape[1]):
                if g_truth[i, j] == 1:
                    g_truth[i, j] = 2
                elif g_truth[i, j] == 2:
                    g_truth[i, j] = 1
        self.g_truth = g_truth 

class USA(DataReader):
    def __init__(self):
        super(USA, self).__init__()
        raw_data_package1 = sio.loadmat(r"/home/miaorui/datasets/Sa1.mat")
        self.data_cube1 = raw_data_package1["T1"].astype(np.float32)
        data_package2 = sio.loadmat(r"/home/miaorui/datasets/Sa2.mat")
        self.data_cube2 = data_package2["T2"].astype(np.float32)
        truth = sio.loadmat(r"/home/miaorui/datasets/SaGT.mat")
        g_truth = truth["GT"].astype(np.float32)
        for i in range(g_truth.shape[0]):
            for j in range(g_truth.shape[1]):
                if g_truth[i, j] == 1:
                    g_truth[i, j] = 2
                elif g_truth[i, j] == 2:
                    g_truth[i, j] = 1
        self.g_truth = g_truth

        multi_truth = sio.loadmat(r"/home/miaorui/data/USAData/USA_MultChange.mat")
        multi_g_truth = multi_truth["Mul_CMap"].astype(np.float32) + 1
        self.multi_truth = multi_g_truth

class Hermiston(DataReader):
    def __init__(self):
        super(Hermiston, self).__init__()
        raw_data_package1 = sio.loadmat(r"/home/miaorui/datasets/Hermiston/hermiston2004.mat")
        self.data_cube1 = raw_data_package1["HypeRvieW"].astype(np.float32)
        data_package2 = sio.loadmat(r"/home/miaorui/datasets/Hermiston/hermiston2007.mat")
        self.data_cube2 = data_package2["HypeRvieW"].astype(np.float32)

        multi_truth = sio.loadmat(r"/home/miaorui/datasets/Hermiston/rdChangesHermiston_5classes.mat")
        # print(multi_truth)
        multi_g_truth = multi_truth["gt5clasesHermiston"].astype(np.float32)
        self.multi_truth = multi_g_truth + 1

        g_truth = multi_g_truth
        # 寻找值不为0的区域
        nonZeroIndices = g_truth != 0
        # 将不为0的区域赋值为1
        g_truth[nonZeroIndices] = 1
        self.g_truth = g_truth + 1

class Benton(DataReader):
    def __init__(self):
        super(Benton, self).__init__()
        raw_data_package1 = sio.loadmat(r"/home/miaorui/Hyperspectral-Change-Detection-Dataset-Irrigated-Agricultural-Area-main/PreImg_2004.mat")
        self.data_cube1 = raw_data_package1["img_2004"].astype(np.float32)
        data_package2 = sio.loadmat(r"/home/miaorui/Hyperspectral-Change-Detection-Dataset-Irrigated-Agricultural-Area-main/PostImg_2007.mat")
        self.data_cube2 = data_package2["img_2007"].astype(np.float32)
        truth = sio.loadmat(r"/home/miaorui/Hyperspectral-Change-Detection-Dataset-Irrigated-Agricultural-Area-main/Reference_Map_Binary.mat")
        self.g_truth = truth["Ref_map_binary"].astype(np.float32) + 1
        multi_truth = sio.loadmat(r"/home/miaorui/Hyperspectral-Change-Detection-Dataset-Irrigated-Agricultural-Area-main/Reference_Map_Multiclass.mat")
        # print(multi_truth)
        multi_g_truth = multi_truth["Ref_map_multiclass"].astype(np.float32) 
        for i in range(multi_g_truth.shape[0]):
            for j in range(multi_g_truth.shape[1]):
                if multi_g_truth[i, j] == 1:
                    multi_g_truth[i, j] = 7
                elif multi_g_truth[i, j] == 7:
                    multi_g_truth[i, j] = 1
        self.multi_truth = multi_g_truth


class Urban(DataReader):
    def __init__(self):
        super(Urban, self).__init__()
        data_package = sio.loadmat(r"/home/miaorui/data/UrbanData/Urban_MultChange.mat")
        data_cube1 = data_package["X_3d"].astype(np.float32)
        self.data_cube1 = np.transpose(data_cube1, (1, 2, 0))
        data_cube2 = data_package["Y_3d"].astype(np.float32)
        self.data_cube2 = np.transpose(data_cube2, (1, 2, 0))
        self.g_truth = data_package["Two_CMap"].astype(np.float32) + 1

        multi_g_truth = data_package["Mul_CMap"].astype(np.float32) + 1
        self.multi_truth = multi_g_truth


if __name__ == "__main__":
    data1 = River().cube1
    data2 = River().cube2
    data_gt = River().truth

    print(data1.shape)
    print(data2.shape)
    print(data_gt.shape)