from collections import defaultdict
import pickle as pk
import torch
import numpy as np
from torch_geometric.transforms import SVDFeatureReduction
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.data import Data, Batch
import random
import warnings
import createGraph
from random import shuffle
import time
from loadData import data_reader
from param_parser import parameter_parser
import math
import scipy.io as sio
from sklearn.decomposition import PCA

args = parameter_parser()
device = args.device

def get_indices(args, class_num, gt_reshape):

    train_index = []
    test_index = []
    
    # index_path = './dataset/{}/index/'.format(args.dataset_name)
    # mkdir(index_path)

    for i in range(class_num):
        #print("???????", i)
        idx = np.where(gt_reshape == i + 1)[-1]
        samplesCount = len(idx)
        # print("Class ",i,":", samplesCount)
        train_num = np.ceil(samplesCount * args.train_ratio).astype('int32')
        np.random.shuffle(idx)
        train_index.append(idx[:train_num])
        test_index.append(idx[train_num:])
        

    train_index = np.concatenate(train_index, axis=0)
    test_index = np.concatenate(test_index, axis=0)

    train_index = torch.LongTensor(train_index)
    test_index = torch.LongTensor(test_index)

    #print('train_index.shape', train_index.shape)  #torch.Size([295])

    return train_index, test_index

def get_label(gt_reshape, train_index, test_index):
    train_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(train_index)):
        train_samples_gt[train_index[i]] = gt_reshape[train_index[i]]

    test_samples_gt = np.zeros(gt_reshape.shape)
    for i in range(len(test_index)):
        test_samples_gt[test_index[i]] = gt_reshape[test_index[i]]

    return train_samples_gt, test_samples_gt


def label_to_one_hot(data_gt, class_num):
    height, width = data_gt.shape
    ont_hot_label = []
    for i in range(height):
        for j in range(width):
            temp = np.zeros(class_num, dtype=np.int64)
            if data_gt[i, j] != 0:
                temp[int(data_gt[i, j]) - 1] = 1
            ont_hot_label.append(temp)
    ont_hot_label = np.reshape(ont_hot_label, [height * width, class_num])
    return ont_hot_label


def get_label_mask(train_samples_gt, test_samples_gt, data_gt, class_num):
    height, width = data_gt.shape
    # train
    train_label_mask = np.zeros([height * width, class_num])
    temp_ones = np.ones([class_num])
    for i in range(height * width):
        if train_samples_gt[i] != 0:
            train_label_mask[i] = temp_ones
    train_label_mask = np.reshape(train_label_mask, [height * width, class_num])

    # test
    test_label_mask = np.zeros([height * width, class_num])
    temp_ones = np.ones([class_num])
    # test_samples_gt = np.reshape(test_samples_gt, [height * width])
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [height * width, class_num])

    return train_label_mask, test_label_mask

###for meta_demo.py###
def load_tasks(gt, task_pairs: list, task_type: str = None, dataname: str = None):
    
    if dataname is None:
        raise KeyError("dataname is None!")
    else:
        multi_gt_reshape = np.reshape(gt, -1)
        multi_class_num = np.max(multi_gt_reshape)
        # print(multi_class_num)  #7.0

    if task_type == "Binary change":
        if args.train_num is None:
            train_ratio = args.train_ratio
            max_iteration = 100
            i = 0
            while i < len(task_pairs) and i < max_iteration:
                task_1, task_2 = task_pairs[i]
                # train_index, test_index = get_indices(args, class_num, multi_gt_reshape)
                # train_samples_gt, test_samples_gt = get_label(multi_gt_reshape, train_index, test_index)
                task2_train_index = []
                task2_test_index = []

                idx_2 = np.where(multi_gt_reshape == task_2)[-1]
                samplesCount_2 = len(idx_2)
                train_num_2 = np.ceil(samplesCount_2 * train_ratio).astype('int32')
                #train_num_2 = 5
                # if train_num_2 < 2:
                #     train_num_2 = 2
                np.random.shuffle(idx_2)
                task2_train_index.append(idx_2[:train_num_2])  
                task2_test_index.append(idx_2[train_num_2:])  

                task2_train_index = np.concatenate(task2_train_index, axis=0)
                task2_test_index = np.concatenate(task2_test_index, axis=0)
                task2_train_index = torch.LongTensor(task2_train_index)
                task2_test_index = torch.LongTensor(task2_test_index)

                task1_train_index = []
                task1_test_index = []

                idx_1 = np.where(multi_gt_reshape == task_1)[-1]
                samplesCount_1 = len(idx_1)
                train_num_1 = np.ceil(samplesCount_1 * train_ratio).astype('int32')
                #train_num_1 = 5
                # if samplesCount_1*2 < samplesCount_2:
                #     train_num_1 = np.ceil(samplesCount_1 * train_ratio / 2).astype('int32')
                # if train_num_1 < 2:
                #     train_num_1 = 2
                np.random.shuffle(idx_1)
                task1_train_index.append(idx_1[:train_num_1])
                task1_test_index.append(idx_1[train_num_1:])   

                task1_train_index = np.concatenate(task1_train_index, axis=0)
                task1_test_index = np.concatenate(task1_test_index, axis=0)
                task1_train_index = torch.LongTensor(task1_train_index)
                task1_test_index = torch.LongTensor(task1_test_index)

                train_samples_gt = np.zeros(multi_gt_reshape.shape)
                test_samples_gt = np.zeros(multi_gt_reshape.shape)
                for j in range(len(task1_train_index)):
                    train_samples_gt[task1_train_index[j]] = 2
                for j in range(len(task1_test_index)):
                    test_samples_gt[task1_test_index[j]] = 2
                for j in range(len(task2_train_index)):
                    train_samples_gt[task2_train_index[j]] = 1
                for j in range(len(task2_test_index)):
                    test_samples_gt[task2_test_index[j]] = 1
                
                i = i + 1
                yield task_1, task_2, train_samples_gt, test_samples_gt, train_num_1, train_num_2
        else:
            train_num = args.train_num
            max_iteration = 100
            i = 0
            while i < len(task_pairs) and i < max_iteration:
                task_1, task_2 = task_pairs[i]
                # train_index, test_index = get_indices(args, class_num, multi_gt_reshape)
                # train_samples_gt, test_samples_gt = get_label(multi_gt_reshape, train_index, test_index)
                task2_train_index = []
                task2_test_index = []

                idx_2 = np.where(multi_gt_reshape == task_2)[-1]
                #train_num_2 = 5
                # if train_num_2 < 2:
                #     train_num_2 = 2
                np.random.shuffle(idx_2)
                task2_train_index.append(idx_2[:train_num])  
                task2_test_index.append(idx_2[train_num:])  

                task2_train_index = np.concatenate(task2_train_index, axis=0)
                task2_test_index = np.concatenate(task2_test_index, axis=0)
                task2_train_index = torch.LongTensor(task2_train_index)
                task2_test_index = torch.LongTensor(task2_test_index)

                task1_train_index = []
                task1_test_index = []

                idx_1 = np.where(multi_gt_reshape == task_1)[-1]
                #train_num_1 = 5
                # if samplesCount_1*2 < samplesCount_2:
                #     train_num_1 = np.ceil(samplesCount_1 * train_ratio / 2).astype('int32')
                # if train_num_1 < 2:
                #     train_num_1 = 2
                np.random.shuffle(idx_1)
                task1_train_index.append(idx_1[:train_num])
                task1_test_index.append(idx_1[train_num:])   

                task1_train_index = np.concatenate(task1_train_index, axis=0)
                task1_test_index = np.concatenate(task1_test_index, axis=0)
                task1_train_index = torch.LongTensor(task1_train_index)
                task1_test_index = torch.LongTensor(task1_test_index)

                train_samples_gt = np.zeros(multi_gt_reshape.shape)
                test_samples_gt = np.zeros(multi_gt_reshape.shape)
                for j in range(len(task1_train_index)):
                    train_samples_gt[task1_train_index[j]] = 2
                for j in range(len(task1_test_index)):
                    test_samples_gt[task1_test_index[j]] = 2
                for j in range(len(task2_train_index)):
                    train_samples_gt[task2_train_index[j]] = 1
                for j in range(len(task2_test_index)):
                    test_samples_gt[task2_test_index[j]] = 1
                
                i = i + 1
                yield task_1, task_2, train_samples_gt, test_samples_gt, train_num, train_num
    elif task_type == "Multiple change":
        max_iteration = 100
        i = 0
        while i < len(task_pairs) and i < max_iteration:
            task_1, task_2 = task_pairs[i]   

            task1_index = []

            idx_1 = np.where(multi_gt_reshape == task_1)[-1]
            train_num_1 = len(idx_1)
            np.random.shuffle(idx_1)
            task1_index.append(idx_1[:])  

            task1_index = np.concatenate(task1_index, axis=0)
            task1_index = torch.LongTensor(task1_index)
            
            task2_index = []

            # idx_2 = np.where(multi_gt_reshape != task_1 and multi_gt_reshape != 0)[-1]
            idx_2 = np.where(np.logical_and(multi_gt_reshape != task_1, multi_gt_reshape != 0))[-1]
            train_num_2 = len(idx_2)
            np.random.shuffle(idx_2)
            task2_index.append(idx_2[:])   

            task2_index = np.concatenate(task2_index, axis=0)
            task2_index = torch.LongTensor(task2_index)

            # print(task1_train_index.shape)  #torch.Size([1964])
            train_samples_gt = np.zeros(multi_gt_reshape.shape)
            for j in range(len(task1_index)):
                train_samples_gt[task1_index[j]] = 2
            for j in range(len(task2_index)):
                train_samples_gt[task2_index[j]] = 1

            i = i + 1

            yield task_1, task_2, train_samples_gt, train_num_1, train_num_2

    else:
        raise KeyError("task_type should be Binary change and Multiple change!")

def get_overalmask(gt):
    multi_gt_reshape = np.reshape(gt, -1)
    multi_class_num = np.max(multi_gt_reshape)
    
    train_index = []
    test_index = []
    if args.train_num is None:
        train_ratio = args.train_ratio_multi
        for i in range(multi_class_num):
            idx = np.where(multi_gt_reshape == i + 1)[-1]
            samplesCount = len(idx)
            # print("Class ",i,":", samplesCount)
            train_num = np.ceil(samplesCount * train_ratio).astype('int32')
            #train_num = 5
            # if train_num < 2:
            #     train_num = 2
            np.random.shuffle(idx)
            train_index.append(idx[:train_num])
            test_index.append(idx[train_num:])
    else:
        train_num = args.train_num
        for i in range(multi_class_num):
            idx = np.where(multi_gt_reshape == i + 1)[-1]
            #train_num = 5
            # if train_num < 2:
            #     train_num = 2
            np.random.shuffle(idx)
            train_index.append(idx[:train_num])
            test_index.append(idx[train_num:])

    train_index = np.concatenate(train_index, axis=0)
    train_index = torch.LongTensor(train_index)
    test_index = np.concatenate(test_index, axis=0)
    test_index = torch.LongTensor(test_index)
        
    train_samples_gt = np.zeros(multi_gt_reshape.shape)
    test_samples_gt = np.zeros(multi_gt_reshape.shape)

    for j in range(len(train_index)):
        train_samples_gt[train_index[j]] = multi_gt_reshape[train_index[j]]

    for j in range(len(test_index)):
        test_samples_gt[test_index[j]] = multi_gt_reshape[test_index[j]]

    height, width = gt.shape
    test_label_mask = np.zeros([height * width, multi_class_num])
    temp_ones = np.ones([multi_class_num])
    for i in range(height * width):
        if test_samples_gt[i] != 0:
            test_label_mask[i] = temp_ones
    test_label_mask = np.reshape(test_label_mask, [height * width, multi_class_num])

    return train_samples_gt, test_samples_gt, test_label_mask

def ImagePCAFeatureReduction(data, out_channels=1):
    
    if data.size(-1) > out_channels:
            pca = PCA(n_components=out_channels) 
            data = pca.fit_transform(np.array(data.cpu()))
    # print(data.shape)
    return data

class main():
    def __init__(self, input_dim=1):
        super(main, self).__init__()   
        self.input_dim = input_dim 
    
    def fit(self):
        
        args = parameter_parser()
        dataname = args.dataset_name 
        if dataname == "China":            
            data1 = data_reader.China().normal_cube1
            data2 = data_reader.China().normal_cube2
            data_gt = data_reader.China().truth
            data_multigt = data_reader.China().multitruth
        elif dataname == "USA":
            data1 = data_reader.USA().normal_cube1
            data2 = data_reader.USA().normal_cube2
            data_gt = data_reader.USA().truth
            data_multigt = data_reader.USA().multitruth
        elif dataname == "Bay":
            data1 = data_reader.Bay().normal_cube1
            data2 = data_reader.Bay().normal_cube2
            data_gt = data_reader.Bay().truth
            data_multigt = None
        elif dataname == "River":
            data1 = data_reader.River().normal_cube1
            data2 = data_reader.River().normal_cube2
            data_gt = data_reader.River().truth
            data_multigt = None
        elif dataname == "Hermiston":            
            data1 = data_reader.Hermiston().normal_cube1
            data2 = data_reader.Hermiston().normal_cube2
            data_gt = data_reader.Hermiston().truth
            data_multigt = data_reader.Hermiston().multitruth
        elif dataname == "Benton":            
            data1 = data_reader.Benton().normal_cube1
            data2 = data_reader.Benton().normal_cube2
            data_gt = data_reader.Benton().truth
            data_multigt = data_reader.Benton().multitruth
        elif dataname == "Urban":            
            data1 = data_reader.Urban().normal_cube1
            data2 = data_reader.Urban().normal_cube2
            data_gt = None
            data_multigt = data_reader.Urban().multitruth
        else:
            raise ValueError("Unkknow dataset")

        #Concat
        T1 = torch.tensor(data1).to(args.device)
        T1 = torch.unsqueeze(T1.permute([2, 0, 1]), 0)
        T2 = torch.tensor(data2).to(args.device)
        T2 = torch.unsqueeze(T2.permute([2, 0, 1]), 0)

        # T = torch.cat([T1, T2], dim=1)
        T = T1 - T2

        # superpixels
        ls1 = createGraph.SLIC(data1, args)
        ls2 = createGraph.SLIC(data2, args)
        tic0 = time.time()  # 返回当前时间的时间戳（1970纪元后经过的浮点秒数）。
        
        A1, Q1, S1, Seg1 = ls1.get_A()
        edge_index1 = ls1.get_edge()
        A2, Q2, S2, Seg2 = ls2.get_A()
        edge_index2 = ls2.get_edge()

        x1_ = Q1
        x1 = ImagePCAFeatureReduction(x1_, self.input_dim)
        x1 = torch.tensor(x1).to(args.device)
        x2_ = Q2
        x2 = ImagePCAFeatureReduction(x2_, self.input_dim)
        x2 = torch.tensor(x2).to(args.device)
        
        toc0 = time.time()   
        SLIC_Time = toc0 - tic0
        print('CD_SLIC_Time', SLIC_Time)
        
        # create graph
        graphdata1 = Data(x = x1, edge_index=edge_index1, T=T)
        graphdata2 = Data(x = x2, edge_index=edge_index2, T=T)
                          
        #
        pk.dump(graphdata1, open('./dataset/{}/superpixelgraph1.data'.format(dataname), 'bw'))
        pk.dump(graphdata2, open('./dataset/{}/superpixelgraph2.data'.format(dataname), 'bw'))
        
        return data1, data2, data_gt, data_multigt, A1, Q1, A2, Q2, S1, S2, Seg1, Seg2

if __name__ == "__main__":
    
    data1, data2, data_gt, data_multigt, A1, Q1, A2, Q2, S1, S2, Seg1, Seg2 = main().fit()
    