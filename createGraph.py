import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.segmentation import slic, mark_boundaries
from sklearn import preprocessing
from scipy import sparse
import torch.nn.functional as F
from sklearn.decomposition import PCA
from torch_geometric.data import Data, Batch
import random
import warnings
from random import shuffle
from collections import defaultdict
import pickle as pk
import math

def createA(A):

    A = sparse.coo_matrix(A, dtype=np.float32)
    I = sparse.eye(A.shape[0])
    A_tilde = A + I
    degrees = A_tilde.sum(axis=0)[0].tolist()
    D = sparse.diags(degrees, [0])
    D = D.power(-0.5)
    A_tilde_hat = D.dot(A_tilde).dot(D)  #.dot函数的作用是获取两个元素a,b的乘积
    A_tilde_hat = sparse.coo_matrix(A_tilde_hat)

    return A_tilde_hat

# 对labels做后处理，防止出现label不连续现象
def SegmentsLabelProcess(labels):
    labels = np.array(labels, np.int64)
    H, W = labels.shape
    ls = list(set(np.reshape(labels, [-1]).tolist()))

    dic = {}
    for i in range(len(ls)):
        dic[ls[i]] = i

    new_labels = labels
    for i in range(H):
        for j in range(W):
            new_labels[i, j] = dic[new_labels[i, j]]
    return new_labels

class InitialGraph(object):
    def __init__(self, T, n_segments_init, device):

        self.height, self.width, self.bands = T.shape
        data = np.reshape(T, [self.height * self.width, self.bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [self.height, self.width, self.bands])
        self.n_segments_init = n_segments_init
        self.device = device

    def get_A(self, sigma=0.1):
        
        image = self.data

        segments = slic(image, n_segments=self.n_segments_init, start_label=0, compactness=0.1, max_num_iter=20)

        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments

        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count

        segments = np.reshape(segments, [-1])

        Q = np.zeros([superpixel_count, self.bands], dtype=np.float32)
        x = np.reshape(image, [-1, self.bands])

        S = np.zeros([self.height * self.width, superpixel_count], dtype=np.float32)

        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            Q[i] = superpixel
            S[idx, i] = 1
        
        self.Q = torch.from_numpy(Q).to(self.device)
        self.S = torch.from_numpy(S).to(self.device)
        
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)

        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:

                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = Q[idx1]
                    pix2 = Q[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss

        adj = createA(A)
        
        self.adj = adj

        return self.adj, self.Q, self.S, self.segments 
    
    def get_edge(self, adj=None):
        
        if adj == None:
            adj = self.adj
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long).to(self.device)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long).to(self.device)
        self.edge_index = torch.stack([row, col], dim=0).to(self.device)
        
        return self.edge_index


class SLIC(object):
    def __init__(self, T, args):

        self.height, self.width, self.bands = T.shape
        data = np.reshape(T, [self.height * self.width, self.bands])
        minMax = preprocessing.StandardScaler()
        data = minMax.fit_transform(data)
        self.data = np.reshape(data, [self.height, self.width, self.bands])
        self.args = args
        self.n_segments_init = math.ceil(self.height*self.width/self.args.segments_scale_init)

    def get_A(self, sigma=0.1):
        
        image = self.data

        segments = slic(image, n_segments=self.n_segments_init, start_label=0, compactness=0.1, max_num_iter=20)

        if segments.max() + 1 != len(list(set(np.reshape(segments, [-1]).tolist()))):
            segments = SegmentsLabelProcess(segments)
        self.segments = segments
        #print('segments.shape', segments.shape)  #segments.shape (984, 740)
        superpixel_count = segments.max() + 1
        self.superpixel_count = superpixel_count
        #print(segments)
        #print('type(segments)', type(segments))
        #print("superpixel_count", superpixel_count)

        segments = np.reshape(segments, [-1])

        
        Q = np.zeros([superpixel_count, self.bands], dtype=np.float32)
        x = np.reshape(image, [-1, self.bands])

        S = np.zeros([self.height * self.width, superpixel_count], dtype=np.float32)
        
        #Superpixel_gt = np.zeros([superpixel_count], dtype=np.float32)

        for i in range(superpixel_count):
            idx = np.where(segments == i)[0]
            count = len(idx)
            pixels = x[idx]
            superpixel = np.sum(pixels, 0) / count
            Q[i] = superpixel
            S[idx, i] = 1
            #Superpixel_gt[i] = np.argmax(np.bincount(self.GT[idx]))

        
        self.Q = torch.from_numpy(Q).to(self.args.device)
        self.S = torch.from_numpy(S).to(self.args.device)
        #self.Superpixel_gt = torch.from_numpy(Superpixel_gt).to(self.args.device)
        
        '''print('S.shape', S.shape)
        print('self.Q.shape', self.Q.shape)'''
        
        #A = torch.sigmoid(torch.matmul(self.Q, self.Q.T))
        #adj = createA(np.array(A.cpu()), self.args)
        
        A = np.zeros([self.superpixel_count, self.superpixel_count], dtype=np.float32)

        (h, w) = self.segments.shape
        for i in range(h - 2):
            for j in range(w - 2):
                sub = self.segments[i:i + 2, j:j + 2]
                sub_max = np.max(sub).astype(np.int32)
                sub_min = np.min(sub).astype(np.int32)
                if sub_max != sub_min:

                    idx1 = sub_max
                    idx2 = sub_min
                    if A[idx1, idx2] != 0:
                        continue

                    pix1 = Q[idx1]
                    pix2 = Q[idx2]
                    diss = np.exp(-np.sum(np.square(pix1 - pix2)) / sigma ** 2)
                    A[idx1, idx2] = A[idx2, idx1] = diss

        adj = createA(A)
        
        self.adj = adj

        return self.adj, self.Q, self.S, self.segments #, self.Superpixel_gt
    
    def get_edge(self, adj=None):
        
        if adj == None:
            adj = self.adj
        row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long).to(self.args.device)
        col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long).to(self.args.device)
        self.edge_index = torch.stack([row, col], dim=0).to(self.args.device)
        
        return self.edge_index


def ImagePCAFeatureReduction(data, out_channels=100):
    
    if data.size(-1) > out_channels:
            pca = PCA(n_components=out_channels) 
            data = pca.fit_transform(np.array(data.cpu()))
    return data

