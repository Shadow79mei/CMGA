
from utils import seed_everything, seed

seed_everything(seed)

from torch import nn, optim
import torch
from torch_geometric.loader import DataLoader
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import warnings
import time
import data_process
from loadData import mtdata_reader
from torch_geometric.data import Data
from sklearn.decomposition import PCA
import createGraph
import math
from model import FinalModel
from meta import MAML
from random import shuffle
import scipy.io as sio
from copy import deepcopy

import warnings
warnings.filterwarnings("ignore")

def ImagePCAFeatureReduction(data, out_channels=1):
    
    if data.size(-1) > out_channels:
            pca = PCA(n_components=out_channels) 
            data = pca.fit_transform(np.array(data.cpu()))
    # print(data.shape)
    return data

def crop_image(image, block_size, overlap):
    element = image.shape
    # print(element)
    # (940, 475, 270)
    # (940, 475)
    height, width = element[0], element[1]
    cropped_blocks = []
    
    # 计算滑动窗口的步长
    stride = block_size - overlap
    
    # 遍历图像的块
    for y in range(0, height-block_size+1, stride):
        for x in range(0, width-block_size+1, stride):
            # 计算当前块的起始和结束坐标
            start_y = y
            end_y = y + block_size
            start_x = x
            end_x = x + block_size
            
            # 裁剪当前块并添加到结果列表中
            if len(element) > 2:
                block = image[start_y:end_y, start_x:end_x, :]
            else:
                block = image[start_y:end_y, start_x:end_x]
            cropped_blocks.append(block)
            
    return cropped_blocks

def fit4metatrain(dataname):
    
    if isinstance(dataname, list):
        data_mt_list, gt_mt_list = [], []
        for i in range(len(dataname)):
            if dataname[i] == "Indian":            
                data_mt = mtdata_reader.Indian().normal_cube1
                gt_mt = mtdata_reader.Indian().truth
            elif dataname[i] == "HongHu":
                data_mt = mtdata_reader.HongHu().normal_cube1
                gt_mt = mtdata_reader.HongHu().truth
            elif dataname[i] == "Houston":
                data_mt = mtdata_reader.Houston().normal_cube1
                gt_mt = mtdata_reader.Houston().truth
            elif dataname[i] == "Salinas":
                data_mt = mtdata_reader.Salinas().normal_cube1
                gt_mt = mtdata_reader.Salinas().truth
            elif dataname[i] == "Santa":
                data_mt = mtdata_reader.Santa().normal_cube1 - mtdata_reader.Santa().normal_cube2
                gt_mt = mtdata_reader.Santa().truth
            else:
                raise ValueError("Unkknow dataset")
            data_mt_list.append(data_mt)
            gt_mt_list.append(gt_mt)
    else:
        if dataname == "Indian":            
            data_mt = mtdata_reader.Indian().normal_cube1
            gt_mt = mtdata_reader.Indian().truth
        elif dataname == "HongHu":
            data_mt = mtdata_reader.HongHu().normal_cube1
            gt_mt = mtdata_reader.HongHu().truth
        elif dataname == "Houston":
            data_mt = mtdata_reader.Houston().normal_cube1
            gt_mt = mtdata_reader.Houston().truth
        elif dataname == "Salinas":
            data_mt = mtdata_reader.Salinas().normal_cube1
            gt_mt = mtdata_reader.Salinas().truth
        elif dataname == "Santa":
            data_mt = mtdata_reader.Santa().normal_cube1 - mtdata_reader.Santa().normal_cube2
            gt_mt = mtdata_reader.Santa().truth
        else:
            raise ValueError("Unkknow dataset")
    # print('data_mt_list.shape, gt_mt_list.shape', len(data_mt_list), len(gt_mt_list))  #data_mt_list.shape, gt_mt_list.shape 2 2
    # print(data_mt_list[0].shape, gt_mt_list[0].shape)  #(145, 145, 200) (145, 145)
    # print(data_mt_list[1].shape, gt_mt_list[1].shape)  #(512, 217, 204) (512, 217)
    #print(np.max(gt_mt))  #22
    if isinstance(dataname, list):
        data_list, gt_list = [], []
        A_list, Q_list, S_list, Seg_list = [], [], [], []
        graphdata_list, cnndata_list = [], []

        for i in range(len(dataname)):
            if dataname[i] ==  "Indian":
                data, gt = data_mt_list[i], gt_mt_list[i]
                height, width, bands = data.shape
                n_segments_init = math.ceil(height*width/25)
                print("n_segments_init for {}".format(dataname[i]), n_segments_init)
                T = torch.tensor(data).to(device)
                T = torch.unsqueeze(T.permute([2, 0, 1]), 0) 
                # superpixels
                ls = createGraph.InitialGraph(data, n_segments_init, device)
                A, Q, S, Seg = ls.get_A()
                edge_index = ls.get_edge()
                x = ImagePCAFeatureReduction(Q, input_dim)
                x = torch.tensor(x).to(device)
                # create graph
                graphdata = Data(x=x, edge_index=edge_index)
                cnndata = Data(x=T)
                #append
                data_list.append(data)
                gt_list.append(gt)
                A_list.append(A)
                Q_list.append(Q)
                S_list.append(S)
                Seg_list.append(Seg)
                graphdata_list.append(graphdata)
                cnndata_list.append(cnndata)
            elif dataname[i] == "Santa":
                data = crop_image(data_mt_list[i], 400, 20)
                gt = crop_image(gt_mt_list[i], 400, 20)
                print('{}: patch number of HSI'.format(dataname[i]), len(data))   
                print('{}: patch number of GT'.format(dataname[i]), len(gt))   
                n_segments_init_list = []
                T_list = []
                for j in range(len(data)):
                    T = torch.tensor(data[j]).to(device)
                    T = torch.unsqueeze(T.permute([2, 0, 1]), 0)
                    h, w, b = data[j].shape
                    n_segments_init = math.ceil(h*w/250)
                    print("n_segments_init for {}{}".format(dataname[i], j+1), n_segments_init)
                    n_segments_init_list.append(n_segments_init)
                    T_list.append(T)
                # superpixels
                for j in range(len(data)):
                    ls = createGraph.InitialGraph(data[j], n_segments_init_list[j], device)
                    A, Q, S, Seg = ls.get_A()
                    edge_index = ls.get_edge()
                    x = ImagePCAFeatureReduction(Q, input_dim)
                    x = torch.tensor(x).to(device)  
                    # create graph
                    graphdata = Data(x=x, edge_index=edge_index)
                    cnndata = Data(x=T_list[j])  
                    #append
                    data_list.append(data[j])
                    gt_list.append(gt[j])
                    A_list.append(A)
                    Q_list.append(Q)
                    S_list.append(S)
                    Seg_list.append(Seg)
                    graphdata_list.append(graphdata)
                    cnndata_list.append(cnndata)
            else:
                data = crop_image(data_mt_list[i], 200, 20)
                gt = crop_image(gt_mt_list[i], 200, 20)
                print('{}: patch number of HSI'.format(dataname[i]), len(data))  #9/39/2   
                print('{}: patch number of GT'.format(dataname[i]), len(gt))
                n_segments_init_list = []
                T_list = []
                for j in range(len(data)):
                    T = torch.tensor(data[j]).to(device)
                    T = torch.unsqueeze(T.permute([2, 0, 1]), 0)
                    h, w, b = data[j].shape
                    n_segments_init = math.ceil(h*w/25)
                    print("n_segments_init for {}{}".format(dataname[i], j+1), n_segments_init)
                    n_segments_init_list.append(n_segments_init)
                    T_list.append(T)
                # superpixels
                for j in range(len(data)):
                    ls = createGraph.InitialGraph(data[j], n_segments_init_list[j], device)
                    A, Q, S, Seg = ls.get_A()
                    edge_index = ls.get_edge()
                    x = ImagePCAFeatureReduction(Q, input_dim)
                    x = torch.tensor(x).to(device)  
                    # create graph
                    graphdata = Data(x=x, edge_index=edge_index)
                    cnndata = Data(x=T_list[j])  
                    #append
                    data_list.append(data[j])
                    gt_list.append(gt[j])
                    A_list.append(A)
                    Q_list.append(Q)
                    S_list.append(S)
                    Seg_list.append(Seg)
                    graphdata_list.append(graphdata)
                    cnndata_list.append(cnndata)
        
        pk.dump(graphdata_list, open('./dataset/metatraindatasets1.data', 'bw'))
        pk.dump(cnndata_list, open('./dataset/metatraindatasets2.data', 'bw'))
        # print('graphdata_list', graphdata_list)
        # graphdata_list [Data(x=[423, 1], edge_index=[2, 431]), Data(x=[1587, 1], edge_index=[2, 4347]), Data(x=[1378, 1], edge_index=[2, 3184])]
        # print('cnndata_list', cnndata_list)
        # cnndata_list [Data(x=[1, 200, 145, 145]), Data(x=[1, 204, 200, 200]), Data(x=[1, 204, 200, 200])]
        return data_list, gt_list, A_list, Q_list, S_list, Seg_list

    else:
        if dataname == "Indian":
            height, width, bands = data_mt.shape
            # print(data_mt.shape)  #(940, 475, 270)
            n_segments_init = math.ceil(height*width/25)
            print("n_segments_init for {}".format(dataname), n_segments_init)
            T = torch.tensor(data_mt).to(device)
            T = torch.unsqueeze(T.permute([2, 0, 1]), 0)
            
            # superpixels
            ls = createGraph.InitialGraph(data_mt, n_segments_init, device)
            
            A, Q, S, Seg = ls.get_A()
            # print('A.shape', A.shape)
            edge_index = ls.get_edge()
            # print('edge_index.shape', edge_index.shape)
            # print('superpixel_scale', superpixel_scale)    <Santa8938>
            x = ImagePCAFeatureReduction(Q, input_dim)
            x = torch.tensor(x).to(device)
            
            # create graph
            graphdata = Data(x=x, edge_index=edge_index)
            cnndata = Data(x=T)
            
            pk.dump(graphdata, open('./dataset/{}/metatraindatasets1.data'.format(dataname), 'bw'))
            pk.dump(cnndata, open('./dataset/{}/metatraindatasets2.data'.format(dataname), 'bw'))
            
            return data_mt, gt_mt, A, Q, S, Seg
        
        elif dataname == "HongHu" or dataname == "Houston" or dataname=='Salinas':
            data = crop_image(data_mt, 200, 20)
            gt = crop_image(gt_mt, 200, 20)
            print('{}: patch number of HSI'.format(dataname), len(data))   
            print('{}: patch number of GT'.format(dataname), len(gt)) #9/39/2
            # print(data[0].shape)   #(200, 200, 270)
            n_segments_init_list = []
            T_list = []
            for i in range(len(data)):
                T = torch.tensor(data[i]).to(device)
                T = torch.unsqueeze(T.permute([2, 0, 1]), 0)

                h, w, b = data[i].shape
                n_segments_init = math.ceil(h*w/25)
                print("n_segments_init for {}{}".format(dataname, i+1), n_segments_init)
                n_segments_init_list.append(n_segments_init)
                T_list.append(T)

            # superpixels
            A_list, Q_list, S_list, Seg_list = [], [], [], []
            graphdata_list, cnndata_list = [], []
            for i in range(len(data)):
                ls = createGraph.InitialGraph(data[i], n_segments_init_list[i], device)
                A, Q, S, Seg = ls.get_A()
                edge_index = ls.get_edge()
                x = ImagePCAFeatureReduction(Q, input_dim)
                x = torch.tensor(x).to(device)  

                # create graph
                graphdata = Data(x=x, edge_index=edge_index)
                cnndata = Data(x=T_list[i])  

                A_list.append(A)
                Q_list.append(Q)
                S_list.append(S)
                Seg_list.append(Seg)
                graphdata_list.append(graphdata)
                cnndata_list.append(cnndata)

            pk.dump(graphdata_list, open('./dataset/{}/metatraindatasets1.data'.format(dataname), 'bw'))
            pk.dump(cnndata_list, open('./dataset/{}/metatraindatasets2.data'.format(dataname), 'bw'))
            
            return data, gt, A_list, Q_list, S_list, Seg_list
        
        elif dataname == "Santa":
            data = crop_image(data_mt, 400, 20)
            gt = crop_image(gt_mt, 400, 20)
            print('{}: patch number of HSI'.format(dataname), len(data))   
            print('{}: patch number of GT'.format(dataname), len(gt))   #2
            n_segments_init_list = []
            T_list = []
            for i in range(len(data)):
                T = torch.tensor(data[i]).to(device)
                T = torch.unsqueeze(T.permute([2, 0, 1]), 0)

                h, w, b = data[i].shape
                n_segments_init = math.ceil(h*w/250)
                print("n_segments_init for {}{}".format(dataname, i+1), n_segments_init)
                n_segments_init_list.append(n_segments_init)
                T_list.append(T)

            # superpixels
            A_list, Q_list, S_list, Seg_list = [], [], [], []
            graphdata_list, cnndata_list = [], []
            for i in range(len(data)):
                ls = createGraph.InitialGraph(data[i], n_segments_init_list[i], device)
                A, Q, S, Seg = ls.get_A()
                edge_index = ls.get_edge()
                x = ImagePCAFeatureReduction(Q, input_dim)
                x = torch.tensor(x).to(device)  

                # create graph
                graphdata = Data(x=x, edge_index=edge_index)
                cnndata = Data(x=T_list[i])  

                A_list.append(A)
                Q_list.append(Q)
                S_list.append(S)
                Seg_list.append(Seg)
                graphdata_list.append(graphdata)
                cnndata_list.append(cnndata)
                # print("i=", i)

            # print(graphdata_list)
            # [Data(x=[327, 1], edge_index=[2, 339]), Data(x=[316, 1], edge_index=[2, 340])]
            # print(cnndata_list)
            pk.dump(graphdata_list, open('./dataset/{}/metatraindatasets1.data'.format(dataname), 'bw'))
            pk.dump(cnndata_list, open('./dataset/{}/metatraindatasets2.data'.format(dataname), 'bw'))
            
            return data, gt, A_list, Q_list, S_list, Seg_list

def load_tasks(gt, task_pairs: list):
    
    multi_gt_reshape = np.reshape(gt, -1)
    multi_class_num = np.max(multi_gt_reshape)
    # print(multi_class_num)  #7.0
        
    max_iteration = 100
    train_ratio = 0.5
    i = 0
    while i < len(task_pairs) and i < max_iteration:
        task_1, task_2 = task_pairs[i]   

        task1_train_index = []
        task1_test_index = []
        task2_train_index = []
        task2_test_index = []

        idx_1 = np.where(multi_gt_reshape == task_1)[-1]
        samplesCount_1 = len(idx_1)
        train_num_1 = np.ceil(samplesCount_1 * train_ratio).astype('int32')
        # print(train_num_1)  #1964
        np.random.shuffle(idx_1)
        task1_train_index.append(idx_1[:train_num_1])
        task1_test_index.append(idx_1[train_num_1:])
        
        idx_2 = np.where(multi_gt_reshape == task_2)[-1]
        samplesCount_2 = len(idx_2)
        train_num_2 = np.ceil(samplesCount_2 * train_ratio).astype('int32')
        np.random.shuffle(idx_2)
        task2_train_index.append(idx_2[:train_num_2])  
        task2_test_index.append(idx_2[train_num_2:])   

        task1_train_index = np.concatenate(task1_train_index, axis=0)
        task1_test_index = np.concatenate(task1_test_index, axis=0)
        task2_train_index = np.concatenate(task2_train_index, axis=0)
        task2_test_index = np.concatenate(task2_test_index, axis=0)
        task1_train_index = torch.LongTensor(task1_train_index)
        task1_test_index = torch.LongTensor(task1_test_index)
        task2_train_index = torch.LongTensor(task2_train_index)
        task2_test_index = torch.LongTensor(task2_test_index)

        # print(task1_train_index.shape)  #torch.Size([1964])
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
        yield task_1, task_2, train_samples_gt, test_samples_gt, multi_class_num
        

def compute_loss(output, train_samples_gt, train_samples_gt_onehot, train_label_mask):
    real_labels = train_samples_gt_onehot
    available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
    available_label_count = available_label_idx.sum()  # 有效标签的个数

    we = -torch.mul(real_labels, torch.log(output + 1e-12))
    we = torch.mul(we, train_label_mask)
    pool_cross_entropy = torch.sum(we) / available_label_count
    return pool_cross_entropy


def meta_train_maml(iter, graph, cnndata, S, gt_mt, epoch, maml, opt, meta_train_task_id_list, adapt_steps=2):
    if len(meta_train_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    shuffle(meta_train_task_id_list)

    task_pairs = [(meta_train_task_id_list[i], meta_train_task_id_list[i + 1]) for i in
                  range(0, len(meta_train_task_id_list)-1, 2)]
    #print("???????", len(task_pairs))  #7

    train_loss_list = []
    # meta-training
    train_loss_min = 1000000
    Epoch = 0
    no_improvement = 0
    for ep in range(epoch): 
        meta_train_loss = 0.0
        pair_count = 0

        for task_1, task_2, train_samples_gt, test_samples_gt, multi_class_num in load_tasks(gt=gt_mt, task_pairs=task_pairs):
            pair_count = pair_count + 1

            learner = maml.clone().to(device)
            # learner = deepcopy(maml.module).to(device)
            ###创建一个tensor与源tensor有相同的shape，
            # dtype和device，不共享内存地址，但新tensor的梯度会叠加在源tensor上

            print('task_1: ', task_1, 'task_2: ', task_2)
            train_label_mask, test_label_mask = data_process.get_label_mask(train_samples_gt, test_samples_gt, gt_mt, class_num)

            # label transfer to one-hot encode
            h, w = gt_mt.shape
            train_gt = np.reshape(train_samples_gt, [h, w])
            test_gt = np.reshape(test_samples_gt, [h, w])

            train_gt_onehot = data_process.label_to_one_hot(train_gt, class_num)
            test_gt_onehot = data_process.label_to_one_hot(test_gt, class_num)

            train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(device)
            test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(device)

            train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(device)
            test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(device)

            train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(device)
            test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(device)

            for j in range(adapt_steps):  # adaptation_steps
                support_loss = 0.

                support_preds = learner(S1=S, S2=None, graph1=graph, graph2=cnndata)
                #print(support_preds.shape)
                support_loss = compute_loss(support_preds, train_samples_gt, train_gt_onehot, train_label_mask)
                # learner.adapt(support_batch_loss)

                print('adapt {}/{} | loss: {:.8f}'.format(j + 1, adapt_steps, support_loss))

                learner.adapt(support_loss)

            query_preds = learner(S1=S, S2=None, graph1=graph, graph2=cnndata)
            query_loss = compute_loss(query_preds, test_samples_gt, test_gt_onehot, test_label_mask)
            print('query loss: {:.8f}'.format(query_loss))

            meta_train_loss += query_loss

        print('meta_train_loss @ epoch {}/{}: {}'.format(ep, epoch, meta_train_loss.item()))
        meta_train_loss = meta_train_loss / len(meta_train_task_id_list)
        opt.zero_grad()
        meta_train_loss.backward()
        # for name, param in learner.named_parameters():
        #     if param.grad is None:
        #         print(name, param.grad_fn)
        opt.step()
        if meta_train_loss < train_loss_min:
            train_loss_min = meta_train_loss
            torch.save({'optimizer_state_dict': opt.state_dict()}, "./pre_trained_gnn/{}.pth".format("maml_opt"))
            learner.savestate(gnn_type)
            no_improvement = 0
        else:
            no_improvement = no_improvement + 1
            if no_improvement > 100:
                print("*****************Early Stop!*****************")
                break

        if (ep + 1) % 10 == 0:
            Epoch = Epoch + 1
            train_loss_list.append(meta_train_loss.detach().cpu().item())

    # 绘图部分
    plt.figure(figsize=(8, 8.5))
    plt.plot(np.linspace(1, Epoch, len(train_loss_list)), train_loss_list, color='green')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig("./results/" + 'Meta_training_loss{}.png'.format(iter))

def model_components(flag, b=None, meta_data_list=None):

    if meta_data_list == True:
        model = FinalModel(b, height=None, width=None, bands=None, flag=flag, reductionchannel=reductionchannel, input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim,
                        in_ch=in_ch, out_ch=out_ch, num_classes=2, gnn_type=gnn_type)
        model.gnn.load_state_dict(torch.load(pre_train_path+gnn_type+'.pth'))
        model.cnn.load_state_dict(torch.load(pre_train_path+'cnn.pth'))
        model.featurefusion.load_state_dict(torch.load(pre_train_path+'featurefusion.pth'))
        model.fc.load_state_dict(torch.load(pre_train_path+'fullyconnected.pth'))

        maml = MAML(model, lr=adapt_lr, first_order=False, allow_nograd=False)
        opt = optim.Adam(maml.parameters(), meta_lr)
    else:
        model = FinalModel(b, height=None, width=None, bands=None, flag=flag, reductionchannel=reductionchannel, input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim,
                            in_ch=in_ch, out_ch=out_ch, num_classes=2, gnn_type=gnn_type)

        maml = MAML(model, lr=adapt_lr, first_order=False, allow_nograd=False)
        opt = optim.Adam(maml.parameters(), meta_lr)

    return model, maml, opt

if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = torch.device("cuda:1")
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(device)
    #device = torch.device('cpu')

    adapt_lr = 0.01
    meta_lr = 0.0001
    reductionchannel, input_dim, hid_dim, out_dim, in_ch, out_ch = 128, 1, 128, 100, 100, 64
    gnn_type = 'TransformerConv'
    #gnn_type = 'GAT'
    #gnn_type = 'GCN'
    pre_train_path = './pre_trained_gnn/'

    # load data
    mt_dataname = ['Santa', 'Indian']
	### help="Metatrain_data_name:Indian、Salinas、HongHu、Houston、Santa. 
    ### Default is ['Santa', 'Indian']")
    print('Classify dataname:', mt_dataname)

    class_num = 2

    data_mt_list, gt_mt_list, A_list, Q_list, S_list, Seg_list = fit4metatrain(mt_dataname)

    if isinstance(mt_dataname, list):
        graph_list = pk.load(open('./dataset/metatraindatasets1.data', 'br'))
        cnndata_list = pk.load(open('./dataset/metatraindatasets2.data', 'br'))
        print('metatrain_graph', graph_list)    
        print('metatrain_data2', cnndata_list)
    else:
        graph_list = pk.load(open('./dataset/{}/metatraindatasets1.data'.format(mt_dataname), 'br'))
        cnndata_list = pk.load(open('./dataset/{}/metatraindatasets2.data'.format(mt_dataname), 'br'))
        print('metatrain_graph', graph_list)
        print('metatrain_data2', cnndata_list)
    # meta training on source tasks
    epoch = 500
    if isinstance(graph_list, list):
        print("meta training on source tasks")
        tic1 = time.time()
        for i in range(len(graph_list)):
            h, w, b = data_mt_list[i].shape
            if i ==0:
                model, maml, opt = model_components(flag=1, b=b)
                bands_post = b
            else:
                if bands_post != b:
                    model, maml, opt = model_components(flag=1, b=b, meta_data_list=True)
                    bands_post = b

            class_mt = np.max(gt_mt_list[i]).astype(int)
            print('Metatrain_class_num:', class_mt)
            meta_train_task_id_list = list(set(np.reshape(gt_mt_list[i], [-1])))
            meta_train_task_id_list = list(filter(lambda x: x != 0, meta_train_task_id_list))
            print(meta_train_task_id_list)
            meta_train_maml(i, graph=graph_list[i], cnndata=cnndata_list[i], S=S_list[i], gt_mt=gt_mt_list[i], epoch=epoch, maml=maml, opt=opt,
                        meta_train_task_id_list=meta_train_task_id_list, adapt_steps=2)
        toc1 = time.time()
    else:
        print("meta training on source tasks")
        class_mt = np.max(gt_mt_list).astype(int)
        h, w, b = data_mt_list.shape
        print('Metatrain_class_num:', class_mt)
        meta_train_task_id_list = list(range(1, class_mt + 1))
        # print(meta_train_task_id_list)
        model, maml, opt = model_components(flag=1, b=b)
        tic1 = time.time()
        meta_train_maml(0, graph=graph_list, cnndata=cnndata_list, S=S_list, gt_mt=gt_mt_list, epoch=epoch, maml=maml, opt=opt,
                        meta_train_task_id_list=meta_train_task_id_list, adapt_steps=2)
        toc1 = time.time()

    print("Meta_training time: ", toc1-tic1)
