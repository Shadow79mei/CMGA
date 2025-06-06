
from utils import seed_everything, seed

seed_everything(seed)

from torch import nn, optim
import torch
from torch_geometric.loader import DataLoader
from param_parser import parameter_parser
import data_process
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
import warnings
import time
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from model import FinalModel
from random import shuffle
import scipy.io as sio
from copy import deepcopy
from sklearn.metrics import precision_score

import warnings
warnings.filterwarnings("ignore")

args = parameter_parser()

def classification_map(map, ground_truth, dpi, save_path):
    fig = plt.figure(frameon=False)
    fig.set_size_inches(ground_truth.shape[1] * 2.0 / dpi, ground_truth.shape[0] * 2.0 / dpi)

    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    fig.add_axes(ax)

    ax.imshow(map)
    fig.savefig(save_path, dpi=dpi)

    return 0

def list_to_map4Bay(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([125, 125, 125]) / 255.
        if item == 1:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 2:
            y[index] = np.array([255, 255, 255]) / 255.
    return y    

def list_to_map(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([0, 0, 0]) / 255.
        if item == 1:
            y[index] = np.array([255, 255, 255]) / 255.
    return y

def list_to_colormap(x_list, gt):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if gt[index] != 0:
           if item == 0:
              y[index] = np.array([0, 0, 0]) / 255.
           if item == 1:
              y[index] = np.array([255, 255, 255]) / 255.
        else:
            y[index] = np.array([125, 125, 125]) / 255.
    return y

def list_to_multimap4China(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([4, 6, 9]) / 255.
        if item == 1:
            y[index] = np.array([81, 157, 158]) / 255.
        if item == 2:
            y[index] = np.array([255, 232, 192]) / 255.
        if item == 3:
            y[index] = np.array([200, 158, 196]) / 255.
    return y    #China_Multiple

def list_to_multimap4USA(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([4, 6, 9]) / 255.
        if item == 1:
            y[index] = np.array([223, 221, 108]) / 255. # np.array([161, 222, 224]) / 255.
        if item == 2:
            y[index] = np.array([255, 232, 192]) / 255.
        if item == 3:
            y[index] = np.array([200, 158, 196]) / 255.
        if item == 4:
            y[index] = np.array([249, 134, 170]) / 255.
        if item == 5:
            y[index] = np.array([81, 157, 158]) / 255.
        if item == 6:
            y[index] = np.array([75, 237, 192]) / 255.
    return y    #USA_Multiple / Benton

def list_to_multimap4Hermiston(x_list):
    y = np.zeros((x_list.shape[0], 3))
    for index, item in enumerate(x_list):
        if item == 0:
            y[index] = np.array([4, 6, 9]) / 255.
        if item == 1:
            y[index] = np.array([81, 157, 158]) / 255.
        if item == 2:
            y[index] = np.array([200, 158, 196]) / 255.
        if item == 3:
            y[index] = np.array([223, 221, 108]) / 255.
        if item == 4:
            y[index] = np.array([255, 232, 192]) / 255.
        if item == 5:
            y[index] = np.array([249, 134, 170]) / 255.
    return y    #Hermiston

def generate_png(pred_test, gt_hsi, run_date, Dataset, path_result, iter_):

    gt = gt_hsi.flatten()
    x_label = pred_test
    x = np.ravel(x_label)

    if args.dataset_name == "Bay":
        y_list = list_to_colormap(x, gt)
        y_gt = list_to_map4Bay(gt)
    else:
        y_list = list_to_map(x)
        y_gt = list_to_map(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    classification_map(y_re, gt_hsi, 300,
                       path_result + run_date + Dataset + '_iter' + str(iter_) + '.png')

    classification_map(gt_re, gt_hsi, 300,
                       path_result + Dataset + '_gt.png')

    print('------Get Binary Change maps successful-------')

def classification_results(output, test_label_mask, test_samples_gt, training_time, train_num, iter_):
        tic3 = time.time()
        test_label_mask_cpu = test_label_mask.cpu().numpy()[:, 0].astype('bool')
        test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64')
        predict = torch.argmax(output, 1).cpu().numpy()

        classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu],
                                               predict[test_label_mask_cpu] + 1, digits=4)
        kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], predict[test_label_mask_cpu] + 1)
        toc3 = time.time()
        testing_time = toc3 - tic3
        print(classification, kappa)

        # store results
        print("save results")
        run_date = time.strftime('%Y%m%d-%H%M-', time.localtime(time.time()))
        f = open(args.path_result + run_date + args.dataset_name + '.txt', 'a+')
        str_results =   '\n ======================' \
                        + '\nrun data = ' + run_date \
                        + '\nSegmentation scale = ' + str(args.segments_scale_init) \
                        + '\nreductionchannel, input_dim, hid_dim, out_dim, in_ch, out_ch = '+str(reductionchannel) \
                        + "\nepoch = " + str(adapt_steps_meta_test) \
                        + "\ntrain ratio = " + str(args.train_ratio) \
                        + "\ntrain num = " + str(train_num) \
                        + '\ntrain time = ' + str(training_time) \
                        + '\ntest time = ' + str(testing_time) \
                        + '\n' + classification \
                        + "kappa = " + str(kappa) \
                        + '\n'
        f.write(str_results)
        f.close()

        # 保存图像
        if args.dataset_name == "Bay":
            generate_png(predict, data_gt, run_date, args.dataset_name, args.path_result, iter_)  
        else:
            generate_png(predict, data_gt-1, run_date, args.dataset_name, args.path_result, iter_)
        

def generate_multipng(pred_test, gt_hsi, run_date, Dataset, path_result, iter_):

    gt = gt_hsi.flatten()
    x_label = pred_test
    x = np.ravel(x_label)

    if args.dataset_name == "China":
        y_list = list_to_multimap4China(x)
        y_gt = list_to_multimap4China(gt)
    if args.dataset_name == "USA":
        y_list = list_to_multimap4USA(x)
        y_gt = list_to_multimap4USA(gt)
    if args.dataset_name == "Benton":
        y_list = list_to_multimap4USA(x)
        y_gt = list_to_multimap4USA(gt)
    if args.dataset_name == "Hermiston":
        y_list = list_to_multimap4Hermiston(x)
        y_gt = list_to_multimap4Hermiston(gt)

    y_re = np.reshape(y_list, (gt_hsi.shape[0], gt_hsi.shape[1], 3))
    gt_re = np.reshape(y_gt, (gt_hsi.shape[0], gt_hsi.shape[1], 3))

    classification_map(y_re, gt_hsi, 300,
                       path_result + run_date + Dataset + '_iter' + str(iter_) +  '_multi.png')

    classification_map(gt_re, gt_hsi, 300,
                       path_result + Dataset + '_multigt.png')

    print('------Get Multiple Change maps successful-------')

def compute_loss(output, train_samples_gt, train_samples_gt_onehot, train_label_mask):
    real_labels = train_samples_gt_onehot
    available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
    available_label_count = available_label_idx.sum()  # 有效标签的个数

    we = -torch.mul(real_labels, torch.log(output + 1e-12))
    we = torch.mul(we, train_label_mask)
    pool_cross_entropy = torch.sum(we) / available_label_count
    return pool_cross_entropy

def evalute(network_output, train_samples_gt, train_samples_gt_onehot, zeros):

    with torch.no_grad():
        available_label_idx = (train_samples_gt != 0).float()  # 有效标签的坐标,用于排除背景
        available_label_count = available_label_idx.sum()  # 有效标签的个数

        output = torch.argmax(network_output, 1)

        correct_prediction = torch.where(output == torch.argmax(train_samples_gt_onehot, 1),
                                            available_label_idx, zeros).sum()
        OA = correct_prediction.cpu() / available_label_count
        return OA

def load_data():

    ld = data_process.main(input_dim=1)
    data1, data2, data_gt, data_multigt, A1, Q1, A2, Q2, S1, S2, Seg1, Seg2 = ld.fit()

    return ld, data1, data2, data_gt, data_multigt, A1, Q1, A2, Q2, S1, S2, Seg1, Seg2

def meta_test_adam(graph1, graph2, model, meta_test_task_id_list,
                    dataname, task_type, adapt_steps_meta_test, multi_class_num, iter_=0):
    # meta-testing
    if len(meta_test_task_id_list) < 2:
        raise AttributeError("\ttask_id_list should contain at leat two tasks!")

    # print(meta_test_task_id_list)
    if task_type == "Binary change":
        task_pairs = [(meta_test_task_id_list[1], meta_test_task_id_list[0])]
        tic2 = time.time()

        for task_1, task_2, train_samples_gt, test_samples_gt, train_num_1, train_num_2 in data_process.load_tasks(data_gt, task_pairs, task_type, dataname):
            print('task_1: ', task_1, 'task_2: ', task_2)
            train_num = train_num_1 + train_num_2

            test_model = deepcopy(model)
            test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                                betas=(0.9, 0.7),
                                lr=0.001,
                                weight_decay=0.00001)

            train_label_mask, test_label_mask = data_process.get_label_mask(train_samples_gt, test_samples_gt, data_gt, class_num)

            # label transfer to one-hot encode
            train_gt = np.reshape(train_samples_gt, [height, width])
            test_gt = np.reshape(test_samples_gt, [height, width])

            train_gt_onehot = data_process.label_to_one_hot(train_gt, class_num)
            test_gt_onehot = data_process.label_to_one_hot(test_gt, class_num)

            train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
            test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)

            train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
            test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)

            train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)
            test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)

            test_model.train()
            test_adaptloss_list = []
            for _ in range(adapt_steps_meta_test):

                support_preds, loss_cl = test_model(S1=S1, S2=S2, graph1=graph1, graph2=graph2)
                support_loss = compute_loss(support_preds, train_samples_gt, train_gt_onehot, train_label_mask) + 0.3*loss_cl
                test_opi.zero_grad()
                support_loss.backward()
                test_opi.step()

                print('{}/{} training loss: {:.8f}'.format(_, adapt_steps_meta_test, support_loss))

                test_adaptloss_list.append(support_loss.detach().cpu().item())

            toc2 = time.time()
            training_time = toc2 - tic2

            test_model.eval()
            query_preds, loss_cl = test_model(S1=S1, S2=S2, graph1=graph1, graph2=graph2)
            classification_results(query_preds, test_label_mask, test_samples_gt, training_time, train_num, iter_)

            # 绘图部分
            plt.figure(figsize=(8, 8.5))
            plt.plot(np.linspace(1, adapt_steps_meta_test, len(test_adaptloss_list)), test_adaptloss_list, color='green')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig(args.path_result + args.dataset_name + '_adaptloss.png')

    if task_type == "Multiple change":
        np.random.shuffle(meta_test_task_id_list)
        print(meta_test_task_id_list)
        #[2, 6, 3, 5, 4, 7]]
        tic2 = time.time()
        task_pairs = []
        for i in range(0, len(meta_test_task_id_list), 1):
                pair1 = (meta_test_task_id_list[i], 1)
                task_pairs.append(pair1)

        print(task_pairs)

        scores = torch.zeros([height*width, multi_class_num], dtype=int).to(device)
        torch.set_default_dtype(torch.float32)
        confidence = torch.zeros([height*width, multi_class_num], dtype=torch.float).to(device)

        train_samples_gt_overal, test_samples_gt, test_label_mask = data_process.get_overalmask(data_multigt)
        # label transfer to one-hot encode
        train_gt_overal = np.reshape(train_samples_gt_overal, [height, width])
        test_gt = np.reshape(test_samples_gt, [height, width])
        test_gt_onehot = data_process.label_to_one_hot(test_gt, multi_class_num)
        test_samples_gt = torch.from_numpy(test_samples_gt.astype(np.float32)).to(args.device)
        test_gt_onehot = torch.from_numpy(test_gt_onehot.astype(np.float32)).to(args.device)
        test_label_mask = torch.from_numpy(test_label_mask.astype(np.float32)).to(args.device)

        k = 0
        train_num_positive = []
        train_num_negative = []
        for task_1, task_2, train_samples_gt, train_num_1, train_num_2 in data_process.load_tasks(train_gt_overal, task_pairs, task_type, dataname):
            print('task_1: ', task_1, 'task_2: ', task_2)
            k = k + 1
            train_num_positive.append((task_1, train_num_1))
            train_num_negative.append(train_num_2)

            test_model = deepcopy(model)
            test_opi = optim.Adam(filter(lambda p: p.requires_grad, test_model.parameters()),
                                betas=(0.9, 0.7),
                                lr=0.001,
                                weight_decay=0.00001)

            train_label_mask, _ = data_process.get_label_mask(train_samples_gt, test_samples_gt, data_multigt, 2)
            # label transfer to one-hot encode
            train_gt = np.reshape(train_samples_gt, [height, width])
            train_gt_onehot = data_process.label_to_one_hot(train_gt, 2)
            train_samples_gt = torch.from_numpy(train_samples_gt.astype(np.float32)).to(args.device)
            train_gt_onehot = torch.from_numpy(train_gt_onehot.astype(np.float32)).to(args.device)
            train_label_mask = torch.from_numpy(train_label_mask.astype(np.float32)).to(args.device)

            test_model.train()
            test_adaptloss_list = []
            for ep in range(adapt_steps_meta_test):
                support_preds, loss_cl = test_model(S1=S1, S2=S2, graph1=graph1, graph2=graph2)
                support_loss = compute_loss(support_preds, train_samples_gt, train_gt_onehot, train_label_mask) + 0.3*loss_cl
                test_opi.zero_grad()
                support_loss.backward()
                test_opi.step()


                print('{}/{} training loss: {:.8f}'.format(ep, adapt_steps_meta_test, support_loss))

                test_adaptloss_list.append(support_loss.detach().cpu().item())

            query_preds, _ = test_model(S1=S1, S2=S2, graph1=graph1, graph2=graph2)
            # print(query_preds.shape)  #torch.Size([58800, 2])
            predict = torch.argmax(query_preds, 1)
            # print(predict.shape)  #(58800,)
            idx = torch.where(predict == 1)[-1]
            # print(idx.shape)
            scores[idx, task_1-1] = scores[idx, task_1-1] + 1
            confidence[idx, task_1-1] = query_preds[idx, 0]
            #print(np.max(scores))  #3
            test_model.eval()

            # 绘图部分
            plt.figure(figsize=(8, 8.5))
            plt.plot(np.linspace(1, adapt_steps_meta_test, len(test_adaptloss_list)), test_adaptloss_list, color='green')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.savefig(args.path_result + args.dataset_name + str(k) + '_adaptloss.png')

        toc2 = time.time()
        training_time = toc2 - tic2
        test_label_mask_cpu = test_label_mask.cpu().numpy()[:, 0].astype('bool')
        test_samples_gt_cpu = test_samples_gt.cpu().numpy().astype('int64')

        tic3 = time.time()
        output = torch.argmax(scores, 1)
        # 判断一个以上的波段的值是否不为零
        non_zero_bands = (scores != 0).sum(dim=1) > 1
        # 获取同时满足条件的像素位置的索引
        indices = torch.nonzero(non_zero_bands).squeeze()
        # # 打印结果
        # print(indices)  #tensor([ 5221,  5222,  5418,  ..., 59172, 59173, 59174], device='cuda:1')
        # print('indices.shape', indices.shape)  #indices.shape torch.Size([6732])
        # print('max(indices)', max(indices))  #max(indices) tensor(59174, device='cuda:1')
        # 判断形状
        if indices.shape == torch.Size([]):
                pixel_index = indices
                band_indices =  []
                for v in range(multi_class_num):
                    if scores[pixel_index, v] != 0:
                        band_indices.append(v)
                confidence_subset = confidence[pixel_index, band_indices]
                best_band_index = torch.argmax(confidence_subset, dim=-1)
                best_band = band_indices[best_band_index]
                output[pixel_index] = best_band
        else:
            for t in range(len(indices)):
                # 获取像素位置对应波段位置的索引
                pixel_index = indices[t]
                # print(scores[pixel_index].shape)  #torch.Size([6])
                band_indices =  []
                for v in range(multi_class_num):
                    if scores[pixel_index, v] != 0:
                        band_indices.append(v)
                        # print(v)
                # print(band_indices)  #[4, 5]
                # 获取每个像素位置对应的最高置信度所在的波段
                confidence_subset = confidence[pixel_index, band_indices]
                # print(confidence_subset.shape)  #torch.Size([2])
                # print(confidence_subset)  #tensor([3.9413e-01, 2.0848e-04], device='cuda:1', grad_fn=<IndexBackward0>)
                best_band_index = torch.argmax(confidence_subset, dim=-1)
                # print('best_band_index', best_band_index)  #best_band tensor(0, device='cuda:1')
                best_band = band_indices[best_band_index]
                # print('best_band', best_band)  #best_band 4
                # 将最高置信度所在的波段的对应像素值设置为best_band
                output[pixel_index] = best_band
        output = output.cpu().numpy()
        # print(output.shape)  #(40500,)

        classification = classification_report(test_samples_gt_cpu[test_label_mask_cpu],
                                            output[test_label_mask_cpu] + 1, digits=4)
        kappa = cohen_kappa_score(test_samples_gt_cpu[test_label_mask_cpu], output[test_label_mask_cpu] + 1)

        toc3 = time.time()
        testing_time = toc3 - tic3
        print(classification, kappa)

        # store results
        print("save results")
        run_date = time.strftime('%Y%m%d-%H%M-', time.localtime(time.time()))
        f = open(args.path_result + run_date + args.dataset_name + '.txt', 'a+')
        str_results =   '\n ======================' \
                        + '\nMultiple Change' \
                        + '\nrun data = ' + run_date \
                        + '\nSegmentation scale = ' + str(args.segments_scale_init) \
                        + '\nreductionchannel = '+str(reductionchannel) \
                        + "\nepoch = " + str(adapt_steps_meta_test) \
                        + "\ntrain ratio = " + str(args.train_ratio_multi) \
                        + "\ntrain num(Positive) = " + str(train_num_positive) \
                        + "\ntrain num(Negative) = " + str(train_num_negative) \
                        + '\ntrain time = ' + str(training_time) \
                        + '\ntest time = ' + str(testing_time) \
                        + '\n' + classification \
                        + "kappa = " + str(kappa) \
                        + '\n'
        f.write(str_results)
        f.close()

        # 保存图像
        generate_multipng(output, data_multigt-1, run_date, args.dataset_name, args.path_result, iter_)


def model_components():
    model = FinalModel(b=None, height=height, width=width, bands=bands, flag=0, reductionchannel=reductionchannel, input_dim=1, hid_dim=128, out_dim=100,
                        in_ch=100, out_ch=64, num_classes=2, gnn_type=gnn_type)

    model.gnn.load_state_dict(torch.load(pre_train_path+gnn_type+'.pth'))
    model.cnn.load_state_dict(torch.load(pre_train_path+'cnn.pth'))
    model.featurefusion.load_state_dict(torch.load(pre_train_path+'featurefusion.pth'))
    model.fc.load_state_dict(torch.load(pre_train_path+'fullyconnected.pth'))

    for p in model.gnn.parameters():
        p.requires_grad = False
    # for p in model.cnn.parameters():
    #     p.requires_grad = False

    model.eval()

    return model

if __name__ == '__main__':
    print("PyTorch version:", torch.__version__)

    if torch.cuda.is_available():
        print("CUDA is available")
        print("CUDA version:", torch.version.cuda)
        device = args.device
    else:
        print("CUDA is not available")
        device = torch.device("cpu")

    print(device)
    #device = torch.device('cpu')

    ITER = 5
    dataname = args.dataset_name
    reductionchannel = 128
    gnn_type = 'TransformerConv'
    #gnn_type = 'GAT'
    #gnn_type = 'GCN'
    pre_train_path = './pre_trained_gnn/'

    # load data
    ld, data1, data2, data_gt, data_multigt, A1, Q1, A2, Q2, S1, S2, Seg1, Seg2 = load_data()

    class_num = 2
    height, width, bands = data1.shape
    gt_reshape = np.reshape(data_gt, [-1])
    multi_gt_reshape = np.reshape(data_multigt, [-1])

    # gt = torch.LongTensor(gt_reshape)
    # print('gt.shape', gt.shape)  # torch.Size([58800])

    print('CD dataname:', dataname)
    # print('data1.shape:', data1.shape)
    # print('data2.shape:', data2.shape)
    # print('data_gt.shape:', data_gt.shape)
    # print('gt_reshape.shape:', gt_reshape.shape)
    # print('class_num:', class_num)

    for iter in range(ITER):
        print("第{}次迭代".format(iter+1))
        # meta testing on target tasks
        graph1 = pk.load(open('./dataset/{}/superpixelgraph1.data'.format(dataname), 'br'))
        graph2 = pk.load(open('./dataset/{}/superpixelgraph2.data'.format(dataname), 'br'))
        print('metatest_graph1', graph1)
        print('metatest_graph2', graph2)

        print("meta testing on target tasks")
        adapt_steps_meta_test = 70
        model = model_components()

        total = sum([param.nelement() for param in model.parameters()])
        print('Number of parameter: % .4fM' % (total / 1e6))
        total_gnn = sum([param.nelement() for param in model.gnn.parameters()])
        print('Number of trainable parameter: % .4fM' % ((total-total_gnn) / 1e6))

        if args.task_type == "Binary change":
            print("Binary Change Detection")
            meta_test_task_id_list_binary = list(range(1, class_num + 1))
            meta_test_adam(graph1=graph1, graph2=graph2, model=model, meta_test_task_id_list=meta_test_task_id_list_binary,
                            dataname=dataname, task_type="Binary change", adapt_steps_meta_test=adapt_steps_meta_test, multi_class_num=class_num, iter_=iter)

        elif args.task_type == "Multiple change":
            print("Multiple Change Detection")
            multi_class_num = np.max(data_multigt)
            print('multi_class_num:', multi_class_num)
            meta_test_task_id_list_multiple = list(range(2, multi_class_num + 1))
            meta_test_adam(graph1=graph1, graph2=graph2, model=model, meta_test_task_id_list=meta_test_task_id_list_multiple,
                            dataname=dataname, task_type="Multiple change", adapt_steps_meta_test=70, multi_class_num=multi_class_num, iter_=iter)
        else:
            raise ValueError("Unkknow task_type")

