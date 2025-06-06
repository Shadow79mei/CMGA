import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from torch_geometric.data import Data
from utils import act
import numpy as np
from data_process import ImagePCAFeatureReduction
from copy import deepcopy
from torch.autograd import Variable
import createGraph
import math

from param_parser import parameter_parser
args = parameter_parser()
    
class GNN(torch.nn.Module):
    def __init__(self, input_dim, hid_dim=None, out_dim=None, gcn_layer_num=2, gnn_type='GAT'):
        super().__init__()

        if gnn_type == 'GCN':
            GraphConv = GCNConv
        elif gnn_type == 'GAT':
            GraphConv = GATConv
        elif gnn_type == 'TransformerConv':
            GraphConv = TransformerConv
        else:
            raise KeyError('gnn_type can be only GAT, GCN and TransformerConv')

        self.gnn_type = gnn_type
        self.gcn_layer_num = gcn_layer_num
        if hid_dim is None:
            hid_dim = int(0.618 * input_dim)  # "golden cut"
        if out_dim is None:
            out_dim = hid_dim
        if gcn_layer_num < 2:
            self.conv_layers = GraphConv(input_dim, out_dim, heads=args.heads)
        elif gcn_layer_num == 2:
            self.conv_layers = torch.nn.ModuleList([GraphConv(input_dim, hid_dim, heads=args.heads), GraphConv(hid_dim*args.heads, out_dim, heads=args.heads)])
        else:
            layers = [GraphConv(input_dim, hid_dim, heads=args.heads)]
            for i in range(gcn_layer_num - 2):
                layers.append(GraphConv(hid_dim*args.heads, hid_dim, heads=args.heads))
            layers.append(GraphConv(hid_dim*args.heads, out_dim, heads=args.heads))
            self.conv_layers = torch.nn.ModuleList(layers)

    def forward(self, x, edge_index):
        if self.gcn_layer_num < 2:
            node_emb = self.conv_layers(x, edge_index)
        else:
            for conv in self.conv_layers[0:-1]:
                x = conv(x, edge_index)
                x = act(x)
                x = F.dropout(x, training=self.training)

            node_emb = self.conv_layers[-1](x, edge_index)

        return node_emb

def bn_conv_lrelu(in_c, out_c):
    return nn.Sequential(
        nn.BatchNorm2d(in_c), # InstanceNorm2d
        nn.Conv2d(in_c, out_c, 1, padding=0, bias=False),
        nn.LeakyReLU()
    )
def bn_bsconv_lrelu(in_c, out_c, kernel_size):
    return nn.Sequential(
        nn.BatchNorm2d(in_c),
        nn.Conv2d(in_c, in_c, 1, padding=0, bias=False),
        nn.Conv2d(in_c, out_c, kernel_size=kernel_size,
            stride=1, padding=kernel_size//2, groups=out_c),
        nn.LeakyReLU()
    )
class CAM_Module(torch.nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = torch.nn.Parameter(torch.zeros(1))
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma*out + x
        return out
    
class FeatureExtractor(nn.Module):
    def __init__(self, in_ch=128, out_ch=100):
        super().__init__()
        self.mid_channel = in_ch
        self.out_channel = out_ch
        self.scale_1 = bn_conv_lrelu(self.out_channel, self.out_channel)
        self.scale_2 = bn_bsconv_lrelu(self.out_channel, self.out_channel,kernel_size=3)
        self.scale_3 = bn_bsconv_lrelu(self.out_channel, self.out_channel,kernel_size=5)

        self.sigma0 = torch.nn.Parameter(torch.tensor([1.0],requires_grad=True))
        self.sigma1 = torch.nn.Parameter(torch.tensor([1.0],requires_grad=True))
        self.sigma2 = torch.nn.Parameter(torch.tensor([1.0],requires_grad=True))

        self.attention_spectral = CAM_Module(self.out_channel)
    
    def forward(self, s):

        s1 = self.scale_1(s)
        s2 = self.scale_2(s)
        s3 = self.scale_3(s)    
        y = self.sigma0*s1+self.sigma1*s2+self.sigma2*s3

        y = self.attention_spectral(y)
        return y

class FinalModel(torch.nn.Module):
    def __init__(self, b, height, width, bands, flag, reductionchannel, input_dim=150, hid_dim=128, out_dim=100, in_ch=100, out_ch=64, num_classes=2, gnn_type=None):
        super().__init__()
        self.num_class = num_classes
        self.height = height
        self.width = width
        self.bands = bands
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.flag = flag
        self.reductionchannel = reductionchannel

        
        self.gnn = GNN(input_dim=input_dim, hid_dim=hid_dim, out_dim=out_dim, gcn_layer_num=args.layers, gnn_type=gnn_type).to(args.device)
        
        if self.flag:
            self.featurereduction = bn_conv_lrelu(b, reductionchannel).to(args.device)
        else:
            self.featurereduction = bn_conv_lrelu(bands, reductionchannel).to(args.device)
        
        self.cnn = FeatureExtractor(in_ch=reductionchannel, out_ch=reductionchannel).to(args.device)

        self.featurefusion = torch.nn.Sequential(
            torch.nn.Linear(out_dim*args.heads+reductionchannel, out_dim),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(out_dim, out_ch),
            torch.nn.LeakyReLU()
            ).to(args.device)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(out_ch, num_classes),
            torch.nn.Softmax(dim=1)).to(args.device)
        

    def loss_cl(self, x1, x2, g1, g2, S1, S2):
        # 计算余弦相似度
        pixelCS = F.cosine_similarity(x1, x2, dim=1).to(args.device)
        loss = torch.mean(pixelCS) 

        return loss

    def forward(self, S1, S2=None, graph1=None, graph2=None):

        if S2 == None:
            x1 = graph1.x
            x2 = graph2.x
            _, b, h, w = x2.shape

            emb0 = self.gnn(x1, graph1.edge_index)
            dot = torch.matmul(S1, emb0)

            #print(x2.shape) #torch.Size([1, 200, 145, 145])
            stem = self.featurereduction(x2)
            pf_ = self.cnn(stem)
            # print(emb02.shape) #torch.Size([1, 60, 145, 145, 1])
            pf = torch.squeeze(pf_.permute([0, 2, 3, 1]), 0).reshape([h*w, -1])
            # print(dot2.shape)  #torch.Size([21025, 60])

            dot = torch.cat([dot, pf], dim=-1)

            out = self.featurefusion(dot)
            pre = self.fc(out)

            return pre

        else:
            x1 = graph1.x
            x2 = graph2.x  
            t = graph1.T 
            #print('???????', t.shape)

            ###################Spatial###################
            ########T1
            emb01 = self.gnn(x1, graph1.edge_index)
            dot1 = torch.matmul(S1, emb01)
            ########T2
            vice_model = deepcopy(self.gnn)
            for (vice_name, vice_model_param), (name, param) in zip(vice_model.named_parameters(), self.gnn.named_parameters()):
                    vice_model_param.data = param.data
            emb02 = vice_model(x2, graph2.edge_index)
            dot2 = torch.matmul(S2, emb02)
            dot = dot1 - dot2

            loss = self.loss_cl(dot1.contiguous(), dot2.contiguous(), emb01, emb02, S1, S2)
            
            ###################Spectral###################
            stem = self.featurereduction(t)
            pf_ = self.cnn(stem)
            #print(emb02.shape) #torch.Size([1, 100, 420, 140])
            pf = torch.squeeze(pf_.permute([0, 2, 3, 1]), 0).reshape([self.height*self.width, -1])

            ###################Fusion###################
            dot = torch.cat([dot, pf], dim=-1)
            
            out = self.featurefusion(dot)
            pre = self.fc(out)

            return pre, loss


if __name__ == '__main__':
    pass
