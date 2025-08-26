import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Module, Parameter

from util_functions import use_cuda
device = use_cuda()


class CosineLayer(torch.nn.Module):

    def __init__(self, in_features, out_features):
        super(CosineLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.weightp = Parameter(torch.FloatTensor(out_features, in_features)).to(device) # torch.nn.parameter.
        torch.nn.init.uniform_(self.weightp) # trainable prototype

    def forward(self, inputs):
        cosine = F.linear(F.normalize(inputs, p=2, dim=1), F.normalize(self.weightp, p=2, dim=1))
        cosine = 5 * cosine
        return cosine

class GNNLayer(Module):
    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, features, adj, active=True):
        if active:
            support = self.act(F.linear(features, self.weight))
        else:
            support = F.linear(features, self.weight)
        output = torch.spmm(adj, support)
        return output


class GCN(nn.Module):
    def __init__(self, args, n_in, n_in_csd, n_h, dropout, csd_matrix): # n_h: class number
        super(GCN, self).__init__()

        self.fc1 = nn.Linear(n_in_csd, n_in, bias=True)

        self.gcn1 = GNNLayer(n_in, args.hidden_dim1)
        self.gcn_mid = GNNLayer(args.hidden_dim1, args.hidden_dimm)
        self.gcn2 = GNNLayer(args.hidden_dimm, args.hidden_dim2)

        self.dropout = dropout
        self.act = nn.ReLU()

        self.prototype_res = CosineLayer(args.hidden_dim2, n_h)


    def forward(self, X, S_X, csd_matrix, csd_matrix_adj):

        n_1 = self.gcn1(X, S_X)
        n_1d = F.dropout(n_1, p=self.dropout, training=self.training)
        n_mid = self.gcn_mid(n_1d, S_X)
        n_midd = F.dropout(n_mid, p=self.dropout, training=self.training) 
        n_2 = self.gcn2(n_midd, S_X)
        z1 = n_2

        l_1 = self.gcn1(self.fc1(csd_matrix), csd_matrix_adj)
        l_1d = F.dropout(l_1, p=self.dropout, training=self.training)
        l_mid = self.gcn_mid(l_1d, csd_matrix_adj)
        l_midd = F.dropout(l_mid, p=self.dropout, training=self.training)
        l_2 = self.gcn2(l_midd, csd_matrix_adj)
        z2 = l_2

        _ = self.prototype_res(z2)
        img_w = self.prototype_res.weightp

        return z1, z2, img_w
    