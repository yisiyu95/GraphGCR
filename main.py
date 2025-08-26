import os
import time
import json
import random
import numpy as np
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.distance import cdist

import copy
import scipy.sparse as sp
import torch_geometric.utils as tg_utils
from torch_geometric.data import Data
import torch_geometric.transforms as T

from GCN import GCN
from util_functions import get_data_split, get_acc, setup_seed, use_cuda, cal_rbf_dist
from util_functions import load_data_set, symmetric_normalize_adj, setup


def cos_sim(z1, z2, hidden_norm = True):
    if hidden_norm:
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())

def nei_con_loss(z1, z2, index_conn_train, label, topk, lossfn, hidden_norm = True):  
    sim_all = cos_sim(z1, z2, hidden_norm)
    sim_exd = 5 * sim_all[index_conn_train[:,1],:]
    label_exd = torch.repeat_interleave(label, topk)
    return lossfn(sim_exd, label_exd)

def solution_ours(z1, z2, img_w, index_conn_train, c_train, label, topk, lossfn, lam):

    Lcon = nei_con_loss(z1, z2, index_conn_train, label, topk, lossfn, hidden_norm = True)
    Lproto = nei_con_loss(z1, img_w, index_conn_train, label, topk, lossfn, hidden_norm = True)

    global_label = torch.from_numpy(np.array(range(2*c_train))).to(device)
    z2_sim = cos_sim(z2, z2, hidden_norm = True)
    img_w_sim = cos_sim(img_w, img_w, hidden_norm = True)
    sim_weight = F.softmax(torch.pow(args.para1 * (z2_sim + img_w_sim)/2, -1), dim=1)
    mixed_z2_wei = torch.mm(sim_weight, F.normalize(z2, p=2, dim=1))
    mixed_imgw_wei = torch.mm(sim_weight, F.normalize(img_w, p=2, dim=1))
    alpha = random.uniform(0,1)
    mixed_z2_ran = alpha*F.normalize(z2, p=2, dim=1) + (1-alpha)*mixed_z2_wei
    mixed_imgw_ran = alpha*F.normalize(img_w, p=2, dim=1) + (1-alpha)*mixed_imgw_wei
    mixed_z2 = torch.cat((mixed_z2_wei, mixed_z2_ran), dim=0)
    mixed_imgw = torch.cat((mixed_imgw_wei, mixed_imgw_ran), dim=0)
    cosine = args.para2 * F.linear(mixed_z2, mixed_imgw)
    Ladd = lossfn(cosine, global_label)
    # Ladd = 0

    return Lcon, Lproto, Ladd


def train(args):

    [c_train, c_val] = [args.train_class, args.val_class]
    idx, labellist, G, features, csd_matrix = load_data_set(args.dataset)
    csd_matrix_1nn_graph = cal_rbf_dist(data=csd_matrix.numpy(), n_neighbors=args.n_neighbors, t=1.0)
    G_ori = torch.tensor(data=G).to(device)
    G = symmetric_normalize_adj(G).todense()
    csd_matrix_1nn_graph = symmetric_normalize_adj(csd_matrix_1nn_graph).todense()

    idx_train, idx_test, idx_val = get_data_split(c_train=c_train, c_val=c_val, idx=idx, labellist=labellist)
    y_true = np.array([int(temp[0]) for temp in labellist])  # [n, 1]
    y_true = torch.from_numpy(y_true).type(torch.LongTensor).to(device)
    num_class = torch.unique(y_true).shape[0]

    G = torch.tensor(data=G).to(device)
    csd_matrix_1nn_graph = torch.tensor(data=csd_matrix_1nn_graph, dtype=torch.float32).to(device)
    csd_matrix = csd_matrix.to(device)
    print('csd_matrix size:', csd_matrix.size())
    features_ori = copy.deepcopy(features)
    features = features.to(device)
    result_dir = './result/'
    result_file = open(file=result_dir + f'GraphGCR_{args.dataset}' + str([args.train_class, args.val_class])
                        + '.txt', mode='w')

    model = GCN(args, n_in=features.shape[1], n_in_csd=csd_matrix.shape[1], n_h=num_class, dropout=args.dropout, csd_matrix=csd_matrix).to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    if [args.train_class, args.val_class] == [2, 2]:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.wd) # RMSprop
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd) # Adam
    lam = torch.rand((num_class,num_class)).to(device)

    tmp_coo = sp.coo_matrix(G_ori.cpu()) 
    edge_index, edge_attr = tg_utils.from_scipy_sparse_matrix(tmp_coo)
    data = Data(x=features_ori, edge_index=edge_index, edge_attr=edge_attr)
    print(f"Data: {data}")
    data_diffusion = T.GDC(sparsification_kwargs={'k': args.topk, 'dim': 1, 'method': 'topk'})(data)
    diffusion_idx = data_diffusion.edge_index[1].reshape(-1, args.topk)
    diff_node = torch.reshape(diffusion_idx[idx_train], (len(idx_train)*args.topk, -1)).to(device)
    rep_node = torch.unsqueeze(torch.repeat_interleave(torch.from_numpy(np.array(idx_train)).to(device), args.topk).to(device), dim=1)
    index_conn_train = torch.cat((rep_node, diff_node), dim=1)
    test_acc_list = []

    for epoch in range(args.n_epochs + 1):
        model.train()
        optimizer.zero_grad()
        z1, z2, img_w = model(X=features, S_X=G, csd_matrix=csd_matrix, csd_matrix_adj = csd_matrix_1nn_graph)
        [Lcon, Lproto, Ladd] = solution_ours(z1, z2[:c_train], img_w[:c_train], index_conn_train, c_train, y_true[idx_train], args.topk, criterion, lam) # [:,idx_train]
        loss_X = args.weight_con * Lcon + args.weight_proto * Lproto + args.weight_add * Ladd
        loss = loss_X
        loss.backward()
        optimizer.step()

        model.eval()
        z1, z2, _ = model(X=features, S_X=G, csd_matrix=csd_matrix, csd_matrix_adj = csd_matrix_1nn_graph)
        pred_label = F.linear(F.normalize(z1, p=2, dim=1), F.normalize(z2, p=2, dim=1))

        test_acc = get_acc(pred_label[idx_test], y_true[idx_test], c_train=c_train, c_val=c_val, model='test')
        val_acc =0
        if len(idx_val) > 0:
            val_acc = get_acc(pred_label[idx_val], y_true[idx_val], c_train=c_train, c_val=c_val, model='val')

        test_acc_list.append(test_acc.item())
        acc_mx = np.max(test_acc_list)
        if epoch % 10 == 0:
            print('Epoch: {:04d}'.format(epoch), 
                'val_acc: {:.4f}'.format(val_acc), 
                'test_acc: {:.4f}'.format(test_acc.item()), 
                'test_acc_max: {:.4f}'.format(acc_mx.item())
                ) 
            result_dict={
                'Epoch': epoch,
                'seed': args.seed, 
                'train-val-class': str([args.train_class,args.val_class]), 
                'val_acc': val_acc,
                'test_acc': test_acc,
                'test_acc_max': acc_mx
                }
            json_data = json.dumps(result_dict)
            print(json_data, file=result_file)
    result_file.close()


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(description='MODEL')

    parser.add_argument("--dataset", type=str, default='C-M10-M', choices=['cora', 'citeseer', 'C-M10-M'],
                        help="dataset")
    parser.add_argument("--train-class", type=int, default=3,
                        help="the #train_class")
    parser.add_argument("--val-class", type=int, default=0,
                        help="the #validation classes")
    parser.add_argument('--seed', type=int, default=0, 
                        help='Random seed.')
    parser.add_argument('--cuda', type=bool, default=True, 
                        help='GPU')
    parser.add_argument("--dropout", type=float, default=0.5, 
                        help="dropout probability")


    args = parser.parse_args()
    if args.seed is not None:
        setup_seed(args)
    device = use_cuda()
    setup(args)
    print(args)
    train(args)
