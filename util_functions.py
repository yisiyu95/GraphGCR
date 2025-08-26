import os
import numpy as np
import scipy.sparse as sp
import torch
import random
import torch.nn.functional as F
from collections import defaultdict


def setup(args):

    if args.dataset == 'C-M10-M':
        if [args.train_class, args.val_class] == [3,0]: 
            args.lr = 1e-3
            args.wd = 1e-5
            args.n_epochs = 1000
            args.hidden_dim1 = 128
            args.hidden_dimm = 64
            args.hidden_dim2 = 32
            args.n_neighbors = 5
            args.topk = 10
            args.para1 = 10
            args.para2 = 10
            args.weight_con = 1
            args.weight_proto = 1
            args.weight_add = 1
        else: # [2,2]
            args.lr = 1e-3
            args.wd = 1e-5
            args.n_epochs = 1000
            args.hidden_dim1 = 64
            args.hidden_dimm = 32
            args.hidden_dim2 = 32
            args.n_neighbors = 5
            args.topk = 100
            args.para1 = 10
            args.para2 = 10
            args.weight_con = 1
            args.weight_proto = 1
            args.weight_add = 1

    elif args.dataset == 'cora':
        if [args.train_class, args.val_class] == [3,0]: 
            args.lr = 1e-3
            args.wd = 1e-5
            args.n_epochs = 1000
            args.hidden_dim1 = 128
            args.hidden_dimm = 64
            args.hidden_dim2 = 64
            args.n_neighbors = 2
            args.topk = 100
            args.para1 = 10
            args.para2 = 10
            args.weight_con = 1
            args.weight_proto = 0.1
            args.weight_add = 1
        else: # [2,2]
            args.lr = 1e-4
            args.wd = 1e-5
            args.n_epochs = 1500
            args.hidden_dim1 = 256
            args.hidden_dimm = 128
            args.hidden_dim2 = 64
            args.n_neighbors = 5
            args.topk = 100
            args.para1 = 10
            args.para2 = 10
            args.weight_con = 1
            args.weight_proto = 0.1
            args.weight_add = 1

    elif args.dataset == 'citeseer':
        if [args.train_class, args.val_class] == [2,0]: 
            args.lr = 5e-4
            args.wd = 1e-8
            args.n_epochs = 1700
            args.hidden_dim1 = 256
            args.hidden_dimm = 128
            args.hidden_dim2 = 64
            args.n_neighbors = 5
            args.topk = 100
            args.para1 = 1
            args.para2 = 1
            args.weight_con = 1
            args.weight_proto = 0.01
            args.weight_add = 0.5
        else: # [2,2]
            args.lr = 1e-4
            args.wd = 1e-5
            args.n_epochs = 2000
            args.hidden_dim1 = 256
            args.hidden_dimm = 128
            args.hidden_dim2 = 64
            args.n_neighbors = 5
            args.topk = 100
            args.para1 = 1
            args.para2 = 1
            args.weight_con = 1
            args.weight_proto = 0.01
            args.weight_add = 0.5

    else:
        print("error!")
        print("please add the new dataset's parameters")
        print("------------------------------")
        print("lr: ")
        print("hidden_dim1: ")
        print("hidden_dimm: ")
        print("hidden_dim2: ")
        print("n_neighbors: ")
        print("topk: ")
        print("weight_con: ")
        print("weight_proto: ")
        print("weight_add: ")
        print("------------------------------")
        exit(0)


def get_adj_raw_feat(G):
    features = row_normalize(G)
    features = torch.from_numpy(features)
    print(features.shape, type(features))
    return features


def load_data_set(dataset):
    filepath = 'datasets'
    label_file = os.path.join(filepath, '{}/group.txt'.format(dataset))
    edge_file = os.path.join(filepath, '{}/graph.txt'.format(dataset))
    feature_file = os.path.join(filepath, '{}/feature.txt'.format(dataset))
    csd_file = os.path.join(filepath, 'csd_files/{}_text_csds.txt'.format(dataset))
    # csd_file = os.path.join(filepath, 'csd_files/{}_label_csds.txt'.format(dataset))

    idx, labellist = read_node_label(label_file)
    G = read_graph_as_matrix(nodeids=idx, edge_file=edge_file)
    features = np.genfromtxt(feature_file, dtype=np.float)[:, 1:]
    if (('M10-M' in dataset) is False):
        features = row_normalize(features)
    csd_matrix = get_csd_matrix(csd_file)

    return idx, labellist, G, torch.FloatTensor(features), csd_matrix


def get_csd_matrix(csd_file):
    csdmatrix = np.loadtxt(csd_file)  # [c, csd_dim]
    csdmatrix = torch.FloatTensor(csdmatrix)
    csdmatrix = F.normalize(csdmatrix, p=2, dim=1)
    return csdmatrix


def dot_sim(x, y):
    # Inner product similarity
    ip_sim = torch.mm(x, y)
    return ip_sim


def get_data_split(c_train, c_val, idx, labellist):
    '''Input: 
        idx: list[n, 1]
        labellist: list[n, string]
    Return:
            train_list: [num_train_samples, 1]
            val_list: [num_val_samples, 1]
            test_list: [num_test_samples, 1]
            total_class: num_class
    '''
    label_list_dict = defaultdict(list)
    for x, labels in zip(idx, labellist):
        for y in labels:
            label_list_dict[int(y)].append(int(x))

    train_list = []
    val_list = []
    test_list = []
    for i in label_list_dict.keys():
        if i < c_train:
            train_list = train_list + label_list_dict[i]
        elif c_train <= i < (c_train + c_val):
            val_list = val_list + label_list_dict[i]
        else:
            test_list = test_list + label_list_dict[i]
    return train_list, test_list, val_list


def get_data_split_gzsl(c_train, c_val, idx, labellist):
    """

    Args:
        c_train: list, shape=[n]
        c_val:
        idx:
        labellist:

    Returns:
        train_list: list, shape=[l]
        test_unseen_list: list, shape=[u_unseen]
        test_seen_list:  list, shape=[u_seen]
        u = u_unseen + u_seen
        n = l + u
    """
    label_list_dict = defaultdict(list)
    for x, labels in zip(idx, labellist):
        for y in labels:
            label_list_dict[int(y)].append(int(x))

    train_list = []
    val_list = []
    test_seen_list = []
    test_unseen_list = []
    for i in label_list_dict.keys():
        if i < c_train:
            class_i_sam_indices = label_list_dict[i]
            np.random.shuffle(class_i_sam_indices)
            class_i_sam_num = len(class_i_sam_indices)
            train_num = int(class_i_sam_num * 0.8)
            train_list = train_list + class_i_sam_indices[0: train_num]
            test_seen_list = test_seen_list + class_i_sam_indices[train_num: class_i_sam_num]
        elif c_train <= i < (c_train + c_val):
            val_list = val_list + label_list_dict[i]
        else:
            test_unseen_list = test_unseen_list + label_list_dict[i]
    return train_list, test_unseen_list, test_seen_list, val_list


def get_acc(pred, label, c_train, c_val, model):
    mypred = torch.ones(pred.shape) * float('-inf')
    if (model == 'train'):
        mypred[:, :c_train] = pred[:, :c_train]
    elif model == 'val':
        mypred[:, c_train: c_train + c_val] = pred[:, c_train: c_train + c_val]
    elif model == 'test':
        mypred[:, c_train + c_val:] = pred[:, c_train + c_val:]
    elif model == 'all':
        mypred = pred
    return get_acc_basic(mypred, label)


def get_acc_gzsl(predict, label, class_index):
    """

    Args:
        predict: tensor, shape=[u, c]
        label: tensor, shape=[u], {0, 1, 2, ..., c-1}
        class_index:  tensor, shape=[c_], subset of {0, 1, 2, ..., c-1}

    Returns:

    """
    # assume the c_train, c_val, c_test are ranked according to their c_ids
    predict = torch.argmax(predict, dim=1)
    acc_per_class = 0
    for i in class_index:
        idx = (predict == i)
        is_right = label[idx] == predict[idx]
        is_right = is_right.float()
        acc_per_class += torch.mean(is_right)
    acc_per_class /= class_index.shape[0]
    return acc_per_class


def get_acc_basic(predict, label):
    predict = torch.argmax(predict, dim=1)
    acc = (label.cpu() == predict)
    result = acc.cpu().sum().numpy()
    return result / len(acc)


# -------------------------------------
def read_node_label(filename):
    # print(os.getcwd())
    fin = open(filename, 'r')
    X = []
    Y = []
    while 1:
        l = fin.readline()
        if l == '':
            break
        vec = l.strip().split()
        X.append(vec[0])
        Y.append(vec[1:])
    fin.close()
    return X, Y


def symmetrize(adj):
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    ####adj = np.maximum(adj, adj.transpose() )
    return adj.todense()


def read_graph_as_matrix(nodeids, edge_file):
    ''' Read a symmetric adjacency matrix from a file
        Input: nodeids: [1,2,3,4,...]
        Return: the sparse adjacency matrix
    '''
    idx = np.array(nodeids, dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(edge_file, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(len(idx), len(idx)),
                        dtype=np.float32)
    print('origial input G', type(adj), sp.coo_matrix.count_nonzero(adj))
    # build symmetric adjacency matrix
    adj = symmetrize(adj)
    # print('symmetrice adj type', type(adj))
    return adj


def symmetric_normalize_adj(adj):
    """Symmetrically normalize adjacency matrix: D^{-1/2}AD^{-1/2}"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_normalize(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)


def use_cuda():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device


def rbf(dist, t=1.0):
    return np.exp(-(dist / t))


def cal_pairwise_dist(x):
    sum_x = np.sum(np.square(x), 1)
    dist = np.add(np.add(-2 * np.dot(x, x.T), sum_x).T, sum_x)
    return dist


def cal_rbf_dist(data, n_neighbors=1, t=1.0):
    """
    Args:
        data: [num_sample, num_feature]
        n_neighbors:
        t:
    Returns:

    """
    dist = cal_pairwise_dist(data)
    n = dist.shape[0]
    rbf_dist = rbf(dist, t)
    W = np.zeros((n, n))
    for i in range(n):
        index_ = np.argsort(dist[i])[1:1 + n_neighbors]
        W[i, index_] = 1
        W[index_, i] = 1
    return W
