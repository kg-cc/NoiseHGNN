import dgl
import torch
import torch.nn as nn
import dgl.function as fn
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
import torch.nn.functional as F

import scipy.sparse as sp

from sklearn.neighbors import kneighbors_graph

# from layers import Attentive, GCNConv_dense, GCNConv_dgl
from utils import *

EOS = 1e-10


def apply_non_linearity(tensor, non_linearity, i):
    if non_linearity == 'elu':
        return F.elu(tensor * i - i) + 1
    elif non_linearity == 'relu':
        return F.relu(tensor)
    elif non_linearity == 'none':
        return tensor
    else:
        raise NameError('We dont support the non-linearity yet')


def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups]
    count = len(init_list) % batch_size
    end_list.append(init_list[-count:]) if count != 0 else end_list
    return end_list


def nearest_neighbors(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    return adj


def nearest_neighbors_sparse(X, k, metric):
    adj = kneighbors_graph(X, k, metric=metric)
    loop = np.arange(X.shape[0])
    [s_, d_, val] = sp.find(adj)
    s = np.concatenate((s_, loop))
    d = np.concatenate((d_, loop))
    return s, d


def nearest_neighbors_pre_exp(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def nearest_neighbors_pre_elu(X, k, metric, i):
    adj = kneighbors_graph(X, k, metric=metric)
    adj = np.array(adj.todense(), dtype=np.float32)
    adj += np.eye(adj.shape[0])
    adj = adj * i - i
    return adj


def normalize(adj, mode, sparse=False):
    # num_nodes = adj.size(0)
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":

            inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def normalize_pubmed_nc(adj, mode, sparse=False):
    num_nodes = adj.size(0)
    if not sparse:
        if mode == "sym":
            inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
            return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
        elif mode == "row":
            inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
            return inv_degree[:, None] * adj
        else:
            exit("wrong norm mode")
    else:
        adj = adj.coalesce()
        if mode == "sym":
            degree = torch.sparse.sum(adj, dim=1).to_dense()
            inv_sqrt_degree = torch.zeros(num_nodes)
            nonzero_mask = degree > 0
            inv_sqrt_degree[nonzero_mask] = 1.0 / torch.sqrt(degree[nonzero_mask])
            D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

            # inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()))
            # D_value = inv_sqrt_degree[adj.indices()[0]] * inv_sqrt_degree[adj.indices()[1]]

        elif mode == "row":
            aa = torch.sparse.sum(adj, dim=1)
            bb = aa.values()
            inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
            D_value = inv_degree[adj.indices()[0]]
        else:
            exit("wrong norm mode")
        new_values = adj.values() * D_value

        return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


# def normalize(adj, mode, sparse=False):
#     if not sparse:
#         if mode == "sym":
#             inv_sqrt_degree = 1. / (torch.sqrt(adj.sum(dim=1, keepdim=False)) + EOS)
#             return inv_sqrt_degree[:, None] * adj * inv_sqrt_degree[None, :]
#         elif mode == "row":
#             inv_degree = 1. / (adj.sum(dim=1, keepdim=False) + EOS)
#             return inv_degree[:, None] * adj
#         else:
#             exit("wrong norm mode")
#     else:
#         adj = adj.coalesce()
#         if mode == "sym":
#             inv_sqrt_degree = 1. / (torch.sqrt(torch.sparse.sum(adj, dim=1).values()) + EOS)
#             indices = adj.indices()
#             values = adj.values()
#
#             max_index_0 = torch.max(indices[0]).item()
#             max_index_1 = torch.max(indices[1]).item()
#
#             if max_index_0 >= len(inv_sqrt_degree) or max_index_1 >= len(inv_sqrt_degree):
#                 raise IndexError("Index out of range in normalization process")
#
#             D_value = inv_sqrt_degree[indices[0]] * inv_sqrt_degree[indices[1]]
#
#         elif mode == "row":
#             aa = torch.sparse.sum(adj, dim=1)
#             bb = aa.values()
#             inv_degree = 1. / (torch.sparse.sum(adj, dim=1).values() + EOS)
#             D_value = inv_degree[adj.indices()[0]]
#         else:
#             exit("wrong norm mode")
#         new_values = adj.values() * D_value
#
#         return torch.sparse.FloatTensor(adj.indices(), new_values, adj.size())


def symmetrize(adj):  # only for non-sparse
    return (adj + adj.T) / 2


def cal_similarity_graph(node_embeddings):
    similarity_graph = torch.mm(node_embeddings, node_embeddings.t())
    return similarity_graph


def top_k(raw_graph, K):
    values, indices = raw_graph.topk(k=int(K), dim=-1)
    assert torch.max(indices) < raw_graph.shape[1]
    mask = torch.zeros(raw_graph.shape).cuda()
    mask[torch.arange(raw_graph.shape[0]).view(-1, 1), indices] = 1.

    mask.requires_grad = False
    sparse_graph = raw_graph * mask
    return sparse_graph


def knn_fast(X, k, b):
    X = F.normalize(X, dim=1, p=2)
    index = 0
    values = torch.zeros(X.shape[0] * (k + 1)).cuda()
    rows = torch.zeros(X.shape[0] * (k + 1)).cuda()
    cols = torch.zeros(X.shape[0] * (k + 1)).cuda()
    norm_row = torch.zeros(X.shape[0]).cuda()
    norm_col = torch.zeros(X.shape[0]).cuda()
    while index < X.shape[0]:
        if (index + b) > (X.shape[0]):
            end = X.shape[0]
        else:
            end = index + b
        sub_tensor = X[index:index + b]
        similarities = torch.mm(sub_tensor, X.t())
        vals, inds = similarities.topk(k=k + 1, dim=-1)
        values[index * (k + 1):(end) * (k + 1)] = vals.view(-1)
        cols[index * (k + 1):(end) * (k + 1)] = inds.view(-1)
        rows[index * (k + 1):(end) * (k + 1)] = torch.arange(index, end).view(-1, 1).repeat(1, k + 1).view(-1)
        norm_row[index: end] = torch.sum(vals, dim=1)
        norm_col.index_add_(-1, inds.view(-1), vals.view(-1))
        index += b
    norm = norm_row + norm_col
    rows = rows.long()
    cols = cols.long()
    values *= (torch.pow(norm[rows], -0.5) * torch.pow(norm[cols], -0.5))
    return rows, cols, values


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def torch_sparse_to_dgl_graph(torch_sparse_mx):
    torch_sparse_mx = torch_sparse_mx.coalesce()
    indices = torch_sparse_mx.indices()
    values = torch_sparse_mx.values()
    rows_, cols_ = indices[0, :], indices[1, :]
    dgl_graph = dgl.graph((rows_, cols_), num_nodes=torch_sparse_mx.shape[0], device='cuda')
    dgl_graph.edata['w'] = values.detach().cuda()
    return dgl_graph


def dgl_graph_to_torch_sparse(dgl_graph):
    values = dgl_graph.edata['w'].cpu().detach()
    rows_, cols_ = dgl_graph.edges()
    indices = torch.cat((torch.unsqueeze(rows_, 0), torch.unsqueeze(cols_, 0)), 0).cpu()
    torch_sparse_mx = torch.sparse.FloatTensor(indices, values)
    return torch_sparse_mx


def torch_sparse_eye(num_nodes):
    indices = torch.arange(num_nodes).repeat(2, 1)
    values = torch.ones(num_nodes)
    return torch.sparse.FloatTensor(indices, values)


class GCNConv_dense(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dense, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def init_para(self):
        self.linear.reset_parameters()

    def forward(self, input, A, sparse=False):
        hidden = self.linear(input)
        if sparse:
            output = torch.sparse.mm(A, hidden)
        else:
            output = torch.matmul(A, hidden)
        return output


class GCNConv_dgl(nn.Module):
    def __init__(self, input_size, output_size):
        super(GCNConv_dgl, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x, g):
        with g.local_scope():
            g.ndata['h'] = self.linear(x)
            g.update_all(fn.u_mul_e('h', 'w', 'm'), fn.sum(msg='m', out='h'))
            return g.ndata['h']


class Attentive(nn.Module):
    def __init__(self, isize):
        super(Attentive, self).__init__()
        self.w = nn.Parameter(torch.ones(isize))

    def forward(self, x):
        return x @ torch.diag(self.w)


class SparseDropout(nn.Module):
    def __init__(self, dprob=0.5):
        super(SparseDropout, self).__init__()
        # dprob is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - dprob

    def forward(self, x):
        mask = ((torch.rand(x._values().size()) + (self.kprob)).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)


class MLP_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, act):
        super(MLP_learner, self).__init__()

        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(nn.Linear(isize, isize))
        else:
            self.layers.append(nn.Linear(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(nn.Linear(isize, isize))
            self.layers.append(nn.Linear(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.act = act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h)
            if i != (len(self.layers) - 1):
                if self.act == "relu":
                    h = F.relu(h)
                elif self.act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            nn.init.xavier_normal_(layer.weight, gain=1.414)
            # layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities


class GNN_learner(nn.Module):
    def __init__(self, nlayers, isize, k, knn_metric, i, sparse, mlp_act, adj):
        super(GNN_learner, self).__init__()

        self.adj = adj
        self.layers = nn.ModuleList()
        if nlayers == 1:
            self.layers.append(GCNConv_dgl(isize, isize))
        else:
            self.layers.append(GCNConv_dgl(isize, isize))
            for _ in range(nlayers - 2):
                self.layers.append(GCNConv_dgl(isize, isize))
            self.layers.append(GCNConv_dgl(isize, isize))

        self.input_dim = isize
        self.output_dim = isize
        self.k = k
        self.knn_metric = knn_metric
        self.non_linearity = 'relu'
        self.param_init()
        self.i = i
        self.sparse = sparse
        self.mlp_act = mlp_act

    def internal_forward(self, h):
        for i, layer in enumerate(self.layers):
            h = layer(h, self.adj)
            if i != (len(self.layers) - 1):
                if self.mlp_act == "relu":
                    h = F.relu(h)
                elif self.mlp_act == "tanh":
                    h = F.tanh(h)
        return h

    def param_init(self):
        for layer in self.layers:
            layer.weight = nn.Parameter(torch.eye(self.input_dim))

    def forward(self, features):
        if self.sparse:
            embeddings = self.internal_forward(features)
            rows, cols, values = knn_fast(embeddings, self.k, 1000)
            rows_ = torch.cat((rows, cols))
            cols_ = torch.cat((cols, rows))
            values_ = torch.cat((values, values))
            values_ = apply_non_linearity(values_, self.non_linearity, self.i)
            adj = dgl.graph((rows_, cols_), num_nodes=features.shape[0], device='cuda')
            adj.edata['w'] = values_
            return adj
        else:
            embeddings = self.internal_forward(features)
            embeddings = F.normalize(embeddings, dim=1, p=2)
            similarities = cal_similarity_graph(embeddings)
            similarities = top_k(similarities, self.k + 1)
            similarities = apply_non_linearity(similarities, self.non_linearity, self.i)
            return similarities
