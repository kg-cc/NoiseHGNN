import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.nn.pytorch import GraphConv

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv
from torch.nn import Sequential, Linear, ReLU

from conv import myGATConv_plus


class my_weight_GAT(nn.Module):
    def __init__(self,
                 g,
                 edge_dim,
                 num_etypes,
                 in_dims,
                 num_hidden,
                 num_classes,
                 num_layers,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual,
                 alpha):
        super(my_weight_GAT, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        self.fc_list = nn.ModuleList([nn.Linear(in_dim, num_hidden, bias=True) for in_dim in in_dims])

        for fc in self.fc_list:
            nn.init.xavier_normal_(fc.weight, gain=1.414)
        # input projection (no residual)
        self.gat_layers.append(myGATConv_plus(edge_dim, num_etypes,
                                              num_hidden, num_hidden, heads[0],
                                              feat_drop, attn_drop, negative_slope, False, self.activation,
                                              alpha=alpha))

        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(myGATConv_plus(edge_dim, num_etypes,
                                                  num_hidden * heads[l - 1], num_hidden, heads[l],
                                                  feat_drop, attn_drop, negative_slope, residual, self.activation,
                                                  alpha=alpha))

        # output projection
        self.gat_layers.append(myGATConv_plus(edge_dim, num_etypes,
                                              num_hidden * heads[-2], num_classes, heads[-1],
                                              feat_drop, attn_drop, negative_slope, residual, None, alpha=alpha))

        self.p_mlp_A = nn.Linear(num_hidden * heads[-2], 512, bias=True)

        self.epsilon = torch.FloatTensor([1e-12]).cuda()

        proj_dim = 64
        self.proj_head = Sequential(Linear(num_hidden * heads[-2], proj_dim), ReLU(inplace=True),
                                    Linear(proj_dim, proj_dim))

    def get_mlp_feature(self, features_list):
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)
        return h

    def get_feat_mask(self, features, mask_rate):
        feat_node = features.shape[1]
        mask = torch.zeros(features.shape)
        samples = np.random.choice(feat_node, size=int(feat_node * mask_rate), replace=False)
        mask[:, samples] = 1
        return mask.cuda(), samples

    def forward(self, features_list, e_feat, g, p=0.5):
        # target_idx = self.dl.nodes['target_idx']
        h = []
        for fc, feature in zip(self.fc_list, features_list):
            h.append(fc(feature))
        h = torch.cat(h, 0)

        if p > 0:
            mask_v1, _ = self.get_feat_mask(h, p)
            h = h * (1 - mask_v1)

        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](g, h, e_feat, res_attn=res_attn)
            h = h.flatten(1)

        # output projection
        logits, _ = self.gat_layers[-1](g, h, e_feat, res_attn=None)
        logits = logits.mean(1)
        # This is an equivalent replacement for tf.l2_normalize, see https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/math/l2_normalize for more information.

        logits = logits / (torch.max(torch.norm(logits, dim=1, keepdim=True), self.epsilon))
        h = self.proj_head(h)
        return logits, h

    def get_P_u_v(self, h):
        h = self.p_mlp_A(h)
        return h
