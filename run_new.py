import copy
import random
import shutil
import sys
from pathlib import Path

sys.path.append('../../')
sys.path.append('../')
import time
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

from utils.pytorchtools import EarlyStopping
from utils.data import load_data

from GNN import my_weight_GAT
import dgl

from utils.tools import sparse_mx_to_torch_sparse_tensor, torch_sparse_to_dgl_graph, dgl_graph_to_torch_sparse, \
    get_mata_A, calc_loss, symmetrize, graph_structure_loss, set_seed
from graph_generator import normalize, MLP_learner, GNN_learner, normalize_pubmed_nc


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)

    else:
        pass


def seed_torch(seed=0):
    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


def sp_to_spt(mat):
    coo = mat.tocoo()
    values = coo.data
    indices = np.vstack((coo.row, coo.col))

    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def mat2tensor(mat):
    if type(mat) is np.ndarray:
        return torch.from_numpy(mat).type(torch.FloatTensor)
    return sp_to_spt(mat)


def run_model_DBLP(args):
    local_time = time.localtime()
    args.lct = time.strftime("%Y_%m_%d_%H_%M_%S", local_time)
    feats_type = args.feats_type
    features_list, adjM, labels, train_val_test_idx, dl, target_idx = load_data(args, args.dataset)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    features_list = [mat2tensor(features).to(device) for features in features_list]
    if feats_type == 0:
        in_dims = [features.shape[1] for features in features_list]
    elif feats_type == 1 or feats_type == 5:
        save = 0 if feats_type == 1 else 2
        in_dims = []  # [features_list[0].shape[1]] + [10] * (len(features_list) - 1)
        for i in range(0, len(features_list)):
            if i == save:
                in_dims.append(features_list[i].shape[1])
            else:
                in_dims.append(10)
                features_list[i] = torch.zeros((features_list[i].shape[0], 10)).to(device)
    elif feats_type == 2 or feats_type == 4:
        save = feats_type - 2
        in_dims = [features.shape[0] for features in features_list]
        for i in range(0, len(features_list)):
            if i == save:
                in_dims[i] = features_list[i].shape[1]
                continue
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    elif feats_type == 3:
        in_dims = [features.shape[0] for features in features_list]
        for i in range(len(features_list)):
            dim = features_list[i].shape[0]
            indices = np.vstack((np.arange(dim), np.arange(dim)))
            indices = torch.LongTensor(indices)
            values = torch.FloatTensor(np.ones(dim))
            features_list[i] = torch.sparse.FloatTensor(indices, values, torch.Size([dim, dim])).to(device)
    labels = torch.LongTensor(labels).to(device)
    train_idx = train_val_test_idx['train_idx']
    train_idx = np.sort(train_idx)
    val_idx = train_val_test_idx['val_idx']
    val_idx = np.sort(val_idx)
    test_idx = train_val_test_idx['test_idx']
    test_idx = np.sort(test_idx)

    # ----------------------------------------------------------------------------------------------------------------------------------------- ------------------- #
    # todo anchor_A
    adjM = sparse_mx_to_torch_sparse_tensor(adjM + adjM.T)
    if args.dataset == "Freebase":
        anchor_adj = torch_sparse_to_dgl_graph(adjM)
        anchor_adj.to(device)
        anchor_adj = dgl.remove_self_loop(anchor_adj)
        anchor_adj = dgl.add_self_loop(anchor_adj)
    elif args.dataset == "PubMed_NC":
        anchor_adj = normalize_pubmed_nc(adjM, 'sym', args.sparse)
        anchor_adj.to(device)
        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)
        anchor_adj = dgl.remove_self_loop(anchor_adj)
        anchor_adj = dgl.add_self_loop(anchor_adj)
    else:
        if args.sparse:
            anchor_adj_raw = adjM
        #  post anchor
        anchor_adj = normalize(anchor_adj_raw, 'sym', args.sparse)
        anchor_adj.to(device)
        anchor_adj = torch_sparse_to_dgl_graph(anchor_adj)

    mata_g1, mata_g2, mata_g3 = get_mata_A(args, dl)

    accs_micro = []
    accs_macro = []

    for r in range(args.repeat):
        seed_torch(r)

        output_dir = Path.cwd().joinpath(
            args.output_path,
            "noiseHGNN",
            args.dataset,
            f"seed_{r}")

        args.output_dir = output_dir

        check_writable(output_dir, False)

        num_classes = dl.labels_train['num_classes']
        heads = [args.num_heads] * args.num_layers + [1]

        net = my_weight_GAT(anchor_adj, args.edge_feats, len(dl.links['count']) * 2 + 1, in_dims, args.hidden_dim,
                            num_classes,
                            args.num_layers, heads, F.elu, args.dropout, args.dropout, args.slope, True, 0.05)

        net.to(device)
        features = net.get_mlp_feature(features_list)
        if args.graph_learner == "MLP":
            graph_learner = MLP_learner(3, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                        args.activation_learner)
        elif args.graph_learner == "GCN":
            graph_learner = GNN_learner(3, features.shape[1], args.k, args.sim_function, 6, args.sparse,
                                        args.activation_learner, anchor_adj)

        graph_learner = graph_learner.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizer_learner = torch.optim.Adam(graph_learner.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # training loop
        net.train()
        graph_learner.train()
        early_stopping = EarlyStopping(patience=args.patience, verbose=True,
                                       save_path='{}_{}_{}.pt'.format(args.output_dir, args.dataset, args.num_layers))

        for epoch in range(args.epoch):
            t_start = time.time()
            # training
            net.train()
            graph_learner.train()

            Adj = copy.deepcopy(anchor_adj)
            Adj.edata['w'] = F.dropout(Adj.edata['w'], p=args.anchor_A_mask_rate, training=True)
            logits_1, h1 = net(features_list, None, Adj, args.anchor_feat_mask_rate)

            features = net.get_mlp_feature(features_list)
            learned_adj = graph_learner(features)
            if not args.sparse:
                learned_adj = symmetrize(learned_adj)
                learned_adj = normalize(learned_adj, 'sym', args.sparse)
            learn_Adj = learned_adj
            learn_Adj.edata['w'] = F.dropout(learn_Adj.edata['w'], args.learn_A_mask_rate, True)
            logits_2, h2 = net(features_list, None, learn_Adj, args.learn_feat_mask_rate)
            logp_2 = F.log_softmax(logits_2, 1)

            logp_1 = F.log_softmax(logits_1, 1)
            train_loss_1 = F.nll_loss(logp_1[train_idx], labels[train_idx])
            train_loss_2 = F.nll_loss(logp_2[train_idx], labels[train_idx])
            learned_A = dgl.node_subgraph(learned_adj, target_idx)
            anchor_A = dgl.node_subgraph(anchor_adj, target_idx)
            mata_A = mata_g1

            # loss_sce = graph_structure_loss(learned_A, anchor_A)
            loss_sce = graph_structure_loss(learned_A, mata_A)

            if args.loss_ce and args.loss_sce:
                train_loss = train_loss_1 + + train_loss_2 + loss_sce
            elif args.loss_ce:
                train_loss = train_loss_1 + + train_loss_2
            elif args.loss_sce:
                train_loss = train_loss_1 + + loss_sce
            else:
                train_loss = train_loss_1
            # autograd
            optimizer.zero_grad()
            optimizer_learner.zero_grad()
            train_loss.backward()
            optimizer.step()
            optimizer_learner.step()
            t_end = time.time()

            # print training info
            print('Epoch {:05d} | Train_Loss: {:.4f} | Time: {:.4f}'.format(epoch, train_loss.item(), t_end - t_start))

            t_start = time.time()
            # validation
            net.eval()
            with torch.no_grad():
                # logits, _ = net(features_list, e_feat, anchor_adj, 0, "anchor")
                logits, h = net(features_list, None, anchor_adj, p=0)
                logp = F.log_softmax(logits, 1)
                val_loss = F.nll_loss(logp[val_idx], labels[val_idx])
            t_end = time.time()
            # print validation info
            print('Epoch {:05d} | Val_Loss {:.4f} | Time(s) {:.4f}'.format(
                epoch, val_loss.item(), t_end - t_start))
            # early stopping
            early_stopping(val_loss, net)
            if early_stopping.early_stop:
                print('Early stopping!')
                break

        # testing with evaluate_results_nc
        # net.load_state_dict(torch.load('checkpoint/checkpoint_{}_{}.pt'.format(args.dataset, args.num_layers)))
        net.load_state_dict(torch.load('{}_{}_{}.pt'.format(args.output_dir, args.dataset, args.num_layers)))
        net.eval()
        with torch.no_grad():
            logits, h = net(features_list, None, anchor_adj, p=0)

            test_logits = logits[test_idx]
            pred = test_logits.cpu().numpy().argmax(axis=1)
            onehot = np.eye(num_classes, dtype=np.int32)
            dl.gen_file_for_evaluate(test_idx=test_idx, label=pred, file_name=f"{args.dataset}_{args.run}.txt")
            pred = onehot[pred]
            result = dl.evaluate(pred)
            print(result)
            accs_micro.append(result["micro-f1"])
            accs_macro.append(result["macro-f1"])
            with open(args.output_dir.parent.joinpath("exp_results"), "a+", encoding='utf-8') as f:
                f.write(
                    f"feat_mask_rate:{args.feat_mask_rate}   feat_drop:{args.feat_mask_rate}   edge_add:{args.edge_add}   edge_drop:{args.edge_drop}  edge_wrong:{args.edge_wrong} update_A:{args.use_update}  result:{result} \n")

    with open(args.output_dir.parent.joinpath("exp_results"), "a+", encoding='utf-8') as f:
        accs_micro = np.array(accs_micro)
        accs_macro = np.array(accs_macro)
        f.write(f"micro:{accs_micro.mean(axis=0)} + {accs_micro.std(axis=0)}\n")
        f.write(f"macro:{accs_macro.mean(axis=0)} + {accs_macro.std(axis=0)}\n")
        f.write(
            f"loss_type（1:kl  2:sce   3:ce）type:{args.loss_type}  use_update:{args.use_update}  update_epoch:{args.update_epoch}   \n")
        f.write(f"args:{args}\n")
        f.write(f"\n\n")


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='MRGNN testing for the DBLP dataset')
    ap.add_argument('--feats-type', type=int, default=3,
                    help='Type of the node features used. ' +
                         '0 - loaded features; ' +
                         '1 - only target node features (zero vec for others); ' +
                         '2 - only target node features (id vec for others); ' +
                         '3 - all id vec. Default is 2;' +
                         '4 - only term features (id vec for others);' +
                         '5 - only term features (zero vec for others).')
    ap.add_argument('--hidden-dim', type=int, default=64, help='Dimension of the node hidden state. Default is 64.')
    ap.add_argument('--num-heads', type=int, default=8, help='Number of the attention heads. Default is 8.')
    ap.add_argument('--epoch', type=int, default=300, help='Number of epochs.')
    ap.add_argument('--patience', type=int, default=30, help='Patience.')
    ap.add_argument('--num-layers', type=int, default=2)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--dropout', type=float, default=0.5)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--slope', type=float, default=0.05)
    ap.add_argument('--dataset', type=str)
    ap.add_argument('--edge-feats', type=int, default=64)
    ap.add_argument('--run', type=int, default=1)

    # ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    ap.add_argument("--alph", type=int, default=5, help="alph to sce loss")
    ap.add_argument("--threshold", type=float, default=0.30, help="how much threshold to update A")
    ap.add_argument("--use_update", type=int, default=1, help="use update A")
    ap.add_argument('--loss_type', type=int, default=1, help='A_d to A loss 1:sce 2:mse  3:abs')
    ap.add_argument('--feat_mask_rate', type=float, default=0)
    ap.add_argument('--edge_drop', type=float, default=0.0)
    ap.add_argument('--edge_add', type=float, default=0)
    ap.add_argument('--edge_wrong', type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0, help="Random seed")
    ap.add_argument("--output_path", type=str, default="outputs", help="Path to save outputs")
    ap.add_argument("--update_epoch", type=int, default=5, help="how much epoch update A")
    ap.add_argument('--repeat', type=int, default=5, help='Repeat the training and testing for N times. Default is 1.')

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------#
    ap.add_argument('-sparse', type=int, default=1)
    ap.add_argument('-sim_function', type=str, default='cosine', choices=['cosine', 'minkowski'])
    ap.add_argument('-activation_learner', type=str, default='relu', choices=["relu", "tanh"])
    ap.add_argument('-k', type=int, default=15)

    ap.add_argument('-graph_learner', type=str, default='MLP', choices=['MLP', 'GCN'])

    ap.add_argument('--anchor_feat_mask_rate', type=float, default=0.0)
    ap.add_argument('--learn_feat_mask_rate', type=float, default=0.0)

    ap.add_argument('--anchor_A_mask_rate', type=float, default=0.0)
    ap.add_argument('--learn_A_mask_rate', type=float, default=0.0)

    ap.add_argument('--loss_sce', action="store_true",
                    help="Set to True to include loss_A")
    ap.add_argument('--loss_ce', action="store_true",
                    help="Set to True to include loss_ce")

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------#

    args = ap.parse_args()

    output_dir = Path.cwd().joinpath(
        args.output_path,
        args.dataset,
        f"seed_{args.seed}")

    args.output_dir = output_dir
    check_writable(output_dir, overwrite=False)

    os.makedirs('checkpoint', exist_ok=True)
    print("====================args=====================")
    print(args)

    if args.dataset == "ACM":
        args.epoch = 300
        args.feats_type = 2
        args.patience = 50
        args.edge_wrong = 0.3
        args.k = 35
        args.anchor_A_mask_rate = 0.3
        args.learn_A_mask_rate = 0.4
        args.loss_kl = False
        args.loss_sce = True
        args.loss_ce = True
        args.graph_learner = "GCN"

    elif args.dataset == "DBLP":
        args.edge_wrong = 0.3
        args.feats_type = 3
        args.anchor_A_mask_rate = 0.3
        args.learn_A_mask_rate = 0.5
        args.k = 25
        args.loss_sce = True
        args.loss_ce = True
        args.graph_learner = "GCN"


    elif args.dataset == "Freebase":
        args.epoch = 500
        args.feats_type = 3
        args.edge_wrong = 0.3
        args.anchor_A_mask_rate = 0.3
        args.learn_A_mask_rate = 0.4
        args.k = 15
        args.lr = 0.0005
        args.loss_sce = True
        args.loss_ce = True
        args.graph_learner = "GCN"

    elif args.dataset == "PubMed_NC":
        args.epoch = 300
        args.lr = 0.005
        args.edge_wrong = 0.0
        args.anchor_A_mask_rate = 0.2
        args.learn_A_mask_rate = 0.2
        args.feats_type = 2
        args.k = 25
        args.loss_sce = True
        args.loss_ce = True
        args.graph_learner = "GCN"

    seed_torch(2024)
    run_model_DBLP(args)
