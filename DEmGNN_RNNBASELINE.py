#!/usr/bin/env python
# coding: utf-8

import argparse
import os

from modules.config import TaskMode, get_dataset_config

parser = argparse.ArgumentParser()
parser.add_argument('--VERSION', type=str, default='V2', help='V2 is DBLP. Other candidates: V1, IoT, RW, WIDE, wikipedia, reddit, lastfm')
parser.add_argument('--GNN_TYPE', type=str, default='WeightedGCN', help='GCN, SAGE, GAT, WeightedGCN')
parser.add_argument('--num_epochs', type=int, default=300, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--gnn_out_features_half', type=int, default=64, help='half of the output features of GNN')
parser.add_argument('--rnn_hidden_size', type=int, default=128, help='hidden size of RNN')
parser.add_argument('--num_rnn_layers', type=int, default=2, help='number of layers in RNN')
parser.add_argument('--rnn_dropout', type=float, default=0.3, help='dropout rate for RNN')
parser.add_argument('--gnn_dropout', type=float, default=0.1, help='dropout rate for GNN')
parser.add_argument('--combine_decay_factor', type=float, default=0.5, help='combine_decay_factor')
parser.add_argument('--HISTORY_LENGTH', type=int, default=0, help='number of history time steps to combine')
parser.add_argument('--NUM_RUNS', type=int, default=5, help='number of runs')
parser.add_argument('--gpu', type=str, default=None, help='primary cuda device index or device string (e.g., 0 or cuda:0)')
parser.add_argument('--gpu2', type=str, default=None, help='secondary cuda device index or device string (e.g., 1 or cuda:1)')
parser.add_argument('--data_base_path', type=str, default=None, help='base path for dataset and embedding files')
parser.add_argument('--task', type=int, default=0, choices=[0, 1], help='0: weighted edge prediction, 1: weighted graph completion')
args = parser.parse_args()

print(args)


def _normalize_device(device_arg: str, fallback: str) -> str:
    if device_arg is None:
        return fallback
    if device_arg == 'cpu' or device_arg.startswith('cuda'):
        return device_arg
    return f'cuda:{device_arg}'


device = 'cuda:0'
lr = args.lr
combine_decay_factor = args.combine_decay_factor
HISTORY_LENGTH = args.HISTORY_LENGTH
assert HISTORY_LENGTH == 0, 'HISTORY_LENGTH must be 0'
SEED = args.seed
NUM_RUNS = args.NUM_RUNS
GNN_TYPE = args.GNN_TYPE
print("Use", GNN_TYPE, end=',' )
if GNN_TYPE == 'WeightedGCN':
    pass
elif GNN_TYPE in ['GAT']:
    pass
elif GNN_TYPE in ['GCN', 'SAGE']:
    pass
else:
    raise ValueError('GNN_TYPE must be one of "GCN", "SAGE", "GAT", "WeightedGCN"')
num_epochs = args.num_epochs

VERSION = args.VERSION

dataset_config = get_dataset_config(VERSION, args.task, base_path=args.data_base_path)
sample_file = dataset_config['sample_file']
sample_file_completion = dataset_config['sample_file_completion']
data_path = dataset_config['data_path']
seq_length = dataset_config['seq_length']
valid_seq_length = dataset_config['valid_seq_length']
device = _normalize_device(args.gpu, dataset_config['device'])
device2 = _normalize_device(args.gpu2, dataset_config['device2'])
embedding_file = dataset_config['embedding_file']
embedding_multiple_times = dataset_config['embedding_multiple_times']
num_nodes = dataset_config.get('num_nodes')

visible_devices = []
for dev in (device, device2):
    if dev.startswith('cuda'):
        parts = dev.split(':', 1)
        visible_devices.append(parts[1] if len(parts) > 1 else '0')
if visible_devices:
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(dict.fromkeys(visible_devices))

class TASKS:
    WEIGHTED_EDGE_PREDICTION = TaskMode.WEIGHTED_EDGE_PREDICTION
    WEIGHTED_GRAPH_COMPLETION = TaskMode.WEIGHTED_GRAPH_COMPLETION

if args.task == TASKS.WEIGHTED_EDGE_PREDICTION:
    print('Weighted edge prediction')
elif args.task == TASKS.WEIGHTED_GRAPH_COMPLETION:
    print('Weighted graph completion')
else:
    raise ValueError('task must be one of 0, 1')

import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import MessagePassing
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
import json
import numpy as np

def set_random_seed(random_seed: int):
    r"""
    set random seed for reproducibility
    Args:
        random_seed (int): random seed
    """
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print(f'INFO: fixed random seed: {random_seed}')

class WeightedGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_time_steps):
        super(WeightedGCNConv, self).__init__(aggr='add')
        self.num_time_steps = num_time_steps

        self.linears = nn.ModuleList([nn.Linear(in_channels, out_channels) for _ in range(num_time_steps)])
        self.linear_out = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index, edge_attr_tuple):
        out = None

        for t in range(self.num_time_steps):
            edge_attr = edge_attr_tuple[:,:,t]
            linear = self.linears[t]

            if out is None:
                out = self.propagate(edge_index, x=x, edge_attr=edge_attr, linear=linear)
            else:
                out += self.propagate(edge_index, x=x, edge_attr=edge_attr, linear=linear)

        return out

    def message(self, x_j, edge_attr, linear):
        feats = linear(x_j)
        return edge_attr * feats.relu()

    def update(self, aggr_out):
        return self.linear_out(aggr_out)

class GNNLayer(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super(GNNLayer, self).__init__()
        if GNN_TYPE == 'GAT':
            self.conv1 = GATConv(in_channels, out_channels, edge_dim=1)
            self.conv2 = GATConv(out_channels, out_channels, edge_dim=1)
        elif GNN_TYPE == 'GCN':
            self.conv1 = GCNConv(in_channels, out_channels)
            self.conv2 = GCNConv(out_channels, out_channels)
        elif GNN_TYPE == 'SAGE':
            self.conv1 = SAGEConv(in_channels, out_channels)
            self.conv2 = SAGEConv(out_channels, out_channels)
        elif GNN_TYPE == 'WeightedGCN':
            self.conv1 = WeightedGCNConv(in_channels, out_channels, HISTORY_LENGTH + 1)
            self.conv2 = WeightedGCNConv(out_channels, out_channels, HISTORY_LENGTH + 1)
        else:
            raise ValueError('GNN_TYPE must be one of "GCN", "SAGE", "GAT", "WeightedGCN"')
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

    def forward(self, x, edge_index, edge_attr):

        if GNN_TYPE in ['GAT']:
            x = self.conv1(x, edge_index, edge_attr=edge_attr)
        elif GNN_TYPE in ['WeightedGCN']:
            x = self.conv1(x, edge_index, edge_attr_tuple=edge_attr)
        elif GNN_TYPE in ['GCN', 'SAGE']:
            x = self.conv1(x, edge_index)
        else:
            raise ValueError('GNN_TYPE must be one of "GCN", "SAGE", "GAT", "WeightedGCN"')
        x = self.norm1(x).relu()
        x = self.dropout1(x)

        if GNN_TYPE in ['GAT']:
            x = self.conv2(x, edge_index, edge_attr=edge_attr)
        elif GNN_TYPE in ['WeightedGCN']:
            x = self.conv2(x, edge_index, edge_attr_tuple=edge_attr)
        elif GNN_TYPE in ['GCN', 'SAGE']:
            x = self.conv2(x, edge_index)
        else:
            raise ValueError('GNN_TYPE must be one of "GCN", "SAGE", "GAT", "WeightedGCN"')
        x = self.norm2(x).relu()
        x = self.dropout2(x)

        return x

data_cache = {}

class BaselineRNNGNN(nn.Module):
    def __init__(self, node_features, gnn_out_features_half, rnn_hidden_size, num_rnn_layers=args.num_rnn_layers, rnn_dropout=args.rnn_dropout, gnn_dropout=args.gnn_dropout):
        super(BaselineRNNGNN, self).__init__()

        self.gnn_out_features = gnn_out_features_half * 2
        self.gnn = GNNLayer(node_features, self.gnn_out_features, gnn_dropout)

        self.rnn = nn.LSTM(
            input_size=self.gnn_out_features,
            hidden_size=rnn_hidden_size,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=rnn_dropout if num_rnn_layers > 1 else 0.0
        )

        self.RNN_output_transform = nn.Linear(rnn_hidden_size, self.gnn_out_features)

        self.predictor_edge = nn.Linear(self.gnn_out_features, 1)
        self.predictor_weight = nn.Linear(self.gnn_out_features, 1)

    def forward(self, graphs, seq_length, this_batch_data_size):
        """
        Args:
            graphs: List of PyG Data objects for each time step (length = seq_length)
            seq_length: int - number of time steps (sequence length)
            this_batch_data_size: int - number of nodes in the current batch

        Returns:
            predictions: Tensor - RNN predictions for each time step
            rnn_output: Tensor - RNN hidden state output
        """
        rnn_inputs = []

        for t in range(seq_length):
            if t not in data_cache:

                all_edges = {}
                for i in range(max(0, t - HISTORY_LENGTH), t + 1):
                    all_edges_processed = {k: False for k in all_edges.keys()}
                    if args.task == TASKS.WEIGHTED_EDGE_PREDICTION or i < t:
                        data = graphs[i]
                        source, target, weight = data.edge_index[0], data.edge_index[1], data.edge_attr
                    else:

                        source, target = already_pos_can_see[i]
                        weight = already_edges_attrs[i]
                    for j in range(source.size(0)):
                        edge = (source[j].item(), target[j].item())
                        all_edges_processed[edge] = True
                        if GNN_TYPE == 'WeightedGCN':
                            if edge not in all_edges:
                                all_edges[edge] = tuple(0 for _ in range(i - t + HISTORY_LENGTH)) + (weight[j].item(),)
                            else:
                                all_edges[edge] = all_edges[edge] + (weight[j].item(),)
                        else:
                            all_edges[edge] = all_edges.get(edge, 0) * combine_decay_factor + weight[j].item()
                    for k, v in all_edges_processed.items():
                        if not v:
                            if GNN_TYPE == 'WeightedGCN':
                                all_edges[k] = all_edges[k] + tuple([0])
                            else:
                                all_edges[k] = all_edges[k] * combine_decay_factor

                all_edges = [(k[0], k[1], v) for k, v in all_edges.items()]
                edge_index = torch.tensor([[e[0] for e in all_edges], [e[1] for e in all_edges]], dtype=torch.long).to(device)
                edge_attr = torch.tensor([e[2] for e in all_edges], dtype=torch.float32).unsqueeze(1).to(device)

                x = graphs[t].x
                data_cache[t] = (x, edge_index, edge_attr)
            else:
                x, edge_index, edge_attr = data_cache[t]

            gnn_embeddings = self.gnn(x, edge_index, edge_attr)[:this_batch_data_size, :]

            rnn_inputs.append(gnn_embeddings)

        rnn_output, (h_n, c_n) = self.rnn(torch.stack(rnn_inputs, dim=1))

        predictions = self.RNN_output_transform(rnn_output)
        return [predictions[:,i,:] for i in range(predictions.shape[1])]

    def predict_edge(self, node_embeddings_a, node_embeddings_b):
        """
        Args:
            node_embeddings_a: Tensor - Source Node embeddings for a single time step (shape: [num_nodes, gnn_out_features])
            node_embeddings_b: Tensor - Target Node embeddings for a single time step (shape: [num_nodes, gnn_out_features])
        """
        edge_features = torch.cat([node_embeddings_a[:, :self.gnn_out_features // 2], node_embeddings_b[:, :self.gnn_out_features // 2]], dim=-1)
        edge_pred = self.predictor_edge(edge_features)
        return edge_pred

    def predict_edge_weight(self, node_embeddings_a, node_embeddings_b):
        """
        Args:
            node_embeddings_a: Tensor - Source Node embeddings for a single time step (shape: [num_nodes, gnn_out_features])
            node_embeddings_b: Tensor - Target Node embeddings for a single time step (shape: [num_nodes, gnn_out_features])
        """
        edge_features = torch.cat([node_embeddings_a[:, self.gnn_out_features // 2:], node_embeddings_b[:, self.gnn_out_features // 2:]], dim=-1)
        edge_features_existance = torch.cat([node_embeddings_a[:, :self.gnn_out_features // 2], node_embeddings_b[:, :self.gnn_out_features // 2]], dim=-1)
        edge_existance = self.predictor_edge(edge_features_existance)
        edge_pred = self.predictor_weight(edge_features).exp() - 1
        ans = torch.zeros_like(edge_pred)
        ans[(edge_existance >= 0)] = edge_pred[(edge_existance >= 0)]
        ans[(edge_existance < 0)] = edge_pred[(edge_existance < 0)] * (edge_existance[edge_existance < 0].sigmoid()) / 2000
        return ans

def _eval_mrr(y_pred_pos, y_pred_neg, type_info='torch'):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''

    if type_info == 'torch':

        y_pred_pos = y_pred_pos.view(-1, 1)

        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)

        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hits1_list = (ranking_list <= 1).to(torch.float)
        hits3_list = (ranking_list <= 3).to(torch.float)
        hits10_list = (ranking_list <= 10).to(torch.float)
        mrr_list = 1./ranking_list.to(torch.float)

        return {'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@10_list': hits10_list,
                'mrr_list': mrr_list}

    else:
        y_pred_pos = y_pred_pos.reshape(-1, 1)
        optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
        pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
        ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
        hits1_list = (ranking_list <= 1).astype(np.float32)
        hits3_list = (ranking_list <= 3).astype(np.float32)
        hits10_list = (ranking_list <= 10).astype(np.float32)
        mrr_list = 1./ranking_list.astype(np.float32)

        return {'hits@1_list': hits1_list,
                'hits@3_list': hits3_list,
                'hits@10_list': hits10_list,
                'mrr_list': mrr_list}

def get_pos_neg_edges(edge_index, num_nodes, neg_to_pos_ratio=1000):
    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    all_nodes = torch.cat([source_nodes, target_nodes], dim=0).unique()

    all_edges = torch.stack([all_nodes.repeat(num_nodes), torch.arange(num_nodes).repeat(all_nodes.size(0))], dim=0)

    mask = torch.concat([
        torch.stack([source_nodes, target_nodes], dim=0),
        torch.stack([target_nodes, source_nodes], dim=0)
    ])
    mask = mask.t()
    mask = mask[:, 0] * num_nodes + mask[:, 1]
    mask = mask.unique()
    all_edges = all_edges[:, ~torch.isin(all_edges[0], mask)]

    need_num = neg_to_pos_ratio * edge_index.size(1)

    neg_edges_index = []
    already = 0
    while already < need_num:
        neg_edges_index.append(torch.randperm(all_edges.size(1)))
        already += all_edges.size(1)
    neg_edges_index_concat = torch.cat(neg_edges_index, dim=0)
    neg_edges = all_edges[:, neg_edges_index_concat[:need_num]]

    return edge_index, neg_edges

def get_pos_neg_edges_completion(edge_index, num_nodes, edge_attr, neg_to_pos_ratio=1000):
    num_edges = edge_index.size(1)

    sorted_indices = torch.randperm(num_edges)
    already_edges = edge_index[:, sorted_indices[:num_edges // 2]]
    already_edges_attr = edge_attr[sorted_indices[:num_edges // 2]]

    pos_edges = edge_index[:, sorted_indices[num_edges // 2:]]
    pos_edges_attr = edge_attr[sorted_indices[num_edges // 2:]]

    source_nodes = edge_index[0]
    target_nodes = edge_index[1]

    all_nodes = torch.cat([source_nodes, target_nodes], dim=0).unique()

    all_edges = torch.stack([all_nodes.repeat(num_nodes), torch.arange(num_nodes).repeat(all_nodes.size(0))], dim=0)

    mask = torch.concat([
        torch.stack([source_nodes, target_nodes], dim=0),
        torch.stack([target_nodes, source_nodes], dim=0)
    ])
    mask = mask.t()
    mask = mask[:, 0] * num_nodes + mask[:, 1]
    mask = mask.unique()
    all_edges = all_edges[:, ~torch.isin(all_edges[0], mask)]

    need_num = neg_to_pos_ratio * (num_edges - num_edges // 2)

    neg_edges_index = []
    already = 0
    while already < need_num:
        neg_edges_index.append(torch.randperm(all_edges.size(1)))
        already += all_edges.size(1)
    neg_edges_index_concat = torch.cat(neg_edges_index, dim=0)
    neg_edges = all_edges[:, neg_edges_index_concat[:need_num]]

    return pos_edges, neg_edges, already_edges, pos_edges_attr, already_edges_attr

def get_mrr(t, embeddings):
    output = embeddings
    if args.task == TASKS.WEIGHTED_EDGE_PREDICTION:
        pos = pos_samples[t]
        neg = neg_samples[t]
    else:
        pos = pos_samples_completion[t]
        neg = neg_samples_completion[t]
    if device != device2:
        model.to(device2)
        output = output.clone().to(device2)
        pos = pos.clone().to(device2)
        neg = neg.clone().to(device2)
    pos_preds = model.predict_edge(output[pos[0]], output[pos[1]]).reshape(-1)

    batch_size = 102400
    num_batches = (neg.size(1) - 1) // batch_size + 1
    neg_preds = []
    for i in range(num_batches):
        neg_preds.append(model.predict_edge(output[neg[0, i * batch_size:(i + 1) * batch_size]], output[neg[1, i * batch_size:(i + 1) * batch_size]]))
    neg_preds = torch.cat([n.reshape(-1) for n in neg_preds], dim=0).reshape([pos_preds.shape[0], -1])
    mrr = _eval_mrr(pos_preds, neg_preds)['mrr_list'].mean().item()
    if device != device2:
        model.to(device)
    return mrr

with open(os.path.join(data_path, 'edges_source.json'), 'r') as f:
    edges_source = json.load(f)
with open(os.path.join(data_path, 'edges_target.json'), 'r') as f:
    edges_target = json.load(f)
with open(os.path.join(data_path, 'weights.json'), 'r') as f:
    edges_weight = json.load(f)
if embedding_file is not None:
    node_feat = np.load(os.path.join(data_path, embedding_file), allow_pickle=True)
    num_nodes = node_feat.shape[0]
    node_features_dim = node_feat.shape[-1]
else:

    node_features_dim = num_nodes
    node_feat = np.eye(num_nodes)
all_seq_length = len(edges_source)
test_seq_length = all_seq_length - seq_length - valid_seq_length

graphs = []
valid_graphs = []
test_graphs = []
for i in range(all_seq_length):
    source, target, weight = torch.tensor(edges_source[i], dtype=torch.long), \
                             torch.tensor(edges_target[i], dtype=torch.long), \
                             torch.tensor(edges_weight[i], dtype=torch.float32)
    edge_index = torch.stack([source, target], dim=0)
    edge_attr = weight.unsqueeze(1)

    data = Data(x=torch.tensor(node_feat[:,0,:] if embedding_multiple_times else node_feat, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr).to(device)
    if i < seq_length:
        graphs.append(data)
    elif i < seq_length + valid_seq_length:
        valid_graphs.append(data)
    else:
        test_graphs.append(data)

if args.task == TASKS.WEIGHTED_EDGE_PREDICTION:
    if not os.path.exists(sample_file):
        pos_samples = []
        neg_samples = []

        for t in range(all_seq_length):
            data = graphs[t] if t < seq_length else valid_graphs[t - seq_length] if t < seq_length + valid_seq_length else test_graphs[t - seq_length - valid_seq_length]
            pos, neg = get_pos_neg_edges(data.edge_index.clone().cpu(), data.num_nodes)
            pos_samples.append(pos)
            neg_samples.append(neg)

        torch.save((pos_samples, neg_samples), sample_file)
    else:
        pos_samples, neg_samples = torch.load(sample_file)

    pos_samples = [pos.to(device) for pos in pos_samples]
    neg_samples = [neg.to(device) for neg in neg_samples]
else:
    if not os.path.exists(sample_file_completion):
        pos_samples_completion = []
        neg_samples_completion = []
        already_pos_can_see = []
        pos_edges_attrs = []
        already_edges_attrs = []

        for t in range(all_seq_length):
            data = graphs[t] if t < seq_length else valid_graphs[t - seq_length] if t < seq_length + valid_seq_length else test_graphs[t - seq_length - valid_seq_length]
            pos, neg, already, pos_edges_attr, already_edges_attr = get_pos_neg_edges_completion(data.edge_index.clone().cpu(), data.num_nodes, data.edge_attr.clone().cpu())
            pos_samples_completion.append(pos)
            neg_samples_completion.append(neg)
            already_pos_can_see.append(already)
            pos_edges_attrs.append(pos_edges_attr)
            already_edges_attrs.append(already_edges_attr)

        torch.save((pos_samples_completion, neg_samples_completion, already_pos_can_see, pos_edges_attrs, already_edges_attrs), sample_file_completion)
    else:
        pos_samples_completion, neg_samples_completion, already_pos_can_see, pos_edges_attrs, already_edges_attrs = torch.load(sample_file_completion)

test_mses = []
test_mrrs = []
test_maes = []

for run_id in range(NUM_RUNS):
    print(f'Run {run_id + 1}/{NUM_RUNS}')
    set_random_seed(SEED + run_id)
    model = BaselineRNNGNN(node_features=node_features_dim, gnn_out_features_half=args.gnn_out_features_half, rnn_hidden_size=args.rnn_hidden_size, num_rnn_layers=2, rnn_dropout=0.3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_edge = nn.BCEWithLogitsLoss()
    criterion_edge_weight = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.7, patience=5)

    min_valid_mse = float('inf')
    best_test_mse = None
    best_test_mrr = None
    best_test_mae = None

    for epoch in range(num_epochs):
        model.train()

        loss0s = []
        this_timestep_acc = []
        this_timestep_recall = []
        this_timestep_RNN_acc = []
        this_timestep_RNN_recall = []
        this_timestep_pred_acc = []
        this_timestep_pred_recall = []

        optimizer.zero_grad()

        embeddings = model(graphs, seq_length, num_nodes)

        all_losses = 0
        for t in range(1, seq_length):
            if args.task == TASKS.WEIGHTED_EDGE_PREDICTION:
                pos_inputs_all = embeddings[t - 1]
            else:
                pos_inputs_all = embeddings[t]

            pos_inputs_a = pos_inputs_all[graphs[t].edge_index[0]]
            pos_inputs_b = pos_inputs_all[graphs[t].edge_index[1]]

            neg_inputs_a = pos_inputs_all[torch.randint(0, pos_inputs_all.size(0), (pos_inputs_a.size(0),))]
            neg_inputs_b = pos_inputs_all[torch.randint(0, pos_inputs_all.size(0), (pos_inputs_a.size(0),))]

            target = torch.cat([torch.ones(pos_inputs_a.size(0), 1), torch.zeros(pos_inputs_a.size(0), 1)], dim=0).to(device)
            edge_prediction = model.predict_edge(torch.cat([pos_inputs_a, neg_inputs_a], dim=0), torch.cat([pos_inputs_b, neg_inputs_b], dim=0))
            weight_target = torch.cat([graphs[t].edge_attr, torch.zeros(pos_inputs_a.size(0), 1).to(device)], dim=0)
            edge_weight_prediction = model.predict_edge_weight(torch.cat([pos_inputs_a, neg_inputs_a], dim=0), torch.cat([pos_inputs_b, neg_inputs_b], dim=0))

            loss = criterion_edge(edge_prediction, target)
            loss_MSE = criterion_edge_weight(edge_weight_prediction, weight_target)
            loss += loss_MSE
            all_losses += loss

            edge_prediction = (torch.sigmoid(edge_prediction) > 0.5).long()
            acc = (edge_prediction == target).sum().item() / target.size(0)
            recall = (edge_prediction[target == 1] == 1).sum().item() / target[target == 1].size(0)
            this_timestep_acc.append(acc)
            this_timestep_recall.append(recall)

            loss0s.append(loss.item())
        all_losses.backward()
        optimizer.step()
        scheduler.step(np.mean(loss0s))

        print(f'Epoch {epoch+1}, ACC: {np.mean(this_timestep_acc):.4f}, recall: {np.mean(this_timestep_recall):.4f}, Loss: {np.mean(loss0s):.4f}')

        if (epoch + 1) % 5 == 0:
            model.eval()

            mrrs_base_emb = [None]
            mses_base_emb = [None]
            maes_base_emb = [None]
            data = graphs + valid_graphs + test_graphs
            embeddings = model(data, all_seq_length, num_nodes)

            for t in range(1, all_seq_length):

                if args.task == TASKS.WEIGHTED_EDGE_PREDICTION:
                    emb = embeddings[t - 1]
                    pos_num = data[t].edge_index.size(1)
                    neg = neg_samples[t][:, :pos_num]
                    edges = torch.cat([data[t].edge_index, torch.stack([neg[0], neg[1]], dim=0)], dim=1)
                    target = torch.cat([data[t].edge_attr, torch.zeros(neg.size(1), 1).to(device)], dim=0)
                else:
                    emb = embeddings[t]
                    pos = pos_samples_completion[t]
                    pos_attr = pos_edges_attrs[t].to(device)
                    pos_num = pos.size(1)
                    neg = neg_samples_completion[t][:, :pos_num]
                    edges = torch.cat([pos, torch.stack([neg[0], neg[1]], dim=0)], dim=1)
                    target = torch.cat([pos_attr, torch.zeros(neg.size(1), 1).to(device)], dim=0)
                node_a = emb[edges[0]]
                node_b = emb[edges[1]]
                edge_prediction = model.predict_edge_weight(node_a, node_b)
                loss_MSE = criterion_edge_weight(edge_prediction, target)
                loss_MAE = torch.abs(edge_prediction - target).mean()
                mses_base_emb.append(loss_MSE.item())
                maes_base_emb.append(loss_MAE.item())

                mrr = get_mrr(t, emb)
                mrrs_base_emb.append(mrr)
            mrrs_base_emb_mean_train = np.mean(mrrs_base_emb[1:seq_length])
            mrrs_base_emb_mean_valid = np.mean(mrrs_base_emb[seq_length:seq_length + valid_seq_length])
            mrrs_base_emb_mean_test = np.mean(mrrs_base_emb[seq_length + valid_seq_length:])
            mses_base_emb_mean_train = np.mean(mses_base_emb[1:seq_length])
            mses_base_emb_mean_valid = np.mean(mses_base_emb[seq_length:seq_length + valid_seq_length])
            mses_base_emb_mean_test = np.mean(mses_base_emb[seq_length + valid_seq_length:])
            maes_base_emb_mean_train = np.mean(maes_base_emb[1:seq_length])
            maes_base_emb_mean_valid = np.mean(maes_base_emb[seq_length:seq_length + valid_seq_length])
            maes_base_emb_mean_test = np.mean(maes_base_emb[seq_length + valid_seq_length:])
            print(f'\tMRR (train, valid, test): {mrrs_base_emb_mean_train:.4f} {mrrs_base_emb_mean_valid:.4f} {mrrs_base_emb_mean_test:.4f}; MSE: {mses_base_emb_mean_train:.4f} {mses_base_emb_mean_valid:.4f} {mses_base_emb_mean_test:.4f}; MAE: {maes_base_emb_mean_train:.4f} {maes_base_emb_mean_valid:.4f} {maes_base_emb_mean_test:.4f};')
            if min_valid_mse > mses_base_emb_mean_valid:
                min_valid_mse = mses_base_emb_mean_valid
                best_test_mse = mses_base_emb_mean_test
                best_test_mrr = mrrs_base_emb_mean_test
                best_test_mae = maes_base_emb_mean_test
    print("RUN", run_id, best_test_mse)
    test_mses.append(best_test_mse)
    test_mrrs.append(best_test_mrr)
    test_maes.append(best_test_mae)

print("Test MSEs:", test_mses)
print(f"Test MSE: {np.mean(test_mses):.4f} +- {np.std(test_mses):.4f}")
print("Test MRRs:", test_mrrs)
print(f"Test MRR: {np.mean(test_mrrs):.4f} +- {np.std(test_mrrs):.4f}")
print("Test MAEs:", test_maes)
print(f"Test MAE: {np.mean(test_maes):.4f} +- {np.std(test_maes):.4f}")
