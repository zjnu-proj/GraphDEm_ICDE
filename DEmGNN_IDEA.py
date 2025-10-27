import os, argparse

parser = argparse.ArgumentParser(description='Modified Code for direction embedding')
parser.add_argument('--data_name', type=str, default='IoT', choices=['IoT', 'WIDE', 'HMob', 'V1', 'V2', 'Mesh-1', 'T-Drive', 'DC', 'random_walk', 'RW'], help='Name of the dataset')
parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout rate')
parser.add_argument('--win_size', type=int, default=-1, help='Window size of historical snapshots (-1 for default)')
parser.add_argument('--epsilon', type=float, default=1e-5, help='Threshold of the zero-refining')
parser.add_argument('--num_pre_epochs', type=int, default=200, help='Number of pre-training epochs')
parser.add_argument('--num_test_snaps', type=int, default=50, help='Number of test snapshots')
parser.add_argument('--num_val_snaps', type=int, default=10, help='Number of validation snapshots')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (-1 for default)')
parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
parser.add_argument('--theta', type=float, default=0.1, help='Decaying factor')
parser.add_argument('--time_decay', type=float, default=0.8, help='Decaying factor for adj in each snapshot')
parser.add_argument('--direction_loss_coef', type=float, default=10, help='Parameter of direction loss')
parser.add_argument('--direction_coef', type=float, default=0.5, help='Related to alpha')
parser.add_argument('--early_stop_patient', type=int, default=40, help='Early stop patient')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_self_defined_predict_function', type=bool, default=False)
parser.add_argument('--use_progressive_loss', type=bool, default=True)
parser.add_argument('--use_weight_map', type=bool, default=False)
parser.add_argument('--use_BCE_loss', type=bool, default=True)
parser.add_argument('--use_noise_input', dest='use_noise_input', action='store_true')
parser.add_argument('--no-use_noise_input', dest='use_noise_input', action='store_false')
parser.add_argument('--eval_on_train', dest='eval_on_train', action='store_true')
parser.add_argument('--gpu', type=str, default='2')
parser.add_argument('--num_start_weight_mapping', type=int, default=50)
parser.set_defaults(use_noise_input=True)
parser.set_defaults(eval_on_train=False)

args = parser.parse_args()
print(args)

num_start_weight_mapping = args.num_start_weight_mapping

import time, random

model_file = 'temp_model_IDEA_' + str(int(time.time())) + '_' + str(random.randint(0, 100000)) + '.pt'

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

cache_folder = 'data/cache'
os.makedirs(cache_folder, exist_ok=True)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as Init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from utils import *
import random
import os

class GraphNeuralNetwork(Module):
    '''
    Class to define the GNN layer (w/ sparse matrix multiplication)
    '''

    def __init__(self, input_dim, output_dim, dropout_rate):
        super(GraphNeuralNetwork, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        self.agg_wei = Init.xavier_uniform_(Parameter(torch.FloatTensor(input_dim, output_dim)))

        self.param = nn.ParameterList()
        self.param.append(self.agg_wei)

        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, feat, sup):
        '''
        Rewrite the forward function
        :param feat: feature input of the GCN layer
        :param sup: GCN support (normalized adjacency matrix)
        :return: aggregated feature output of the GCN layer
        '''

        feat_agg = torch.spmm(sup, feat)
        agg_output = torch.relu(torch.mm(feat_agg, self.param[0]))
        agg_output = F.normalize(agg_output, dim=1, p=2)
        agg_output = self.dropout_layer(agg_output)

        return agg_output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DEmb_GenNet_base(nn.Module):
    '''
    Class to define the generator
    Feature Extraction Module (FEM) + Embedding Derivation Module (EDM) + Embedding Aggregation Module (EAM)
    Embedding aggregation w/ tanh or exp
    '''
    def __init__(self, FEM_dims, EDM_dims, EAM_dims, dropout_rate, is_tanh=True):
        super().__init__()

        self.FEM_dims = FEM_dims
        self.EDM_dims = EDM_dims
        self.EAM_dims = EAM_dims
        self.dropout_rate = dropout_rate

        self.dropout_layer = nn.Dropout(p=self.dropout_rate)
        self.is_tanh = is_tanh

        self.num_FEM_layers = len(self.FEM_dims)-1
        self.FEM_layers = nn.ModuleList()
        self.FEM_layers2 = nn.ModuleList()
        for l in range(self.num_FEM_layers):

            self.FEM_layers.append(nn.Linear(in_features=self.FEM_dims[l], out_features=self.FEM_dims[l+1]))
            self.FEM_layers2.append(nn.Linear(in_features=self.FEM_dims[l], out_features=self.FEM_dims[l+1]))

        self.num_EDM_layers = len(self.EDM_dims)-1
        self.EDM_layers = nn.ModuleList()
        self.EDM_layers2 = nn.ModuleList()
        for l in range(self.num_EDM_layers):

            self.EDM_layers.append(GraphNeuralNetwork(input_dim=self.EDM_dims[l], output_dim=self.EDM_dims[l+1],
                                                          dropout_rate=self.dropout_rate))
            self.EDM_layers.append(None)

            self.EDM_layers.append(None)

            self.EDM_layers2.append(GraphNeuralNetwork(input_dim=self.EDM_dims[l], output_dim=self.EDM_dims[l+1],
                                                          dropout_rate=self.dropout_rate))
            self.EDM_layers2.append(None)

            self.EDM_layers2.append(None)

        self.num_EAM_layers = len(self.EAM_dims)-1

        self.emb_layers = nn.ModuleList()
        self.emb_layers2 = nn.ModuleList()
        for l in range(self.num_EAM_layers):
            self.emb_layers.append(nn.Linear(in_features=self.EAM_dims[l], out_features=self.EAM_dims[l+1]))
            self.emb_layers2.append(nn.Linear(in_features=self.EAM_dims[l], out_features=self.EAM_dims[l+1]))

        self.scal_layers = nn.ModuleList()
        for l in range(self.num_EAM_layers):
            self.scal_layers.append(nn.Linear(in_features=self.EAM_dims[l], out_features=self.EAM_dims[l+1]))
        if args.use_self_defined_predict_function:
            self.prediction_layer = nn.Linear(in_features=self.EAM_dims[-1], out_features=1)
        if args.use_weight_map:
            self.weight_mapping = nn.Sequential(nn.Linear(3, 3), nn.ReLU(), nn.Linear(3, 1))

            self.before_weight_mapping = lambda x: torch.cat([x, x.sigmoid().exp() - 1, (x / 2).sigmoid().exp() - 1], dim=-1)
            self.weight_mapping_weight_data1 = torch.tensor([[1, 0.001, 0.001], [0.001, 1, 0.001], [0.001, 0.001, 1]], dtype=torch.float32).to(device)
            self.weight_mapping_bias_data1 = torch.tensor([0, 0, 0], dtype=torch.float32).to(device)
            self.weight_mapping_weight_data2 = torch.tensor([[1, 0.001, 0.001]], dtype=torch.float32).to(device)
            self.weight_mapping_bias_data2 = torch.tensor([0], dtype=torch.float32).to(device)

    def forward(self, sup, feats, noise, aligns, num_nodes_list, lambd, pred_flag=True, DIRECTION_COEF=0):

        FEM_input_list = feats
        FEM_output_list = None
        FEM_input_list2 = feats
        FEM_output_list2 = None

        for l in range(self.num_FEM_layers):

            FEM_layer = self.FEM_layers[l]
            FEM_output_list = []
            for t in range(len(feats)):
                FEM_input = FEM_input_list[t]

                FEM_output = FEM_layer(FEM_input)
                FEM_output = torch.relu(FEM_output)
                FEM_output_list.append(FEM_output)
            FEM_input_list = FEM_output_list

            FEM_layer2 = self.FEM_layers2[l]
            FEM_output_list2 = []
            for t in range(len(feats)):
                FEM_input2 = FEM_input_list2[t]
                FEM_output2 = FEM_layer2(FEM_input2)
                FEM_output2 = torch.relu(FEM_output2)
                FEM_output_list2.append(FEM_output2)
            FEM_input_list2 = FEM_output_list2

        EDM_input_list = []
        EDM_input_list2 = []

        EDM_input = FEM_output_list[0]

        if args.use_noise_input:
            EDM_input = torch.cat((EDM_input, noise), dim=1)
        EDM_input_list.append(EDM_input)

        EDM_input2 = FEM_output_list2[0]

        if args.use_noise_input:
            EDM_input2 = torch.cat((EDM_input2, noise), dim=1)
        EDM_input_list2.append(EDM_input2)

        for l in range(0, self.num_EDM_layers*3, 3):
            EDM_output_list = []
            EDM_output_list2 = []

            GNN_layer = self.EDM_layers[l]
            GNN_output_list = []

            feat = EDM_input_list[0]
            GNN_output = GNN_layer(feat, sup)
            GNN_output_list.append(GNN_output)

            EDM_input_list = [GNN_output]
            for t in range(len(feats) - 1):
                EDM_output_list.append(GNN_output)

            GNN_layer = self.EDM_layers2[l]
            GNN_output_list2 = []

            feat2 = EDM_input_list2[0]
            GNN_output2 = GNN_layer(feat2, sup)
            GNN_output_list2.append(GNN_output2)

            EDM_input_list2 = [GNN_output2]
            for t in range(len(feats) - 1):
                EDM_output_list2.append(GNN_output2)

        def get_adj_est(emb_cat, num_nodes):
            emb_input = emb_cat
            emb_output = None
            for l in range(self.num_EAM_layers):
                emb_layer = self.emb_layers[l]
                emb_output = emb_layer(emb_input)
                emb_output = torch.tanh(emb_output)
                emb_input = emb_output
            emb = emb_output
            emb = F.normalize(emb, dim=0, p=2)

            scal_input = emb_cat
            scal_output = None
            for l in range(self.num_EAM_layers):
                scal_layer = self.scal_layers[l]
                scal_output = scal_layer(scal_input)
                scal_output = torch.sigmoid(scal_output)
                scal_input = scal_output
            scal = torch.mm(scal_output, scal_output.t())

            emb_src = torch.reshape(emb, (1, num_nodes, self.EAM_dims[-1]))
            emb_dst = torch.reshape(emb, (num_nodes, 1, self.EAM_dims[-1]))
            if args.use_self_defined_predict_function:
                adj_est = self.prediction_layer(emb_src * emb_dst).squeeze(-1)
                adj_est = 1+torch.tanh(adj_est) if self.is_tanh else torch.exp(adj_est)
            elif args.use_weight_map:

                adj_est = -torch.sum((emb_src-emb_dst)**2, dim=2)
                adj_est = 1+torch.tanh(torch.mul(adj_est, scal)) if self.is_tanh else torch.exp(torch.mul(adj_est, scal))

                a_shape = adj_est.shape
                if epoch >= num_start_weight_mapping:
                    adj_est = self.weight_mapping(self.before_weight_mapping(adj_est.view(-1, 1))).view(a_shape)
            else:
                adj_est = -torch.sum((emb_src-emb_dst)**2, dim=2)
                adj_est = 1+torch.tanh(torch.mul(adj_est, scal)) if self.is_tanh else torch.exp(torch.mul(adj_est, scal))
            return adj_est

        if pred_flag==True:
            emb = EDM_output_list[-1] + DIRECTION_COEF * EDM_output_list2[-1]
            feat = FEM_output_list[-1]
            emb_cat = torch.cat((emb, feat), dim=1)

            adj_est = get_adj_est(emb_cat, num_nodes)
            return [adj_est], FEM_output_list[-1], FEM_output_list[-1], DIRECTION_COEF * FEM_output_list2[-1]

        else:

            adj_est_list = []
            for t in range(len(feats) - 1):

                emb = EDM_output_list[t] + t * DIRECTION_COEF * EDM_output_list2[t]
                feat = FEM_output_list[t+1]
                emb_cat = torch.cat((emb, feat), dim=1)
                adj_est = get_adj_est(emb_cat, num_nodes)
                adj_est_list.append(adj_est)
            return adj_est_list, FEM_output_list[-1], FEM_output_list[-1], DIRECTION_COEF * FEM_output_list2[-1]

class DEmb_GenNet_tanh(DEmb_GenNet_base):
    def __init__(self, FEM_dims, EDM_dims, EAM_dims, dropout_rate):
        super().__init__(FEM_dims, EDM_dims, EAM_dims, dropout_rate, is_tanh=True)

class DEmb_GenNet_exp(DEmb_GenNet_base):
    def __init__(self, FEM_dims, EDM_dims, EAM_dims, dropout_rate):
        super().__init__(FEM_dims, EDM_dims, EAM_dims, dropout_rate, is_tanh=False)

last_time_loss = None
loss_ratio = 0.0

def get_pre_gen_loss2(adj_est_list, gnd_list, theta, max_thres, beta, old_edge_list):
    '''
    Function to define the pre-training loss of generator
    :param adj_est_list: list of prediction results
    :param gnd_list: list of ground-truth
    :param theta: parameter of decaying factor
    :return: pre-training loss of generator
    '''
    global last_time_loss, loss_ratio

    loss = 0.0
    win_size = len(adj_est_list)
    for i in range(win_size):
        adj_est = adj_est_list[i]
        gnd = gnd_list[i]

        adj_est = adj_est - torch.diag(torch.diag(adj_est))
        gnd = gnd - torch.diag(torch.diag(gnd))
        decay = 1

        if args.use_BCE_loss:
            pos_data = torch.where(gnd > 0)
            neg_data = torch.where(gnd + torch.diag(torch.diag(gnd)) == 0)
            labels = torch.concat([torch.ones(pos_data[0].shape[0]), torch.zeros(neg_data[0].shape[0])]).to(device)
            predicted = torch.concat([adj_est[pos_data], adj_est[neg_data]])
            loss += F.binary_cross_entropy((predicted.tanh()), labels)
        loss += decay*torch.norm((torch.sqrt(adj_est + 1e-5) - torch.sqrt(gnd + 1e-5))[torch.where(gnd)], p='fro')**2

        if args.use_progressive_loss:

            epsilon = 1e-5 / max_thres
            E = epsilon * torch.ones_like(adj_est)
            q = adj_est
            q = torch.where(q<epsilon, E, q)
            p = gnd
            p = torch.where(p<epsilon, E, p)
            loss += loss_ratio * decay*(torch.norm((adj_est - gnd), p='fro')**2)

    return loss

def get_direction_loss2(base_vec, base_vec2, direc_vec):
    loss = torch.norm(base_vec + direc_vec - base_vec2, p=2)

    return loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

save_flag = False

data_name = args.data_name
dropout_rate = args.dropout_rate
win_size = args.win_size
epsilon = args.epsilon
num_pre_epochs = args.num_pre_epochs
num_test_snaps = args.num_test_snaps
num_val_snaps = args.num_val_snaps
lr = args.lr
wd = args.weight_decay
theta = args.theta

time_decay = args.time_decay
DIRECTION_LOSS_COEF = args.direction_loss_coef
DIRECTION_COEF = args.direction_coef
EARLY_STOP_PATIENT = args.early_stop_patient
SEED = args.seed
setup_seed(SEED)

if args.use_self_defined_predict_function:
    assert not args.use_weight_map, 'Cannot use both self defined predict function and weight map'

if data_name == 'IoT':
    sample_file_name = 'samples_IoT_20250108.pt'
    num_nodes_gbl = 668
    num_nodes = num_nodes_gbl
    num_snaps = 144
    max_thres = 1024
    noise_dim = 48
    feat_dim = 32
    pos_dim = 32
    GNN_feat_dim = feat_dim+pos_dim
    FEM_dims = [GNN_feat_dim, 64, 32]
    EDM_dims = [(FEM_dims[-1]+noise_dim), 64, 64]
    EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 64, 64]
    disc_dims = [FEM_dims[-1], 32, 16, 8]

    alpha = 40
    beta = 0.1
    if win_size < 0:
        win_size = 10
    if lr < 0:
        lr = 1e-4

elif data_name == 'WIDE':
    raise NotImplementedError
    num_nodes_gbl = 422614
    num_snaps = 800
    max_thres = 512
    noise_dim = 512
    feat_dim = 32
    pos_dim = 256
    GNN_feat_dim = feat_dim+pos_dim
    FEM_dims = [GNN_feat_dim, 256, 128]
    EDM_dims = [(FEM_dims[-1]+noise_dim), 512, 256]
    EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 256, 128]
    disc_dims = [FEM_dims[-1], 128, 64, 32]

    alpha = 10
    beta = 0.05
    if win_size < 0:
        win_size = 5
    if lr < 0:
        lr = 5e-4

elif data_name == 'HMob':
    raise NotImplementedError
    num_nodes = 92
    num_snaps = 500
    max_thres = 250
    noise_dim = 64
    pos_dim = 32
    GNN_feat_dim = pos_dim
    FEM_dims = [GNN_feat_dim, 32, 32]
    EDM_dims = [(FEM_dims[-1]+noise_dim), 128, 64]
    EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 64, 64]
    disc_dims = [FEM_dims[-1], 32, 16, 8]

    alpha = 40
    beta = 0.15
    if win_size < 0:
        win_size = 5
    if lr < 0:
        lr = 5e-4

elif data_name in ['V1', 'V2']:
    if data_name == 'V1':
        num_nodes = 20797
        num_snaps = 11
        num_test_snaps = 2
        num_val_snaps = 2
        sample_file_name = 'samples_DBLP_20241208.pt'
    elif data_name == 'V2':
        num_nodes = 3789
        num_snaps = 20
        num_test_snaps = 3
        num_val_snaps = 3
        sample_file_name = 'samples_DBLP_20241215.pt'

    num_nodes_gbl = num_nodes
    max_thres = 10
    noise_dim = 64
    pos_dim = 32
    GNN_feat_dim = 768 + pos_dim
    FEM_dims = [GNN_feat_dim, 32, 32]
    EDM_dims = [(FEM_dims[-1]+noise_dim), 128, 64]
    EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 64, 64]
    disc_dims = [FEM_dims[-1], 32, 16, 8]

    alpha = 40
    beta = 0.15
    if win_size < 0:
        win_size = 5
    if lr < 0:
        lr = 5e-4

    data_name = 'DBLP'
elif data_name in ['RW', 'random_walk']:
    sample_file_name = 'samples_RW_20250109.pt'
    num_nodes = 1000
    num_nodes_gbl = num_nodes
    num_snaps = 200
    max_thres = 1
    noise_dim = 64
    pos_dim = 32
    GNN_feat_dim = pos_dim
    FEM_dims = [GNN_feat_dim, 32, 32]
    EDM_dims = [(FEM_dims[-1]+noise_dim), 128, 64]
    EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 64, 64]
    disc_dims = [FEM_dims[-1], 32, 16, 8]

    alpha = 40
    beta = 0.15
    if win_size < 0:
        win_size = 5
    if lr < 0:
        lr = 5e-4

    data_name = 'random_walk'
elif data_name == 'Mesh-1':
    raise NotImplementedError
    num_nodes = 38
    num_snaps = 445
    max_thres = 2000
    noise_dim = 32
    feat_dim = 32
    pos_dim = 16
    GNN_feat_dim = feat_dim+pos_dim
    FEM_dims = [GNN_feat_dim, 32, 16]
    EDM_dims = [(FEM_dims[-1]+noise_dim), 32, 32]
    EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 32, 16]
    disc_dims = [FEM_dims[-1], 32, 16, 8]

    alpha = 10
    beta = 0.1
    if win_size < 0:
        win_size = 10
    if lr < 0:
        lr = 5e-4

elif data_name == 'T-Drive':
    raise NotImplementedError
    num_nodes = 1279
    num_snaps = 300
    max_thres = 5000
    noise_dim = 512
    pos_dim = 256
    GNN_feat_dim = pos_dim
    FEM_dims = [GNN_feat_dim, 128, 128]
    EDM_dims = [(FEM_dims[-1]+noise_dim), 512, 256]
    EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 256, 128]
    disc_dims = [FEM_dims[-1], 128, 64, 32]

    alpha = 20
    beta = 0.1
    if win_size < 0:
        win_size = 5
    if lr < 0:
        lr = 1e-4

elif data_name == 'DC':
    raise NotImplementedError
    num_nodes = 128
    num_snaps = 700
    max_thres = 5000
    noise_dim = 100
    feat_dim = 32
    pos_dim = 32
    GNN_feat_dim = feat_dim+pos_dim
    FEM_dims = [GNN_feat_dim, 32]
    EDM_dims = [(FEM_dims[-1]+noise_dim), 100, 64]
    EAM_dims = [(EDM_dims[-1]+FEM_dims[-1]), 64]
    disc_dims = [FEM_dims[-1], 32, 16, 8]

    alpha = 10
    beta = 0.1
    if win_size < 0:
        win_size = 5
    if lr < 0:
        lr = 5e-4

else:
    raise ValueError('Invalid data name')

if not args.use_noise_input:
    EDM_dims[0] -= noise_dim
    noise_dim = 0

num_train_snaps = num_snaps-num_test_snaps-num_val_snaps

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

pos_samples, neg_samples = torch.load(sample_file_name)

def get_mrr(t, adj_t):
    pos = pos_samples[t]
    neg = neg_samples[t]
    pos_preds = adj_t[pos[0], pos[1]]
    neg_preds = adj_t[neg[0], neg[1]].reshape([pos_preds.shape[0], -1])
    mrr = _eval_mrr(pos_preds, neg_preds)['mrr_list'].mean().item()
    return mrr

def get_mae_mse(t, adj_t, gnd):
    pos = pos_samples[t]
    neg = neg_samples[t][:, :pos.size(1)]
    target = np.concatenate([gnd[pos[0], pos[1]], np.zeros(neg.size(1))])
    predicted = np.concatenate([adj_t[pos[0], pos[1]], adj_t[neg[0], neg[1]]])
    mae = np.abs(target - predicted).mean()
    mse = ((target - predicted) ** 2).mean()
    return mae, mse

NUM_RUNS = 5

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

test_mses = []
test_mrrs = []
test_maes = []

sup_tnr_cache = {}
old_edge_list_cache = {}
gnd_cache = {}
gnd_cache_val = {}

node_set = set(range(num_nodes_gbl))
node_ids = sorted(list(node_set))
node_map = get_node_map(node_set)

for run_id in range(NUM_RUNS):
    print(f'Run {run_id + 1}/{NUM_RUNS}')
    set_random_seed(SEED + run_id)

    min_valid_mse = float('inf')
    best_test_mse = None
    best_test_mrr = None
    best_test_mae = None

    edge_seq = np.load('data/%s_edge_seq.npy' % (data_name), allow_pickle=True)
    if data_name in ['IoT', 'WIDE']:

        feat_gbl = np.load('data/%s_feat.npy' % (data_name), allow_pickle=True)
        feat_lcl_seq = [torch.FloatTensor(feat_gbl).to(device)] * num_snaps
    else:

        if data_name in ['Mesh-1', 'DC', 'DBLP']:

            feat = np.load('data/%s_feat.npy' % (data_name), allow_pickle=True)
            feat_tnr = torch.FloatTensor(feat).to(device)

        pos_emb = None
        for p in range(num_nodes):
            if p==0:
                pos_emb = get_pos_emb(p, pos_dim)
            else:
                pos_emb = np.concatenate((pos_emb, get_pos_emb(p, pos_dim)), axis=0)
        if data_name in ['Mesh-1', 'DC', 'DBLP']:
            pos_tnr = torch.FloatTensor(pos_emb).to(device)
            feat_tnr = torch.cat((feat_tnr, pos_tnr), dim=1)
        else:
            feat_tnr = torch.FloatTensor(pos_emb).to(device)
        feat_gbl = feat_tnr.cpu().data.numpy()
        feat_lcl_seq = [torch.FloatTensor(feat_gbl).to(device)] * num_snaps

    base_model_class = DEmb_GenNet_tanh
    if data_name in ['WIDE', 'T-Drive']:
        base_model_class = DEmb_GenNet_exp
    gen_net = base_model_class(FEM_dims, EDM_dims, EAM_dims, dropout_rate).to(device)

    pre_gen_opt = optim.Adam(gen_net.parameters(), lr=lr, weight_decay=wd)

    best_val_metric_for_early_stopping = float('inf')
    val_metric_record_last_6 = []
    best_epoch_for_early_stopping = 0

    for epoch in range(num_pre_epochs):

        gen_net.train()

        gen_loss_list = []
        gen_loss_list_base = []

        if args.eval_on_train:
            train_adj_sum = []
            train_gnd_sum = []
            train_metrics = {
                "RMSE": [],
                "MLSD": [],
                "MR": [],
                "FP": [],
                "FN": [],
                "MAE": [],
                "MSE": [],
                "MRR": []
            }

        if args.use_weight_map and epoch == num_start_weight_mapping:

            gen_net.load_state_dict(torch.load(model_file))

            gen_net.weight_mapping[0].weight.data = gen_net.weight_mapping_weight_data1
            gen_net.weight_mapping[0].bias.data = gen_net.weight_mapping_bias_data1
            gen_net.weight_mapping[2].weight.data = gen_net.weight_mapping_weight_data2
            gen_net.weight_mapping[2].bias.data = gen_net.weight_mapping_bias_data2

        for tau in range(win_size, num_train_snaps):

            adj_norm = None
            if tau not in sup_tnr_cache:
                for t in range(tau-win_size, tau):

                    edges = edge_seq[t]

                    adj = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
                    if adj_norm is None:
                        adj_norm = adj/max_thres
                    else:
                        adj_norm = adj_norm * time_decay + adj/max_thres

                sup = get_gnn_sup(adj_norm)
                sup_sp = sp.sparse.coo_matrix(sup)
                sup_sp = sparse_to_tuple(sup_sp)
                idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                vals = torch.FloatTensor(sup_sp[1]).to(device)
                sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
                sup_tnr_cache[tau] = sup_tnr.cpu()
            sup_tnr = sup_tnr_cache[tau].to(device)

            if args.use_noise_input:

                cache_file = os.path.join(cache_folder, f'noise_tnr2_train_{data_name}_{tau}.npy')
                if os.path.exists(cache_file):
                    noise_tnr = torch.FloatTensor(np.load(cache_file)).to(device)
                else:
                    mod_tnr = torch.FloatTensor(get_mod(adj_norm)).to(device)
                    rand_mat = rand_proj(num_nodes, noise_dim)
                    rand_tnr = torch.FloatTensor(rand_mat).to(device)
                    noise_tnr = torch.mm(mod_tnr, rand_tnr)
                    np.save(cache_file, noise_tnr.cpu().data.numpy())
            else:
                noise_tnr = None

            if tau not in gnd_cache:

                t = tau
                edges = edge_seq[t]
                gnd = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
                gnd_norm = gnd/max_thres

                gnd_tnr1 = torch.FloatTensor(gnd_norm).to(device)

                t = tau + 1
                edges = edge_seq[t]
                gnd = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
                gnd_norm = gnd/max_thres

                gnd_tnr2 = torch.FloatTensor(gnd_norm).to(device)

                gnds = [gnd_tnr1, gnd_tnr2]
                gnd_cache[tau] = gnds
            gnds = gnd_cache[tau]

            if tau not in old_edge_list_cache:
                old_edge_list = []
                for target_idx in [tau, tau+1]:

                    new_edges = {}
                    for t in range(tau - win_size, num_train_snaps):
                        for e in edge_seq[t]:
                            if e[0] not in new_edges:
                                new_edges[e[0]] = set()
                            new_edges[e[0]].add(e[1])

                    old_edges = {}
                    for t in range(tau - win_size):

                        for e in edge_seq[t]:
                            if e[0] not in new_edges or e[1] not in new_edges[e[0]]:

                                new_s = e[0]
                                new_t = e[1]
                                if new_s not in old_edges:
                                    old_edges[new_s] = set()
                                old_edges[new_s].add(new_t)
                    old_edge_list.append(old_edges)
                old_edge_list_cache[tau] = old_edge_list
            old_edge_list = old_edge_list_cache[tau]
            feats = feat_lcl_seq[:3]

            adj_est_list, base_vec, base_vec2, direc_vec = gen_net(sup_tnr, feats, noise_tnr, None, None, None, pred_flag=False, DIRECTION_COEF=DIRECTION_COEF)
            if args.eval_on_train:
                train_adj_sum.append(sum([(t - torch.diag(torch.diag(t))).sum() for t in adj_est_list]))
                train_gnd_sum.append(sum([(t - torch.diag(torch.diag(t))).sum() for t in gnds]))

                def calculate_metrics(a, g):
                    if torch.cuda.is_available():
                        adj_est = a.cpu().data.numpy()
                    else:
                        adj_est = a.data.numpy()

                    adj_est *= max_thres

                    np.fill_diagonal(adj_est, 0)

                    adj_est[adj_est <= epsilon] = 0
                    gnd = g.cpu().data.numpy() * max_thres

                    wrong_rate_FP = ((gnd <= 0) * (adj_est > 0)).sum() / (gnd <= 0).sum()
                    wrong_rate_FN = ((gnd > 0) * (adj_est <= 0)).sum() / (gnd > 0).sum()

                    RMSE = get_RMSE(adj_est, gnd, num_nodes)
                    MAE, MSE = get_mae_mse(tau, adj_est, gnd)
                    MLSD = get_MLSD(adj_est, gnd, num_nodes)
                    MR = get_MR(adj_est, gnd, num_nodes)
                    MRR = get_mrr(tau, torch.FloatTensor(adj_est).to(device))
                    return RMSE, MAE, MSE, MLSD, MR, MRR, wrong_rate_FP, wrong_rate_FN
                RMSE_list = []
                MLSD_list = []
                MR_list = []
                FP_rate_list = []
                FN_rate_list = []

                MAE_list = []
                MSE_list = []
                MRR_list = []
                for i in range(len(adj_est_list)):
                    RMSE, MAE, MSE, MLSD, MR, MRR, FP, FN = calculate_metrics(adj_est_list[i], gnds[i])
                    RMSE_list.append(RMSE)
                    MLSD_list.append(MLSD)
                    MR_list.append(MR)
                    FP_rate_list.append(FP)
                    FN_rate_list.append(FN)

                    MAE_list.append(MAE)
                    MSE_list.append(MSE)
                    MRR_list.append(MRR)

                train_metrics['RMSE'].append(np.mean(RMSE_list))
                train_metrics['MLSD'].append(np.mean(MLSD_list))
                train_metrics['MR'].append(np.mean(MR_list))
                train_metrics['FP'].append(np.mean(FP_rate_list))
                train_metrics['FN'].append(np.mean(FN_rate_list))

                train_metrics['MAE'].append(np.mean(MAE_list))
                train_metrics['MSE'].append(np.mean(MSE_list))
                train_metrics['MRR'].append(np.mean(MRR_list))
            pre_gen_loss = get_pre_gen_loss2(adj_est_list, gnds, theta, max_thres, beta, old_edge_list)
            gen_loss_list_base.append(pre_gen_loss.item())
            pre_gen_loss += DIRECTION_LOSS_COEF * get_direction_loss2(base_vec, base_vec2, direc_vec)
            pre_gen_opt.zero_grad()

            pre_gen_loss.backward()
            pre_gen_opt.step()

            gen_loss_list.append(pre_gen_loss.item())

        gen_loss_mean = np.mean(gen_loss_list)
        print('#%d Pre-Train G-Loss %f (BASE: %f)' % (epoch, gen_loss_mean, np.mean(gen_loss_list_base)))
        if args.eval_on_train:
            print("TRAIN", "RMSE", np.mean(train_metrics['RMSE']), "MLSD", np.mean(train_metrics['MLSD']), "MR", np.mean(train_metrics['MR']), "FP", np.mean(train_metrics['FP']), "FN", np.mean(train_metrics['FN']))
            print("\t", "SUM", sum(train_adj_sum).item(), sum(train_gnd_sum).item(), "RATIO", (sum(train_adj_sum) / sum(train_gnd_sum)).item())
            print("TRAIN", "MAE", np.mean(train_metrics['MAE']), "MSE", np.mean(train_metrics['MSE']), "MRR", np.mean(train_metrics['MRR']))

        new_loss = np.mean(gen_loss_list_base)
        if last_time_loss is not None and 0.99 < new_loss / last_time_loss < 1.01:
            print("Increase loss ratio", loss_ratio, end=' ')
            loss_ratio = min(loss_ratio + 0.02, 1.0)
            print("to", loss_ratio)
        last_time_loss = new_loss

        for stage in ['valid', 'test']:

            gen_net.eval()

            MAE_list = []
            MSE_list = []
            MRR_list = []
            for tau in (range(num_snaps-num_test_snaps-num_val_snaps, num_snaps-num_test_snaps) if stage == 'valid' else range(num_snaps-num_test_snaps, num_snaps)):

                adj_norm = None
                if tau not in sup_tnr_cache:
                    for t in range(tau-win_size, tau):

                        edges = edge_seq[t]
                        adj = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
                        if adj_norm is None:
                            adj_norm = adj/max_thres
                        else:
                            adj_norm = adj_norm * time_decay + adj/max_thres

                    sup = get_gnn_sup(adj_norm)
                    sup_sp = sp.sparse.coo_matrix(sup)
                    sup_sp = sparse_to_tuple(sup_sp)
                    idxs = torch.LongTensor(sup_sp[0].astype(float)).to(device)
                    vals = torch.FloatTensor(sup_sp[1]).to(device)
                    sup_tnr = torch.sparse.FloatTensor(idxs.t(), vals, sup_sp[2]).float().to(device)
                    sup_tnr_cache[tau] = sup_tnr.cpu()
                sup_tnr = sup_tnr_cache[tau].to(device)

                cache_file = os.path.join(cache_folder, f'noise_tnr2_{stage}_{data_name}_{tau}.npy')
                if os.path.exists(cache_file):
                    noise_tnr = torch.FloatTensor(np.load(cache_file)).to(device)
                else:
                    mod_tnr = torch.FloatTensor(get_mod(adj_norm)).to(device)
                    rand_mat = rand_proj(num_nodes, noise_dim)
                    rand_tnr = torch.FloatTensor(rand_mat).to(device)
                    noise_tnr = torch.mm(mod_tnr, rand_tnr)
                    np.save(cache_file, noise_tnr.cpu().data.numpy())

                edges = edge_seq[tau]
                if data_name in ['IoT', 'WIDE']:
                    if tau not in gnd_cache_val:

                        gnd_L3 = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
                        feat_tnr = feat_lcl_seq[tau]
                        gnd_cache_val[tau] = gnd_L3
                    gnd_L3 = gnd_cache_val[tau]
                else:
                    if tau not in gnd_cache_val:
                        feat_tnr = feat_lcl_seq[tau]
                        gnd = get_adj_wei_map(edges, node_map, num_nodes, max_thres)
                        gnd_cache_val[tau] = gnd
                    gnd = gnd_cache_val[tau]

                adj_est_list, _, _, _ = gen_net(sup_tnr, feats, noise_tnr, None, None, None, pred_flag=True, DIRECTION_COEF=DIRECTION_COEF)
                if data_name in ['IoT', 'WIDE']:
                    adj_est_L3 = adj_est_list[-1]
                    if torch.cuda.is_available():
                        adj_est_L3 = adj_est_L3.cpu().data.numpy()
                    else:
                        adj_est_L3 = adj_est_L3.data.numpy()

                    adj_est_L3 *= max_thres

                    np.fill_diagonal(adj_est_L3, 0)

                    adj_est_L3[adj_est_L3 <= epsilon] = 0

                    MAE, MSE = get_mae_mse(tau, adj_est_L3, gnd_L3)
                    MRR = get_mrr(tau, torch.FloatTensor(adj_est_L3).to(device))

                    MAE_list.append(MAE)
                    MSE_list.append(MSE)
                    MRR_list.append(MRR)
                else:
                    adj_est = adj_est_list[-1]
                    if torch.cuda.is_available():
                        adj_est = adj_est.cpu().data.numpy()
                    else:
                        adj_est = adj_est.data.numpy()

                    adj_est *= max_thres

                    np.fill_diagonal(adj_est, 0)

                    adj_est[adj_est <= epsilon] = 0

                    MAE, MSE = get_mae_mse(tau, adj_est, gnd)
                    MRR = get_mrr(tau, torch.FloatTensor(adj_est).to(device))

                    MAE_list.append(MAE)
                    MSE_list.append(MSE)
                    MRR_list.append(MRR)
            MAE_mean = np.mean(MAE_list)
            MSE_mean = np.mean(MSE_list)
            MRR_mean = np.mean(MRR_list)
            print('%s Pre-#%d MAE %f MSE %f MRR %f'
                % (('Val' if stage == 'valid' else 'Test'), epoch, MAE_mean, MSE_mean, MRR_mean))
            f_input = open('res/%s_IDEA_rec.txt' % (data_name), 'a+')
            f_input.write('%s Pre #%d MAE %f MSE %f MRR %f\n'
                        % (('Val' if stage == 'valid' else 'Test'), epoch, MAE_mean, MSE_mean, MRR_mean))
            f_input.close()

            val_metric_record_last_6.append(MSE_mean)

            if stage == 'valid':
                update_test_mse_mrr_mae = False
                if min_valid_mse > MSE_mean:
                    min_valid_mse = MSE_mean
                    update_test_mse_mrr_mae = True
            else:
                if update_test_mse_mrr_mae:
                    best_test_mse = MSE_mean
                    best_test_mrr = MRR_mean
                    best_test_mae = MAE_mean

                    torch.save(gen_net.state_dict(), model_file)

        if len(val_metric_record_last_6) > 6:
            val_metric_record_last_6.pop(0)
            averaged_value = np.mean(val_metric_record_last_6)
            if averaged_value < best_val_metric_for_early_stopping:
                best_val_metric_for_early_stopping = averaged_value
                best_epoch_for_early_stopping = epoch
        if epoch - best_epoch_for_early_stopping >= EARLY_STOP_PATIENT:
            print("EARLY STOPPING!")
            break

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

os.remove(model_file)
