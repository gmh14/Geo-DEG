import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from torchdiffeq import odeint_adjoint as odeint
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import numpy as np

from .utils import get_rw_adj, gcn_norm_fill_val, Meter
from .base_classes import ODEblock, ODEFunc
from chemprop.models import MoleculeFeatModel
from chemprop.data import MoleculeDatapoint, MoleculeDataset
import sys
sys.path.append("..")
from GCN.model import GNN

class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, opt, device, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = 0.2
        self.concat = concat
        self.device = device
        self.opt = opt
        self.h = opt['heads']
        self.attention_dim = out_features
        
        assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
        self.d_k = self.attention_dim // opt['heads']

        self.W = nn.Parameter(torch.zeros(size=(in_features, self.attention_dim), dtype=torch.float))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.Wout = nn.Parameter(torch.zeros(size=(self.attention_dim, self.in_features), dtype=torch.float))
        nn.init.xavier_normal_(self.Wout.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(2 * self.d_k, 1, 1), dtype=torch.float))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, x, edge):
        wx = torch.mm(x, self.W)  # h: N x out
        h = wx.view(-1, self.h, self.d_k)
        h = h.transpose(1, 2)

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :, :], h[edge[1, :], :, :]), dim=1).transpose(0, 1).to(
            self.device)  # edge: 2*D x E
        edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=0)).to(self.device)
        attention = softmax(edge_e, edge[0]) # row normalize
        return attention, wx

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

    # currently requires in_features = out_features
    def __init__(self, opt, data, edge_index, edge_weight, train_edge_split_id, feat_dim, device):
        super(LaplacianODEFunc, self).__init__(opt, device)
        self.device = device
        # self.alpha_train = nn.Parameter(torch.ones_like(data.x))
        # self.alpha_train = torch.ones([data.num_nodes, 1]).to(device)
        self.alpha_train = nn.Parameter(torch.tensor(0.0))
        self.beta_train = nn.Parameter(torch.tensor(0.0))

        # self.fixed_L_param = edge_weight[:train_edge_split_id]
        # self.L_param = nn.Parameter(edge_weight[train_edge_split_id:])
        # self.beta_train = nn.Parameter(torch.ones([data.x.shape[0], 1]))
        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device)

        in_features = feat_dim
        out_features = 64
        self.multihead_att_layer = SpGraphAttentionLayer(in_features, out_features, opt, device).to(device)
        self.attention_dim = out_features
        assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
        self.d_k = self.attention_dim // opt['heads']

    def multiply_attention(self, x, attention):
        ax = torch.mean(torch.stack(
            [torch_sparse.spmm(self.edge_index, attention[:, idx], x.shape[0], x.shape[0], x) for idx in
            range(self.opt['heads'])], dim=0),
            dim=0)
        return ax

    def sparse_multiply(self, x, t):
        self.final_edge_index = self.edge_index #TODO modify
        ## 
        self.attention_weights = torch.cat((self.fixed_L_param, self.L_param), dim=0)
        ax = torch_sparse.spmm(self.final_edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
        return ax

    def forward(self, t, x):  # the t param is needed by the ODE solver.
        self.nfe += 1
        # ax = self.sparse_multiply(x, t)
        attention, wx = self.multihead_att_layer(x, self.edge_index)
        ax = self.multiply_attention(x, attention)
        # alpha = torch.sigmoid(self.alpha_train)
        alpha = self.alpha_train
        f = alpha * (ax - x)
        f = f + self.beta_train * self.x0 # residual
        return f


class ConstantODEblock(ODEblock):
    def __init__(self, odefunc, opt, data, train_edge_split_id, device, t=torch.tensor([0, 1])):
        super(ConstantODEblock, self).__init__(odefunc, opt, data, device, t)

        edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                    fill_value=opt['self_loop_weight'],
                                                                    num_nodes=data.num_nodes,
                                                                    dtype=torch.float64)
        # edge_index: 2 x N_edges, edge_weight: N_edges
        feat_dim = data.x.size(1)
        self.odefunc = odefunc(opt, data, edge_index, edge_weight, train_edge_split_id, feat_dim, device)
        self.odefunc.T = t[1].item()

        self.train_integrator = odeint
        self.test_integrator = odeint
        self.set_tol()

    def forward(self, x):
        t = self.t.type_as(x)

        integrator = self.train_integrator if self.training else self.test_integrator

        func = self.odefunc
        state = x

        if self.training:
            state_dt = integrator(
                func, state, t,
                method='dopri5',
                # options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                options=dict(step_t=self.opt['step_size']), #, max_num_steps=self.opt['max_iters']),
                adjoint_method=self.opt['adjoint_method'],
                adjoint_options=dict(step_size = self.opt['adjoint_step_size'], max_iters=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol,
                adjoint_atol=self.atol_adjoint,
                adjoint_rtol=self.rtol_adjoint)
        else:
            state_dt = integrator(
                func, state, t,
                method='dopri5',
                # options=dict(step_size=self.opt['step_size'], max_iters=self.opt['max_iters']),
                options=dict(step_t=self.opt['step_size']), #, max_num_steps=self.opt['max_iters']),
                atol=self.atol,
                rtol=self.rtol)
        # L = np.zeros([x.size(0), x.size(0)])
        # row_idx, col_idx = self.odefunc.edge_index.numpy()
        # coeff = self.odefunc.edge_weight.numpy()
        # L[row_idx, col_idx] = coeff
        # L = L - np.identity(x.size(0))
        # w, v = np.linalg.eig(L)
        # a = np.linalg.inv(v) @ (x.numpy())
        # final_a = a * np.expand_dims(np.exp(1 * w), -1)
        # final_x = v @ final_a
        # import pdb; pdb.set_trace()
        z = state_dt[-1]
        return z

    def __repr__(self):
        return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
            + ")"


class Diffuse(MessagePassing):
    def __init__(self, opt, dataset, device, with_diffusion=True):
        super(Diffuse, self).__init__()
        self.opt = opt
        self.T = opt['time']
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device
        self.with_diffusion = with_diffusion
        self.f = LaplacianODEFunc
        block = ConstantODEblock
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, opt, dataset.data, dataset.train_edge_split_id, device, t=time_tensor).to(device)
        self.meta_node_split_id = dataset.meta_node_split_id

        self.fc1 = nn.Linear(32, 8).double()
        self.bn = nn.BatchNorm1d(8).double()
        self.fc2 = nn.Linear(8, 1).double()
        self.fm = Meter()
        self.bm = Meter()


    def forward(self, x):
        z = F.relu(x)
        z = self.fc1(z)
        z = self.bn(z)
        if self.with_diffusion:
            self.odeblock.set_x0(z)
            z = self.odeblock(z)
        z = F.relu(z)
        z = self.fc2(z)
        out = z[self.meta_node_split_id:, :]
        return out

    def getNFE(self):
        return self.odeblock.odefunc.nfe 

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0


class DiffuseFixGNNFeat(MessagePassing):
    def __init__(self, opt, dataset, device, with_diffusion=True):
        super(DiffuseGNN, self).__init__()
        self.opt = opt
        self.T = opt['time']
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device
        self.with_diffusion = with_diffusion
        self.f = LaplacianODEFunc
        block = ConstantODEblock
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, opt, dataset.data, dataset.train_edge_split_id, device, t=time_tensor).to(device)
        self.meta_node_split_id = dataset.meta_node_split_id

        feat_dim = 64
        self.mapping = nn.Linear(32, 300).double()
        self.fc1 = nn.Linear(300, feat_dim).double()
        self.bn = nn.BatchNorm1d(feat_dim).double()
        self.fc2 = nn.Linear(feat_dim, 1).double()
        self.fm = Meter()
        self.bm = Meter()

    def forward(self, x_meta, x):
        # import pdb; pdb.set_trace()
        x_meta = self.mapping(x_meta)
        # x = torch.reshape(x, [-1, x.shape[1] * x.shape[2]])
        # z = F.relu(x)
        # z = self.fc1(z)
        # z = self.bn(z)
        # import pdb; pdb.set_trace()
        z = torch.cat([x_meta, x], dim=0)
        if self.with_diffusion:
            self.odeblock.set_x0(z)
            z = self.odeblock(z)
        z = self.fc1(z)
        z = self.bn(z)
        z = F.relu(z)
        # z = F.dropout(z, p=0.1)
        z = self.fc2(z)
        out = z[self.meta_node_split_id:, :]
        return out

    def getNFE(self):
        return self.odeblock.odefunc.nfe 

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0


class DiffuseGNN(MessagePassing):
    def __init__(self, opt, dataset, device, with_diffusion=True, drop_ratio=0.0):
        super(DiffuseGNN, self).__init__()
        num_layer = 5
        emb_dim = 300
        num_tasks = 1
        JK = 'last'
        drop_ratio = drop_ratio
        graph_pooling = 'mean'
        gnn_type = 'gin'
        self.opt = opt
        self.T = opt['time']
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device
        self.with_diffusion = with_diffusion
        self.f = LaplacianODEFunc
        block = ConstantODEblock
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, opt, dataset.data, dataset.train_edge_split_id, device, t=time_tensor).to(device)
        self.meta_node_split_id = dataset.meta_node_split_id
        self.meta_mapping = nn.Linear(32, 300)

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type)
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (num_layer + 1) * emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * emb_dim, num_tasks)
        
        self.fm = Meter()
        self.bm = Meter()

    def forward(self, x_meta, data_for_gnn):
        x, x_edge_index, x_edge_attr, x_batch = data_for_gnn.x, data_for_gnn.edge_index, data_for_gnn.edge_attr, data_for_gnn.batch
        x = self.gnn(x, x_edge_index, x_edge_attr)
        x = self.pool(x, x_batch)
        x_meta = self.meta_mapping(x_meta)
        z = torch.cat([x_meta, x], dim=0)
        if self.with_diffusion:
            self.odeblock.set_x0(z)
            z = self.odeblock(z)
        z = self.graph_pred_linear(z)
        out = z[self.meta_node_split_id:, :]
        return out

    def load_from_pretrain(self, path):
        pt = torch.load(path)
        missing_keys, unexpected_keys = self.load_state_dict(pt, strict=False)
        assert(len(unexpected_keys) == 0)

    def getNFE(self):
        return self.odeblock.odefunc.nfe 

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0
        

class DiffuseGNNGrammar(MessagePassing):
    def __init__(self, opt, device, with_diffusion=True, drop_ratio=0.0):
        super(DiffuseGNNGrammar, self).__init__()
        num_layer = 5
        emb_dim = 300
        if opt['multi_task']:
            num_tasks = 6
        else:
            num_tasks = 1
        JK = 'last'
        drop_ratio = drop_ratio
        graph_pooling = 'mean'
        gnn_type = 'gin'
        self.opt = opt
        self.T = opt['time']
        self.device = device
        self.with_diffusion = with_diffusion
        self.opt = opt
        self.f = LaplacianODEFunc
        self.block = ConstantODEblock
        self.time_tensor = torch.tensor([0, self.T]).to(device)
        self.meta_mapping = nn.Linear(32, 300)
        self.classification = opt['classification']

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type)
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (num_layer + 1) * emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * emb_dim, num_tasks)
        
        self.sigmoid = nn.Sigmoid()
        self.fm = Meter()
        self.bm = Meter()
    
    def init_diffusion(self, dataset):
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.odeblock = self.block(self.f, self.opt, dataset.data, dataset.train_edge_split_id, self.device, t=self.time_tensor).to(self.device)
        self.meta_node_split_id = dataset.meta_node_split_id

    def forward(self, x_meta, data_for_gnn):
        x, x_edge_index, x_edge_attr, x_batch = data_for_gnn.x, data_for_gnn.edge_index, data_for_gnn.edge_attr, data_for_gnn.batch
        x = self.gnn(x, x_edge_index, x_edge_attr)
        x = self.pool(x, x_batch)
        x_meta = self.meta_mapping(x_meta)
        z = torch.cat([x_meta, x], dim=0)
        if self.with_diffusion:
            self.odeblock.set_x0(z)
            z = self.odeblock(z)
        z = self.graph_pred_linear(z)
        # if self.classification:
        #     z = self.sigmoid(z)
        out = z[self.meta_node_split_id:, :]
        return out

    def load_from_pretrain(self, path):
        pt = torch.load(path)
        missing_keys, unexpected_keys = self.load_state_dict(pt, strict=False)
        assert(len(unexpected_keys) == 0)

    def getNFE(self):
        return self.odeblock.odefunc.nfe 

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0


class DiffuseBLGNNGrammar(MessagePassing):
    def __init__(self, opt, device, with_diffusion=True, drop_ratio=0.0):
        super(DiffuseBLGNNGrammar, self).__init__()
        num_layer = 5
        emb_dim = 300
        if opt['multi_task']:
            num_tasks = 6
        else:
            num_tasks = 1
        JK = 'last'
        drop_ratio = drop_ratio
        graph_pooling = 'mean'
        gnn_type = 'gin'
        self.opt = opt
        self.T = opt['time']
        self.device = device
        self.with_diffusion = with_diffusion
        self.opt = opt
        self.f = LaplacianODEFunc
        self.block = ConstantODEblock
        self.time_tensor = torch.tensor([0, self.T]).to(device)
        self.positional_encoding = nn.Embedding(150, 16)
        self.x_meta_pos = torch.tensor(list(range(150))).to(device)
        self.meta_mapping = nn.Linear(32 + 16, 300)
        self.classification = opt['classification']

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type)
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (num_layer + 1) * emb_dim, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * emb_dim, num_tasks)
        
        self.sigmoid = nn.Sigmoid()
        self.fm = Meter()
        self.bm = Meter()
    
    def init_diffusion(self, dataset):
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.odeblock = self.block(self.f, self.opt, dataset.data, dataset.train_edge_split_id, self.device, t=self.time_tensor).to(self.device)
        self.meta_node_split_id = dataset.meta_node_split_id

    def forward(self, x_meta, data_for_gnn):
        x, x_edge_index, x_edge_attr, x_batch = data_for_gnn.x, data_for_gnn.edge_index, data_for_gnn.edge_attr, data_for_gnn.batch
        x = self.gnn(x, x_edge_index, x_edge_attr)
        x = self.pool(x, x_batch)
        x_meta = torch.concat([x_meta, self.positional_encoding(self.x_meta_pos)], dim=1)
        x_meta = self.meta_mapping(x_meta)
        z = torch.cat([x_meta, x], dim=0)
        if self.with_diffusion:
            self.odeblock.set_x0(z)
            z = self.odeblock(z)
        z = self.graph_pred_linear(z)
        # if self.classification:
        #     z = self.sigmoid(z)
        out = z[self.meta_node_split_id:, :]
        return out

    def load_from_pretrain(self, path):
        pt = torch.load(path)
        missing_keys, unexpected_keys = self.load_state_dict(pt, strict=False)
        assert(len(unexpected_keys) == 0)

    def getNFE(self):
        return self.odeblock.odefunc.nfe 

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0


class DiffuseMPNGrammar(MessagePassing):
    def __init__(self, opt, model_args, device, with_diffusion=True):
        super(DiffuseMPNGrammar, self).__init__()
        self.opt = opt
        self.T = opt['time']
        self.device = device
        self.with_diffusion = with_diffusion
        self.opt = opt
        self.f = LaplacianODEFunc
        self.block = ConstantODEblock
        self.time_tensor = torch.tensor([0, self.T]).to(device)
        self.meta_mapping = nn.Linear(32, 300)
        if opt['multi_task']:
            model_args.output_size = 6
        else:
            model_args.output_size = 1

        self.mpn = MoleculeFeatModel(model_args)
        self.fnn = self.mpn.create_ffn(model_args)
        self.classification = opt['classification']
        
        self.sigmoid = nn.Sigmoid()
        self.fm = Meter()
        self.bm = Meter()
    
    def init_diffusion(self, dataset):
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.odeblock = self.block(self.f, self.opt, dataset.data, dataset.train_edge_split_id, self.device, t=self.time_tensor).to(self.device)
        self.meta_node_split_id = dataset.meta_node_split_id

    def forward(self, x_meta, data_for_gnn):
        batch = MoleculeDataset(data_for_gnn)
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features()
        x = self.mpn(mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        x_meta = self.meta_mapping(x_meta)
        z = torch.cat([x_meta, x], dim=0)
        if self.with_diffusion:
            self.odeblock.set_x0(z)
            z = self.odeblock(z)
        z = self.fnn(z)
        # if self.classification:
        #     z = self.sigmoid(z)
        out = z[self.meta_node_split_id:, :]
        return out

    def load_from_pretrain(self, path):
        pt = torch.load(path)
        missing_keys, unexpected_keys = self.load_state_dict(pt, strict=False)
        assert(len(unexpected_keys) == 0)

    def getNFE(self):
        return self.odeblock.odefunc.nfe 

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0


class DiffuseConditionGNNGrammar(MessagePassing):
    def __init__(self, opt, device, with_diffusion=True, drop_ratio=0.0):
        super(DiffuseConditionGNNGrammar, self).__init__()
        num_layer = 5
        emb_dim = 300
        
        if opt['with_condition']:
            output_dim = opt['intermediate_feat_dim']
        else:
            output_dim = 1
            
        JK = 'last'
        drop_ratio = drop_ratio
        graph_pooling = 'mean'
        gnn_type = 'gin'
        self.opt = opt
        self.T = opt['time']
        self.device = device
        self.with_diffusion = with_diffusion
        self.opt = opt
        self.f = LaplacianODEFunc
        self.block = ConstantODEblock
        self.time_tensor = torch.tensor([0, self.T]).to(device)
        self.meta_mapping = nn.Linear(32, 300)

        self.gnn = GNN(num_layer, emb_dim, JK, drop_ratio, gnn_type)
        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if JK == "concat":
            self.graph_pred_linear = torch.nn.Linear(self.mult * (num_layer + 1) * emb_dim, output_dim)
        else:
            self.graph_pred_linear = torch.nn.Linear(self.mult * emb_dim, output_dim)
        
        self.fm = Meter()
        self.bm = Meter()
    
    def init_diffusion(self, dataset):
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.odeblock = self.block(self.f, self.opt, dataset.data, dataset.train_edge_split_id, self.device, t=self.time_tensor).to(self.device)
        self.meta_node_split_id = dataset.meta_node_split_id

    def forward(self, x_meta, data_for_gnn):
        x, x_edge_index, x_edge_attr, x_batch = data_for_gnn.x, data_for_gnn.edge_index, data_for_gnn.edge_attr, data_for_gnn.batch
        
        x = self.gnn(x, x_edge_index, x_edge_attr)
        x = self.pool(x, x_batch)
        x_meta = self.meta_mapping(x_meta)
        
        z = torch.cat([x_meta, x], dim=0)
        
        if self.with_diffusion:
            self.odeblock.set_x0(z)
            z = self.odeblock(z)
            
        z = self.graph_pred_linear(z)
        out = z[self.meta_node_split_id:, :]
        return out

    def load_from_pretrain(self, path):
        pt = torch.load(path)
        missing_keys, unexpected_keys = self.load_state_dict(pt, strict=False)
        assert(len(unexpected_keys) == 0)

    def getNFE(self):
        return self.odeblock.odefunc.nfe 

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0
        
        
class DiffuseConditionMPNGrammar(MessagePassing):
    def __init__(self, opt, model_args, device, with_diffusion=True):
        super(DiffuseConditionMPNGrammar, self).__init__()
        self.opt = opt
        self.T = opt['time']
        self.device = device
        self.with_diffusion = with_diffusion
        self.opt = opt
        self.f = LaplacianODEFunc
        self.block = ConstantODEblock
        self.time_tensor = torch.tensor([0, self.T]).to(device)
        self.meta_mapping = nn.Linear(32, 300)
        
        if opt['with_condition']:
            model_args.output_size = opt['intermediate_feat_dim']
        else:
            model_args.output_size = 1

        self.mpn = MoleculeFeatModel(model_args)
        self.fnn = self.mpn.create_ffn(model_args)
        self.classification = opt['classification']
        
        self.sigmoid = nn.Sigmoid()
        self.fm = Meter()
        self.bm = Meter()
    
    def init_diffusion(self, dataset):
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.odeblock = self.block(self.f, self.opt, dataset.data, dataset.train_edge_split_id, self.device, t=self.time_tensor).to(self.device)
        self.meta_node_split_id = dataset.meta_node_split_id

    def forward(self, x_meta, data_for_gnn):
        batch = MoleculeDataset(data_for_gnn)
        mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch = \
            batch.batch_graph(), batch.features(), batch.atom_descriptors(), batch.atom_features(), batch.bond_features()
            
        x = self.mpn(mol_batch, features_batch, atom_descriptors_batch, atom_features_batch, bond_features_batch)
        x_meta = self.meta_mapping(x_meta)
        
        z = torch.cat([x_meta, x], dim=0)
        
        if self.with_diffusion:
            self.odeblock.set_x0(z)
            z = self.odeblock(z)
            
        z = self.fnn(z)
        out = z[self.meta_node_split_id:, :]
        return out

    def load_from_pretrain(self, path):
        pt = torch.load(path)
        missing_keys, unexpected_keys = self.load_state_dict(pt, strict=False)
        assert(len(unexpected_keys) == 0)

    def getNFE(self):
        return self.odeblock.odefunc.nfe 

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0

class ConditionPostProcessing(nn.Module):
    def __init__(self, opt):
        super(ConditionPostProcessing, self).__init__()
        self.opt = opt
        self.num_conditons = len(opt['condition_names'])
        
        self.ffn = nn.Sequential(
            nn.Linear(self.num_conditons + self.opt['intermediate_feat_dim'], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, mol_feat, condition):
        assert(mol_feat.shape[0] == condition.shape[0])
        assert(condition.shape[1] == self.num_conditons)
        assert(mol_feat.shape[1] == self.opt['intermediate_feat_dim'])
        z = torch.cat([mol_feat, condition], dim=1)
        out = self.ffn(z)
        return out