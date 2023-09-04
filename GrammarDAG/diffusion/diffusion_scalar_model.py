import torch
from torch import nn
import torch.nn.functional as F
import torch_sparse
from torchdiffeq import odeint_adjoint as odeint
from torch_geometric.nn.conv import MessagePassing
import numpy as np

from utils import get_rw_adj, gcn_norm_fill_val, Meter
from base_classes import ODEblock, ODEFunc

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class LaplacianODEFunc(ODEFunc):

    # currently requires in_features = out_features
    def __init__(self, opt, data, edge_index, edge_weight, device):
        super(LaplacianODEFunc, self).__init__(opt, device)
        self.device = device
        # self.alpha_train = nn.Parameter(torch.ones_like(data.x))
        self.alpha_train = torch.ones([data.x.shape[0], 1])

        # self.L_param = nn.Parameter(edge_weight.repeat(11, 1))
        self.L_param = nn.Parameter(edge_weight)
        # self.L_param = edge_weight
        self.beta_train = nn.Parameter(torch.ones([data.x.shape[0], 1]))

        self.edge_index = edge_index.to(device)
        self.edge_weight = edge_weight.to(device)

    def sparse_multiply(self, x, t):
        if self.opt['block'] in ['attention']:  # adj is a multihead attention
            mean_attention = self.attention_weights.mean(dim=1)
            ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
        elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
            ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
        elif self.opt['block'] in ['my_attention']:  # adj is a torch sparse matrix
            # all_edge_index = self.edge_index.clone()
            # transpose_edge_index = torch.tensor([self.edge_index[1].numpy(), self.edge_index[0].numpy()]).to(self.device)
            # all_edge_index = torch.cat([all_edge_index, transpose_edge_index], dim=1)
            # idx = int(torch.floor(t * 10).item())
            # all_L_param = torch.cat([self.L_param[idx, :], self.L_param[idx, :]], dim=0)
            # self.final_edge_index, attention_weights = torch_sparse.coalesce(all_edge_index, all_L_param, m=x.shape[0], n=x.shape[0])
            # self.attention_weights = attention_weights / 2.0
            # idx = int(torch.floor(t / self.T * 10).item())
            # idx = 0
            # self.final_edge_index = self.edge_index
            # self.attention_weights = self.L_param[idx, :]
            self.final_edge_index = self.edge_index
            self.attention_weights = self.L_param
            # L = np.zeros([x.size(0), x.size(0)])
            # row_idx, col_idx = self.final_edge_index.detach().numpy()
            # coeff = self.attention_weights.detach().numpy()
            # L[row_idx, col_idx] = coeff
            # print(np.sum(np.abs(L - L.transpose())))
            ax = torch_sparse.spmm(self.final_edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
        else:  # adj is a torch sparse matrix
            ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
        return ax

    def forward(self, t, x):  # the t param is needed by the ODE solver.
        # if self.nfe > self.opt["max_nfe"]:
        #     raise Exception("Max number of function evaluations reached.")
        self.nfe += 1
        ax = self.sparse_multiply(x, t)
        if not self.opt['no_alpha_sigmoid']:
            alpha = torch.sigmoid(self.alpha_train)
        else:
            alpha = self.alpha_train

        f = alpha * (ax - x)
        if self.opt['add_source']:
            f = f + self.beta_train * self.x0
        return f


class ConstantODEblock(ODEblock):
    def __init__(self, odefunc, opt, data, device, t=torch.tensor([0, 1])):
        super(ConstantODEblock, self).__init__(odefunc, opt, data, device, t)

        if opt['data_norm'] == 'rw':
            edge_index, edge_weight = get_rw_adj(data.edge_index, edge_weight=data.edge_attr, norm_dim=1,
                                                                        fill_value=opt['self_loop_weight'],
                                                                        num_nodes=data.num_nodes,
                                                                        dtype=torch.float64)
        else:
            edge_index, edge_weight = gcn_norm_fill_val(data.edge_index, edge_weight=data.edge_attr,
                                                fill_value=opt['self_loop_weight'],
                                                num_nodes=data.num_nodes,
                                                dtype=torch.float64)
        self.odefunc = odefunc(opt, data, edge_index, edge_weight, device)
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


class DiffuseScalar(MessagePassing):
    def __init__(self, opt, dataset, device):
        super(DiffuseScalar, self).__init__()
        self.opt = opt
        self.T = opt['time']
        self.num_features = dataset.data.num_features
        self.num_nodes = dataset.data.num_nodes
        self.device = device
        self.f = LaplacianODEFunc
        block = ConstantODEblock
        # time_tensor = torch.tensor([0, self.T]).to(device)
        time_tensor = torch.tensor([0, self.T]).to(device)
        self.odeblock = block(self.f, opt, dataset.data, device, t=time_tensor).to(device)
        # self.embedding = torch.nn.Embedding(16, 3)
        # self.fc1 = nn.Linear(dataset.data.x.shape[1], 32)
        # self.fc1 = nn.Linear(dataset.data.x.shape[1] * 3, 32)
        # self.bn = nn.BatchNorm1d(32)
        # self.fc2 = nn.Linear(32, 1)
        self.fm = Meter()
        self.bm = Meter()

    def forward(self, x):
        # x = self.embedding(x)
        # x = torch.reshape(x, [-1, x.shape[1] * x.shape[2]])
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        # z = x
        # import pdb; pdb.set_trace()
        # z = F.relu(z)
        # z = self.fc1(z)
        # z = self.bn(z)
        # z = F.relu(z)
        # z = self.fc2(z)
        # import pdb; pdb.set_trace()
        return z

    def getNFE(self):
        return self.odeblock.odefunc.nfe 

    def resetNFE(self):
        self.odeblock.odefunc.nfe = 0