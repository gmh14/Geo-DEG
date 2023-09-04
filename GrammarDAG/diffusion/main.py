import argparse
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch
import torch_sparse

from diffusion_model import Diffuse
from diffusion_scalar_model import DiffuseScalar
from utils import print_model_params
from dataset import DAGDataset

def get_optimizer(name, parameters, lr, weight_decay=0):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamax':
        return torch.optim.Adamax(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise Exception("Unsupported optimizer: {}".format(name))


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    # feat = data.x * data.train_mask + 0.1 * (~data.train_mask)
    # feat = torch.zeros_like(data.x)
    # feat[data.root_id] = 10
    feat = data.x_init
    out = model(feat)

    lf = torch.nn.MSELoss()
    loss = lf(out * data.train_mask, data.y * data.train_mask)
    # loss = lf(out * data.test_mask, data.y * data.test_mask)

    model.fm.update(model.getNFE())
    model.resetNFE()
    loss.backward()
    # import pdb; pdb.set_trace()
    optimizer.step()
    model.bm.update(model.getNFE())
    model.resetNFE()
    return loss.item()


@torch.no_grad()
def test(model, data):  # opt required for runtime polymorphism
    model.eval()
    # feat = data.x * data.train_mask + 0.1 * (~data.train_mask)
    # feat = torch.zeros_like(data.x)
    # feat[data.root_id] = 10
    feat = data.x_init
    out = model(feat)
    lf = torch.nn.MSELoss()
    loss = lf(out * data.test_mask, data.y * data.test_mask)
    # import pdb; pdb.set_trace()
    return loss


def main(cmd_opt):
    opt = cmd_opt
    dataset = DAGDataset()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Diffuse(opt, dataset, device)
    # model = DiffuseScalar(opt, dataset, device) # Changed

    data = dataset.data.to(device)

    all_train_loss = []
    all_test_loss = []
    parameters = [p for p in model.parameters() if p.requires_grad]
    print_model_params(model)
    optimizer = get_optimizer(opt['optimizer'], parameters, lr=opt['lr'], weight_decay=opt['decay'])
    best_time = best_epoch = 0
    test_loss = np.inf
    # tmp_test_loss = test(model, data)
    # import pdb; pdb.set_trace()

    # all_edge_index = model.odeblock.odefunc.edge_index.clone()
    # transpose_edge_index = torch.tensor([model.odeblock.odefunc.edge_index[1].numpy(), model.odeblock.odefunc.edge_index[0].numpy()]).to(model.odeblock.odefunc.device)
    # all_edge_index = torch.cat([all_edge_index, transpose_edge_index], dim=1)
    # all_L_param = torch.cat([model.odeblock.odefunc.L_param, model.odeblock.odefunc.L_param], dim=0)
    # model.odeblock.odefunc.final_edge_index, attention_weights = torch_sparse.coalesce(all_edge_index, all_L_param, m=data.x.shape[0], n=data.x.shape[0])
    # model.odeblock.odefunc.attention_weights = attention_weights / 2.0
    # L = np.zeros([data.x.size(0), data.x.size(0)])
    # row_idx, col_idx = model.odeblock.odefunc.final_edge_index.detach().numpy()
    # coeff = model.odeblock.odefunc.attention_weights.detach().numpy()
    # L[row_idx, col_idx] = coeff
    # L = L - np.identity(data.x.size(0))
    # w, v = np.linalg.eig(L)
    # a = np.linalg.inv(v) @ ((data.x).detach().numpy())
    # final_a = a * np.expand_dims(np.exp(1 * w), -1)
    # final_x = v @ final_a
    # final_x = v @ (np.identity(data.x.size(0)) * np.exp(w)) @ np.linalg.inv(v) @ ((data.x).detach().numpy())
    # L_tilde = v @ (np.identity(data.x.size(0)) * np.exp(w)) @ np.linalg.inv(v)
    # data.x_init = torch.tensor(np.linalg.inv(L_tilde) @ data.x.detach().numpy(), dtype=torch.float32, device=device)
    # import pdb; pdb.set_trace()
    # x_init = torch.zeros_like(data.x)
    # x_init[data.root_id] = 10
    x_init = data.x # * data.train_mask
    # x_init = data.x * data.train_mask # Changed
    data.x_init = x_init

    for epoch in range(1, opt['epoch']):
        start_time = time.time()

        loss = train(model, optimizer, data)
        tmp_test_loss = test(model, data)
        # print(model.odeblock.odefunc.L_param[0])
        # print(model.odeblock.odefunc.attention_weights)
        if epoch % 5 == -1:
            out = model(data.x_init)
            import pdb; pdb.set_trace()
        best_time = opt['time']
        if tmp_test_loss < test_loss:
            best_epoch = epoch
            test_loss = tmp_test_loss
            best_time = opt['time']

        all_train_loss.append(loss * 10000)
        all_test_loss.append(tmp_test_loss * 10000)
        log = 'Epoch: {:03d}, Runtime {:03f}, Loss {:03f}, forward nfe {:d}, backward nfe {:d}, Test: {:.4f}, Best time: {:.4f}'

        print(log.format(epoch, time.time() - start_time, loss * 10000, model.fm.sum, model.bm.sum, tmp_test_loss * 10000, best_time))
    print('best test accuracy {:03f} at epoch {:d} and best time {:03f}'.format(test_loss,
                                                                                best_epoch,
                                                                                best_time))
    np.save('scalar_train_loss_2.npy', np.array(all_train_loss))
    np.save('scalar_test_loss_2.npy', np.array(all_test_loss))
    model.eval()
    out = model(data.x_init)
    import pdb; pdb.set_trace()
    L = np.zeros([data.x.size(0), data.x.size(0)])
    row_idx, col_idx = model.odeblock.odefunc.final_edge_index.detach().numpy()
    coeff = model.odeblock.odefunc.attention_weights.detach().numpy()
    L[row_idx, col_idx] = coeff
    L = L - np.identity(data.x.size(0))
    w, v = np.linalg.eig(L)
    a = np.linalg.inv(v) @ ((data.x * data.train_mask).detach().numpy())
    final_a = a * np.expand_dims(np.exp(1 * w), -1)
    final_x = v @ final_a
    final_x = v @ (np.identity(data.x.size(0)) * np.exp(w)) @ np.linalg.inv(v) @ ((data.x * data.train_mask).detach().numpy())
    model.eval()
    feat = data.x * data.train_mask + 0.1 * (~data.train_mask)
    out = model(feat)

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data args
    parser.add_argument('--data_norm', type=str, default='rw',
                        help='rw for random walk, gcn for symmetric gcn norm')
    parser.add_argument('--self_loop_weight', type=float, default=1.0, help='Weight of self-loops.')
    # GNN args
    parser.add_argument('--fc_out', dest='fc_out', action='store_true',
                        help='Add a fully connected layer to the decoder.')
    parser.add_argument('--optimizer', type=str, default='adam', help='One from sgd, rmsprop, adam, adagrad, adamax.')
    parser.add_argument('--lr', type=float, default=1e-1, help='Learning rate.')
    parser.add_argument('--decay', type=float, default=5e-4, help='Weight decay for optimization')
    parser.add_argument('--epoch', type=int, default=300, help='Number of training epochs per iteration.')
    parser.add_argument('--block', type=str, default='constant', help='constant, mixed, attention, hard_attention')
    parser.add_argument('--no_alpha_sigmoid', dest='no_alpha_sigmoid', action='store_true',
                        help='apply sigmoid before multiplying by alpha')
    parser.add_argument('--add_source', dest='add_source', action='store_true',
                        help='If try get rid of alpha param and the beta*x0 source term')

    # ODE args
    parser.add_argument('--time', type=float, default=1.0, help='End time of ODE integrator.')
    parser.add_argument('--step_size', type=float, default=1,
                        help='fixed step size when using fixed step solvers e.g. rk4')
    parser.add_argument('--max_iters', type=float, default=100, help='maximum number of integration steps')
    parser.add_argument("--adjoint_method", type=str, default="adaptive_heun",
                        help="set the numerical solver for the backward pass: dopri5, euler, rk4, midpoint")
    parser.add_argument('--adjoint_step_size', type=float, default=1,
                        help='fixed step size when using fixed step adjoint solvers e.g. rk4')
    parser.add_argument('--tol_scale', type=float, default=1., help='multiplier for atol and rtol')
    parser.add_argument("--tol_scale_adjoint", type=float, default=1.0,
                        help="multiplier for adjoint_atol and adjoint_rtol")
    parser.add_argument("--max_nfe", type=int, default=1000,
                        help="Maximum number of function evaluations in an epoch. Stiff ODEs will hang if not set.")

    args = parser.parse_args()

    opt = vars(args)

    main(opt)
