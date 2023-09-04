import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from torchdiffeq import odeint_adjoint as odeint
import six


class ODEblock(nn.Module):
  def __init__(self, odefunc, opt, data, device, t):
    super(ODEblock, self).__init__()
    self.opt = opt
    self.t = t
    # self.odefunc = odefunc(opt, data, device)
    self.train_integrator = odeint
    self.test_integrator = None
    self.set_tol()

  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()

  def set_tol(self):
    self.atol = self.opt['tol_scale'] * 1e-7
    self.rtol = self.opt['tol_scale'] * 1e-9
    self.atol_adjoint = self.opt['tol_scale_adjoint'] * 1e-7
    self.rtol_adjoint = self.opt['tol_scale_adjoint'] * 1e-9

  def reset_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9

  def set_time(self, time):
    self.t = torch.tensor([0, time]).to(self.device)

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"


class ODEFunc(MessagePassing):
# class ODEFunc(nn.Module):

  # currently requires in_features = out_features
  def __init__(self, opt, device):
    super(ODEFunc, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = None
    self.edge_weight = None
    self.attention_weights = None
    self.alpha_train = None #nn.Parameter(torch.tensor(0.0))
    self.beta_train = None #nn.Parameter(torch.tensor(0.0))
    self.x0 = None
    self.nfe = 0

  def __repr__(self):
    return self.__class__.__name__
