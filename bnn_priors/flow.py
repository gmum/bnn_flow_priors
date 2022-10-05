import torch
import torch.nn as nn

from bnn_priors.exp_utils import device


class RealNVP(nn.Module):
    def __init__(self, net_s, net_t, num_layers, prior):
        super().__init__()
        self.prior = prior
        self.t = nn.ModuleList([net_t() for _ in range(num_layers)])
        self.s = nn.ModuleList([net_s() for _ in range(num_layers)])
        self.num_flows = num_layers

    def coupling(self, x, index, forward=True):
        (xa, xb) = torch.chunk(x, 2, 1)
        s = self.s[index](xa)
        t = self.t[index](xa)
        if forward:
            yb = (xb - t) * torch.exp(-s)
        else:
            yb = s.exp() * xb + t
        return torch.cat((xa, yb), 1), s, t

    def f(self, x):
        log_det_J, z = x.new_zeros(x.shape[0]), x
        for i in range(self.num_flows):
            z, s, _ = self.coupling(z, i, forward=True)
            z = z.flip(1)
            log_det_J = log_det_J - s.sum(dim=1)
        return z, log_det_J

    def f_inv(self, z):
        x = z
        log_det_J = x.new_zeros(x.shape[0])
        for i in reversed(range(self.num_flows)):
            x = x.flip(1)
            x, s, _ = self.coupling(x, i, forward=False)
            log_det_J = log_det_J + s.sum(dim=1)
        return x, log_det_J

    def forward(self, x):
        z, log_det_J = self.f(x)
        return self.prior.log_prob(z) + log_det_J

    def sample(self, batchSize, D, calculate_nll=False):
        z = self.prior.sample((batchSize,))
        z = z.to(device("try_cuda"))
        x, log_det_J = self.f_inv(z)
        if calculate_nll:
            log_prob_z = self.prior.log_prob(z)
            nll = -(log_prob_z - log_det_J)
            return x.view(-1, D), nll
        return x.view(-1, D)
