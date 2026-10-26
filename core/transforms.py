import math
import torch
from torch import nn
from .subspace import orth

class Interface(nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self.dimension = dimension
        self.register_buffer('rec_basis', torch.empty(dimension, 0))
        self.register_buffer('sid_basis', torch.empty(dimension, 0))
        self.alpha = nn.Parameter(torch.zeros(0))
        self.beta = nn.Parameter(torch.zeros(0))
        self.probe = None
        self.override = None

    def set_basis(self, name, basis):
        basis = basis.to(device=self.rec_basis.device, dtype=torch.float32)
        setattr(self, name + '_basis', basis)
        setattr(self, 'alpha' if name == 'rec' else 'beta', nn.Parameter(basis.new_zeros(basis.shape[1])))

    def increment(self):
        if self.override is not None:
            return self.override
        if self.probe is not None:
            return self.probe
        return self.sid_basis @ self.beta

    def regularizer(self):
        raise NotImplementedError

class Rotation(Interface):

    def __init__(self, heads, head_dim):
        self.heads, self.head_dim = (heads, head_dim)
        self.per_head = head_dim * (head_dim - 1) // 2
        super().__init__(heads * self.per_head)
        indices = torch.triu_indices(head_dim, head_dim, offset=1)
        self.register_buffer('upper', indices, persistent=False)

    def skew(self, coordinate):
        x = coordinate.reshape(self.heads, self.per_head) / math.sqrt(2)
        out = coordinate.new_zeros(self.heads, self.head_dim, self.head_dim)
        out[:, self.upper[0], self.upper[1]] = x
        return out - out.transpose(-1, -2)

    def coefficients(self, matrix):
        return ((matrix[:, self.upper[0], self.upper[1]] - matrix[:, self.upper[1], self.upper[0]]) / math.sqrt(2)).flatten()

    def matrices(self):
        return torch.matrix_exp(self.skew(self.rec_basis @ self.alpha)) @ torch.matrix_exp(self.skew(self.increment()))

    def reconstruction_tangent(self):
        fitted = (self.rec_basis @ self.alpha).detach()
        r = torch.matrix_exp(self.skew(fitted))
        columns = []
        for u in self.rec_basis.T:
            _, dr = torch.autograd.functional.jvp(lambda x: torch.matrix_exp(self.skew(x)), fitted, u)
            columns.append(self.coefficients(r.transpose(-1, -2) @ dr))
        return orth(torch.stack(columns, dim=1)) if columns else self.rec_basis

    def regularizer(self):
        return self.beta.square().mean() if self.beta.numel() else self.alpha.sum() * 0

class Scaling(Interface):

    def project(self, x):
        return x - x.mean(dim=0, keepdim=True)

    def diagonal(self):
        return torch.exp(self.project(self.rec_basis @ self.alpha + self.increment()))

    def reconstruction_tangent(self):
        return self.rec_basis

    def regularizer(self):
        return (self.sid_basis @ self.beta).square().mean()
