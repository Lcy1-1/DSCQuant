import weakref
import torch
import torch.nn.functional as F
from torch import nn
from .transforms import Rotation
from .quantization import quantize_activation, quantize_weight

class QuantLinear(nn.Module):

    def __init__(self, base, config, interface=None, role=None, repeat=1):
        super().__init__()
        self.base = base
        self.config = config
        self.role, self.repeat = (role, repeat)
        self._interface = None if interface is None else weakref.ref(interface)
        self.weight_clip = nn.Parameter(torch.tensor(1.0, device=base.weight.device), requires_grad=False)
        self.activation_clip = nn.Parameter(torch.tensor(1.0, device=base.weight.device), requires_grad=False)
        self.enabled = False
        for param in self.base.parameters():
            param.requires_grad_(False)

    def transformed(self):
        weight, bias = (self.base.weight, self.base.bias)
        interface = self._interface() if self._interface else None
        if interface is None:
            return (weight, bias)
        if isinstance(interface, Rotation):
            matrices = interface.matrices()
            if self.role == 'v':
                shaped = weight.float().reshape(interface.heads, interface.head_dim, -1)
                weight = (matrices.transpose(-1, -2) @ shaped).reshape_as(weight).to(weight.dtype)
                if bias is not None:
                    bias = (matrices.transpose(-1, -2) @ bias.float().reshape(interface.heads, -1, 1)).flatten().to(bias.dtype)
            elif self.role == 'o':
                matrices = matrices.repeat_interleave(self.repeat, dim=0)
                shaped = weight.float().reshape(weight.shape[0], len(matrices), interface.head_dim)
                weight = torch.einsum('ohd,hde->ohe', shaped, matrices).reshape_as(weight).to(weight.dtype)
        else:
            scale = interface.diagonal()
            if self.role == 'up':
                weight = (weight.float() / scale[:, None]).to(weight.dtype)
                if bias is not None:
                    bias = (bias.float() / scale).to(bias.dtype)
            else:
                weight = (weight.float() * scale[None, :]).to(weight.dtype)
        return (weight, bias)

    def forward(self, x):
        if not self.enabled:
            return self.base(x)
        weight, bias = self.transformed()
        return F.linear(quantize_activation(x, self.config.activation_bits, self.activation_clip), quantize_weight(weight, self.config.weight_bits, self.weight_clip), bias)
