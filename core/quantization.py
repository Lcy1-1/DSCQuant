import torch
import torch.nn.functional as F

def ste_round(x):
    return x + (x.round() - x).detach()

def weight_codes(weight, bits, clip=1.0):
    bound = 2 ** (bits - 1) - 1
    scale = (weight.float().abs().amax(dim=1, keepdim=True) * clip / bound).clamp_min(1e-08)
    code = ste_round(weight.float() / scale).clamp(-bound, bound)
    return (code, scale)

def activation_codes(x, bits, clip=1.0):
    xf = x.float()
    lo = torch.minimum(xf.amin(dim=-1, keepdim=True), torch.zeros_like(xf[..., :1])) * clip
    hi = torch.maximum(xf.amax(dim=-1, keepdim=True), torch.zeros_like(xf[..., :1])) * clip
    maximum = 2 ** bits - 1
    scale = ((hi - lo) / maximum).clamp_min(1e-08)
    zero = ste_round(-lo / scale).clamp(0, maximum)
    code = ste_round(xf / scale + zero).clamp(0, maximum)
    return (code, scale, zero)

def quantize_weight(weight, bits, clip=1.0):
    code, scale = weight_codes(weight, bits, clip)
    return (code * scale).to(weight.dtype)

def quantize_activation(x, bits, clip=1.0):
    code, scale, zero = activation_codes(x, bits, clip)
    return ((code - zero) * scale).to(x.dtype)

def pack_int4(codes):
    if codes.ndim != 2:
        raise ValueError('Packed weights must be a matrix.')
    codes = codes.to(torch.int8)
    if bool(((codes < -8) | (codes > 7)).any()):
        raise ValueError('Signed INT4 code out of range.')
    if codes.shape[1] % 2:
        codes = F.pad(codes, (0, 1))
    unsigned = codes.to(torch.int16) & 15
    return (unsigned[:, ::2] | unsigned[:, 1::2] << 4).to(torch.uint8)

def unpack_int4(packed, width):
    code = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(packed.shape[0], -1)
    signed = torch.where(code >= 8, code.to(torch.int16) - 16, code.to(torch.int16))
    return signed[:, :width].to(torch.int8)
