"""PyTorch forward-formula port of NVIDIA FlowNet2's three custom operators.

Restricted to the configurations used by FlowNet2. See third_party/flownet2/provenance.json.
Forward matches the published CUDA formulas, not necessarily their floating-point order.
Resampling autograd differentiates the forward floor-based interpolation; the old CUDA
input-image backward uses truncation for negative coordinates, so it is not bitwise replayed.
"""
import torch
from torch import nn
from torch.nn import functional as F


class _ChannelNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        y = x.square().sum(1, keepdim=True).sqrt()
        ctx.save_for_backward(x, y)
        return y

    @staticmethod
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        return gradient * x / (y + 1e-9)


class ChannelNorm(nn.Module):
    def __init__(self, norm_deg=2):
        super().__init__()
        if norm_deg != 2:
            raise ValueError("Only FlowNet2's L2 norm is implemented")

    def forward(self, x):
        return _ChannelNorm.apply(x)


class Resample2d(nn.Module):
    def __init__(self, kernel_size=1, bilinear=True):
        super().__init__()
        if kernel_size != 1 or not bilinear:
            raise ValueError("Only FlowNet2's single-point bilinear resampling is implemented")

    def forward(self, image, flow):
        b, c, h, w = image.shape
        if flow.shape != (b, 2, h, w):
            raise ValueError("Image and flow resolution mismatch")
        yy, xx = torch.meshgrid(torch.arange(h, device=flow.device, dtype=flow.dtype),
                                torch.arange(w, device=flow.device, dtype=flow.dtype), indexing="ij")
        x, y = xx[None] + flow[:, 0], yy[None] + flow[:, 1]
        left, top = x.floor(), y.floor()
        alpha, beta = x - left, y - top
        flat = image.flatten(2)
        def sample(dx, dy):
            ix = (left + dx).clamp(0, w - 1).long()
            iy = (top + dy).clamp(0, h - 1).long()
            idx = (iy * w + ix).flatten(1).unsqueeze(1).expand(b, c, h * w)
            return flat.gather(2, idx).reshape(b, c, h, w)
        a, d = alpha[:, None], beta[:, None]
        return (1-a)*(1-d)*sample(0, 0) + a*(1-d)*sample(1, 0) \
            + (1-a)*d*sample(0, 1) + a*d*sample(1, 1)


class _Correlation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, second, displacement, stride, chunk):
        b, c, h, w = first.shape
        pad = F.pad(second, (displacement,) * 4)
        offsets = [(dy, dx) for dy in range(-displacement, displacement+1, stride)
                            for dx in range(-displacement, displacement+1, stride)]
        result = first.new_empty(b, len(offsets), h, w)
        for start in range(0, len(offsets), chunk):
            current = offsets[start:start+chunk]
            shifted = torch.stack([pad[:, :, displacement+dy:displacement+dy+h,
                                            displacement+dx:displacement+dx+w]
                                   for dy, dx in current], dim=1)
            result[:, start:start+len(current)] = (first[:, None] * shifted).mean(2)
        ctx.save_for_backward(first, second)
        ctx.offsets, ctx.displacement, ctx.chunk = offsets, displacement, chunk
        return result

    @staticmethod
    def backward(ctx, gradient):
        first, second = ctx.saved_tensors
        b, c, h, w = first.shape
        d = ctx.displacement
        need_first, need_second = ctx.needs_input_grad[:2]
        grad_first = torch.zeros_like(first) if need_first else None
        grad_second_pad = second.new_zeros(b, c, h+2*d, w+2*d) if need_second else None
        pad = F.pad(second, (d,) * 4) if need_first else None
        for start in range(0, len(ctx.offsets), ctx.chunk):
            offsets = ctx.offsets[start:start+ctx.chunk]
            weighted = gradient[:, start:start+len(offsets), None] / c
            if need_first:
                shifted = torch.stack([pad[:, :, d+dy:d+dy+h, d+dx:d+dx+w]
                                       for dy, dx in offsets], dim=1)
                grad_first.add_((weighted * shifted).sum(1))
            if need_second:
                values = weighted * first[:, None]
                for i, (dy, dx) in enumerate(offsets):
                    grad_second_pad[:, :, d+dy:d+dy+h, d+dx:d+dx+w].add_(values[:, i])
        grad_second = grad_second_pad[:, :, d:d+h, d:d+w] if need_second else None
        return grad_first, grad_second, None, None, None


class Correlation(nn.Module):
    def __init__(self, pad_size=20, kernel_size=1, max_displacement=20,
                 stride1=1, stride2=2, corr_multiply=1):
        super().__init__()
        if (pad_size != max_displacement or kernel_size != 1 or stride1 != 1
                or corr_multiply != 1 or max_displacement % stride2):
            raise ValueError("Unsupported FlowNet2 correlation configuration")
        self.displacement, self.stride = max_displacement, stride2

    def forward(self, first, second):
        if first.shape != second.shape:
            raise ValueError("Correlation inputs must match")
        return _Correlation.apply(first, second, self.displacement, self.stride, 8)
