"""Independent MoBA candidate V1; unconfirmed choices are in config.json and docs."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


def blur_rgb(frames: torch.Tensor, kernel: int = 3, sigma: float = 5.0) -> torch.Tensor:
    """Gaussian filter on B,T,3,H,W; reflect padding is a declared assumption."""
    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("Expected B,T,3,H,W RGB frames")
    if kernel % 2 != 1 or kernel < 1 or sigma <= 0:
        raise ValueError("Positive odd kernel and positive sigma required")
    b, t, c, h, w = frames.shape
    coord = torch.arange(kernel, device=frames.device, dtype=frames.dtype) - kernel // 2
    one = torch.exp(-coord.square() / (2 * sigma * sigma))
    one = one / one.sum()
    weight = torch.outer(one, one).expand(c, 1, kernel, kernel)
    flat = frames.reshape(b * t, c, h, w)
    flat = F.pad(flat, (kernel // 2,) * 4, mode="reflect")
    return F.conv2d(flat, weight, groups=c).reshape(b, t, c, h, w)


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int, final_linear: bool = False):
        super().__init__()
        layers = [nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                  nn.BatchNorm2d(cout), nn.ReLU()]
        layers.append(nn.Conv2d(cout, cout, 3, padding=1, bias=final_linear))
        if not final_linear:
            layers.extend([nn.BatchNorm2d(cout), nn.ReLU()])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, cin: int, channels: tuple[int, ...]):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBlock(cin if i == 0 else channels[i - 1], ch,
                      final_linear=(i == len(channels) - 1))
            for i, ch in enumerate(channels)
        ])

    def forward(self, x):
        skips = []
        for i, block in enumerate(self.blocks):
            if i:
                x = F.max_pool2d(x, 2)
            x = block(x)
            if i != len(self.blocks) - 1:
                skips.append(x)
        return x, skips


class ChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int):
        super().__init__()
        self.layers = nn.Sequential(nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1), nn.ReLU(),
            nn.Conv2d(max(1, channels // reduction), channels, 1), nn.Sigmoid())

    def forward(self, x):
        return self.layers(x)


class MRCA(nn.Module):
    def __init__(self, channels: int, kernels: tuple[int, int], reduction: int = 16):
        super().__init__()
        k1, k2 = kernels
        self.w1 = nn.Conv2d(channels, channels, k1, padding=k1 // 2)
        self.w2 = nn.Conv2d(channels, channels, k2, padding=k2 // 2)
        self.gate1 = ChannelGate(channels, reduction)
        self.gate2 = ChannelGate(channels, reduction)

    def forward(self, x):
        z1 = self.w1(F.relu(self.w2(x)))
        z2 = self.w2(F.relu(self.w1(x)))
        return x + self.gate1(z1) * z1 + self.gate2(z2) * z2


class MotionMemory(nn.Module):
    def __init__(self, items: int, channels: int):
        super().__init__()
        self.items = nn.Parameter(torch.empty(items, channels))
        nn.init.uniform_(self.items, -1 / math.sqrt(channels), 1 / math.sqrt(channels))

    def forward(self, feature: torch.Tensor):
        b, c, h, w = feature.shape
        q = feature.permute(0, 2, 3, 1).reshape(-1, c)
        cosine = F.normalize(q, dim=1, eps=1e-8) @ F.normalize(self.items, dim=1, eps=1e-8).T
        attention = cosine.softmax(dim=1)
        read = (attention @ self.items).reshape(b, h, w, c).permute(0, 3, 1, 2)
        return read, attention


class Decoder(nn.Module):
    def __init__(self, channels: tuple[int, ...]):
        super().__init__()
        self.bottleneck = ConvBlock(channels[-1], channels[-1])
        self.ups = nn.ModuleList()
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1, 0, -1):
            cin, cout = channels[i], channels[i - 1]
            self.ups.append(nn.Sequential(nn.ConvTranspose2d(cin, cout, 2, stride=2),
                                          nn.BatchNorm2d(cout), nn.ReLU()))
            self.blocks.append(ConvBlock(2 * cout, cout))
        self.output = nn.Sequential(nn.Conv2d(channels[0], 3, 3, padding=1), nn.Tanh())

    def forward(self, x, skips):
        x = self.bottleneck(x)
        for up, block, skip in zip(self.ups, self.blocks, reversed(skips)):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                raise ValueError("Spatial size must be divisible by eight")
            x = block(torch.cat((x, skip), dim=1))
        return self.output(x)


class MoBACandidate(nn.Module):
    def __init__(self, channels=(64, 128, 256, 512), memory_items=10, motion_channels=6,
                 kernels=((3, 7), (3, 5), (3, 3)), reduction=16):
        super().__init__()
        channels = tuple(channels)
        self.motion_channels = motion_channels
        self.appearance = Encoder(12, channels)
        self.zero = nn.Conv2d(motion_channels, motion_channels, 1)
        nn.init.zeros_(self.zero.weight)
        nn.init.zeros_(self.zero.bias)
        self.motion = Encoder(motion_channels, channels)
        self.memory = MotionMemory(memory_items, channels[-1])
        self.attention = nn.ModuleList(MRCA(ch, tuple(ks), reduction)
                                       for ch, ks in zip(channels[:-1], kernels))
        self.fusion = nn.Sequential(nn.Conv2d(2 * channels[-1], channels[-1], 1),
                                    nn.Tanh(), nn.Conv2d(channels[-1], channels[-1], 1))
        self.decoder = Decoder(channels)

    def _appearance_features(self, rgb):
        if rgb.ndim != 5 or tuple(rgb.shape[1:3]) != (4, 3):
            raise ValueError("Expected exactly four RGB frames: B,4,3,H,W")
        if rgb.shape[-1] % 8 or rgb.shape[-2] % 8:
            raise ValueError("Image size must be divisible by eight")
        b, t, c, h, w = rgb.shape
        fa, skips = self.appearance(blur_rgb(rgb).reshape(b, t * c, h, w))
        skips = [block(x) for block, x in zip(self.attention, skips)]
        return fa, skips

    def forward_train(self, rgb, motion):
        """Training memory query is motion, exactly as the paper describes."""
        fa, skips = self._appearance_features(rgb)
        if motion.shape != (rgb.shape[0], self.motion_channels, *rgb.shape[-2:]):
            raise ValueError("Motion channel/shape mismatch")
        # Declared V1 interpretation: a zero-initialized residual adapter.
        # The identity path prevents the V0 all-zero -> BN/ReLU dead input path.
        # This topology is an engineering assumption, NOT author-confirmed MoBA.
        fo, _ = self.motion(motion + self.zero(motion))
        retrieved, attention = self.memory(fo)
        prediction = self.decoder(self.fusion(torch.cat((fa, retrieved), dim=1)), skips)
        return {"prediction": prediction, "fa": fa, "fo": fo,
                "retrieved": retrieved, "attention": attention}

    def forward_test(self, rgb):
        """No motion, fusion or memory write. Caller must set model.eval()."""
        if self.training:
            raise RuntimeError("Set eval() before target inference; BN must remain source-frozen")
        fa, skips = self._appearance_features(rgb)
        retrieved, _ = self.memory(fa)
        return self.decoder(retrieved, skips)

    def deployed_parameter_count(self):
        modules = (self.appearance, self.memory, self.attention, self.decoder)
        return sum(p.numel() for module in modules for p in module.parameters())
