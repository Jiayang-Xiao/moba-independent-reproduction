"""Same NVIDIA FlowNet2 network and checkpoint, with declared PyTorch operator port."""
import hashlib
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import torch
from torch import nn
from native_flow_ops import ChannelNorm, Resample2d, Correlation

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def network_class():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    base = "third_party.flownet2.networks"
    for name, cls in (("channelnorm", ChannelNorm), ("resample2d", Resample2d), ("correlation", Correlation)):
        package_name = base + "." + name + "_package"
        module_name = package_name + "." + name
        package = ModuleType(package_name)
        package.__path__ = []
        module = ModuleType(module_name)
        setattr(module, cls.__name__, cls)
        sys.modules[package_name] = package
        sys.modules[module_name] = module
    return importlib.import_module("third_party.flownet2.models").FlowNet2


class FrozenFlowNet2(nn.Module):
    def __init__(self, checkpoint, device, untrained_fixture=False):
        super().__init__()
        self.network = network_class()(SimpleNamespace(rgb_max=255.0, fp16=False))
        if untrained_fixture:
            if checkpoint is not None:
                raise ValueError("Fixture must not load production weights")
            self.checkpoint_sha256 = "UNTRAINED_LOCAL_TEST_ONLY"
        else:
            if not checkpoint or not Path(checkpoint).is_file():
                raise FileNotFoundError("Official FlowNet2 checkpoint is required; no fallback estimator")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            weights = saved.get("state_dict", saved)
            weights = {key.removeprefix("module."): value for key, value in weights.items()}
            self.network.load_state_dict(weights, strict=True)
            self.checkpoint_sha256 = sha256(checkpoint)
        self.network.requires_grad_(False)
        self.network.eval()
        self.network.to(device)
        self.eval()

    def train(self, mode=True):
        super().train(False)
        self.network.eval()
        return self

    def forward(self, first, second):
        if first.shape != second.shape or first.ndim != 4 or first.shape[1] != 3:
            raise ValueError("Expected two RGB BCHW images")
        if any(size % 64 for size in first.shape[-2:]):
            raise ValueError("FlowNet2 input resolution must be divisible by64")
        pair = torch.stack(((first+1)*127.5, (second+1)*127.5), dim=2)
        result = self.network(pair)
        if result.shape != (first.shape[0], 2, *first.shape[-2:]):
            raise RuntimeError("Unexpected official FlowNet2 output")
        return result
