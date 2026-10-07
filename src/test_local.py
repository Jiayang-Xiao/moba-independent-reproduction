"""Operator oracles, true-architecture gradient checks, and interruption/resume fixture."""
import argparse
import copy
import json
from pathlib import Path
import random
import tempfile
import numpy as np
import torch
from torch import nn
from common import seed_everything, save_json
from flow import FrozenFlowNet2
from losses import five_losses
from model import MoBACandidate, MotionMemory
from native_flow_ops import ChannelNorm, Correlation, Resample2d
from scoring import auroc, video_anomaly_scores
from train import update, snapshot, write_checkpoint, restore_rng


class FixtureMotion(nn.Module):
    """Test-only differentiable function, NEVER used by the production launcher."""
    def forward(self, first, second):
        return (second-first)[:, :2]


def tests(device="cpu", full=False):
    seed_everything(17); checks = []
    first = torch.randn(1, 3, 4, 5, dtype=torch.float64, requires_grad=True)
    second = torch.randn_like(first, requires_grad=True)
    correlation = Correlation(pad_size=2, max_displacement=2, stride2=1)
    result = correlation(first, second)
    a, b = first.detach().numpy(), second.detach().numpy()
    expected = np.zeros((1, 25, 4, 5))
    for k, (dy, dx) in enumerate(( (dy,dx) for dy in range(-2,3) for dx in range(-2,3) )):
        for y in range(4):
            for x in range(5):
                if 0 <= y+dy < 4 and 0 <= x+dx < 5:
                    expected[0,k,y,x] = np.mean(a[0,:,y,x] * b[0,:,y+dy,x+dx])
    assert np.allclose(result.detach().numpy(), expected, atol=1e-12)
    assert torch.autograd.gradcheck(correlation, (first, second), fast_mode=True)
    checks.append("Correlation displacement order/channel mean/border agrees with independent NumPy oracle and numerical derivatives")

    image = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
    flow = (torch.rand(1,2,3,4,dtype=torch.float64)-0.5).requires_grad_()
    resample = Resample2d(); expected = np.zeros((1,2,3,4))
    image_np, flow_np = image.detach().numpy(), flow.detach().numpy()
    for y in range(3):
        for x in range(4):
            xx, yy = x+flow_np[0,0,y,x], y+flow_np[0,1,y,x]
            l, t = int(np.floor(xx)), int(np.floor(yy)); ax, by = xx-l, yy-t
            for dx,dy,weight in ((0,0,(1-ax)*(1-by)),(1,0,ax*(1-by)),(0,1,(1-ax)*by),(1,1,ax*by)):
                expected[0,:,y,x] += weight * image_np[0,:,np.clip(t+dy,0,2),np.clip(l+dx,0,3)]
    assert np.allclose(resample(image, flow).detach().numpy(), expected, atol=1e-12)
    assert torch.autograd.gradcheck(resample, (image, flow), fast_mode=True)
    checks.append("Warp floor-based bilinear/border agrees with NumPy oracle and numerical derivatives")
    norm = ChannelNorm(); value = torch.randn(1,3,4,4,dtype=torch.float64,requires_grad=True)
    assert torch.autograd.gradcheck(norm, (value,), fast_mode=True)
    zero = torch.zeros(1,3,2,2,requires_grad=True); norm(zero).sum().backward()
    assert torch.isfinite(zero.grad).all() and zero.grad.abs().sum() == 0
    checks.append("ChannelNorm exact forward and stabilized CUDA-formula backward at zero")

    model = MoBACandidate(channels=(8,16,32,64)).to(device).train()
    rgb = torch.rand(2,5,3,64,64,device=device)*2-1
    teacher = FixtureMotion()
    motion = torch.cat([teacher(rgb[:,i],rgb[:,i+1]) for i in range(3)],1)
    output = model.forward_train(rgb[:,:4],motion)
    total, parts = five_losses(output,rgb[:,4],teacher(rgb[:,3],output['prediction']),teacher(rgb[:,3],rgb[:,4]))
    total.backward()
    gradients = {"zero_conv":float(model.zero.weight.grad.norm()),
                 "motion_first_conv":float(model.motion.blocks[0].layers[0].weight.grad.norm()),
                 "memory":float(model.memory.items.grad.norm())}
    assert all(np.isfinite(x) and x>0 for x in gradients.values())
    checks.append("V1 residual-zero adapter gives finite nonzero motion/zero/memory gradients")
    model.eval(); old = {k:v.clone() for k,v in model.state_dict().items()}
    motion_calls=[]; fusion_calls=[]
    h1=model.motion.register_forward_hook(lambda *x:motion_calls.append(1))
    h2=model.fusion.register_forward_hook(lambda *x:fusion_calls.append(1))
    with torch.no_grad(): prediction=model.forward_test(rgb[:,:4])
    h1.remove();h2.remove()
    assert not motion_calls and not fusion_calls and prediction.shape==(2,3,64,64)
    assert all(torch.equal(v,old[k]) for k,v in model.state_dict().items())
    checks.append("RGB-only target path bypasses motion/fusion and preserves all source state")
    assert auroc([0,1,0,1],[0,1,0,1])==1 and auroc([0,1],[0,0])==0.5
    assert np.allclose(video_anomaly_scores([1,2,3],0),[1,.5,0])
    checks.append("AUROC ties and anomaly-score polarity")

    seed_everything(19)
    a=MoBACandidate(channels=(8,16,32,64)).to(device).train()
    opt=torch.optim.Adam(a.parameters(),lr=.0002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=2)
    future=teacher(rgb[:,3],rgb[:,4])
    update(a,teacher,opt,rgb,motion,future)
    with tempfile.TemporaryDirectory() as directory:
        path=Path(directory)/"last.pt"
        write_checkpoint(path,snapshot(a,opt,scheduler,{"fixture":True},0,1,1))
        update(a,teacher,opt,rgb.flip(0),motion.flip(0),future.flip(0))
        b=MoBACandidate(channels=(8,16,32,64)).to(device).train()
        opt2=torch.optim.Adam(b.parameters(),lr=.0002)
        sch2=torch.optim.lr_scheduler.CosineAnnealingLR(opt2,T_max=2)
        saved=torch.load(path,map_location=device,weights_only=False)
        b.load_state_dict(saved['model']);opt2.load_state_dict(saved['optimizer']);sch2.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'])
        update(b,teacher,opt2,rgb.flip(0),motion.flip(0),future.flip(0))
        assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    checks.append("Serialized model/optimizer/RNG interruption-resume matches uninterrupted fixture bitwise")

    true_architecture=None
    if full:
        # Test the actual 162,518,834-parameter FlowNet2 architecture with random weights.
        # This is NOT a checkpoint/quality test and does not replace production pretrained weights.
        fnet=FrozenFlowNet2(None,device,untrained_fixture=True)
        first=torch.rand(1,3,64,64,device=device)*2-1
        second=(torch.rand(1,3,64,64,device=device)*2-1).requires_grad_()
        flow=fnet(first,second);flow.square().mean().backward()
        assert flow.shape==(1,2,64,64) and torch.isfinite(second.grad).all() and float(second.grad.norm())>0
        assert all(not p.requires_grad and p.grad is None for p in fnet.parameters())
        true_architecture={"teacher_parameters":sum(p.numel() for p in fnet.parameters()),
            "output_shape":list(flow.shape),"rgb_gradient_norm":float(second.grad.norm()),
            "weights":"untrained architecture fixture only"}
        del fnet
        full_model=MoBACandidate().to(device).train()
        rgb256=torch.rand(1,5,3,256,256,device=device)*2-1
        motion256=torch.cat([teacher(rgb256[:,i],rgb256[:,i+1]) for i in range(3)],1)
        out256=full_model.forward_train(rgb256[:,:4],motion256)
        loss256,_=five_losses(out256,rgb256[:,4],teacher(rgb256[:,3],out256['prediction']),teacher(rgb256[:,3],rgb256[:,4]))
        loss256.backward()
        assert float(full_model.zero.weight.grad.norm())>0 and torch.isfinite(full_model.zero.weight.grad).all()
        true_architecture['full_moba_zero_gradient_norm']=float(full_model.zero.weight.grad.norm())
        true_architecture['full_moba_motion_first_gradient_norm']=float(full_model.motion.blocks[0].layers[0].weight.grad.norm())
        checks.append("Actual FlowNet2 architecture frozen-weight forward/backward and full256 MoBA candidate backward")
    return {"status":"PASS","identity":"MOBA_INDEPENDENT_FLOW_V1_20261005","device":device,
        "checks":checks,"v1_gradient_norms":gradients,"full_architecture_checks":true_architecture,
        "real_pretrained_teacher_tested_locally":False,"real_datasets_tested_locally":False,
        "scope":"synthetic implementation tests; no baseline AUROC or official reproduction claim"}


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cpu');p.add_argument('--full',action='store_true')
    p.add_argument('--output',default='local_checks.json');args=p.parse_args()
    result=tests(args.device,args.full);save_json(args.output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
