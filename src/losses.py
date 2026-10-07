"""Five paper losses. Motion prediction must stay differentiable; no silent omission."""
import torch
from torch.nn import functional as F


def five_losses(output, future_rgb, predicted_motion, true_motion):
    prediction = output["prediction"]
    if predicted_motion.shape != true_motion.shape:
        raise ValueError("Future motion tensors must have matching shapes")
    if torch.is_grad_enabled() and prediction.requires_grad and not predicted_motion.requires_grad:
        raise ValueError("Predicted motion was detached; L_motion cannot supervise the RGB predictor")
    dx = lambda x: x[..., :, 1:] - x[..., :, :-1]
    dy = lambda x: x[..., 1:, :] - x[..., :-1, :]
    losses = {
        "pred": F.mse_loss(prediction, future_rgb),
        "grad": F.l1_loss(dx(prediction).abs(), dx(future_rgb).abs())
                + F.l1_loss(dy(prediction).abs(), dy(future_rgb).abs()),
        "sim": (1 - F.cosine_similarity(output["fa"], output["fo"], dim=1, eps=1e-8)).mean(),
        "motion": F.l1_loss(predicted_motion, true_motion.detach()),
        "comp": F.mse_loss(output["fo"], output["retrieved"]),
    }
    if not all(torch.isfinite(x).item() for x in losses.values()):
        raise FloatingPointError("Nonfinite paper loss")
    return sum(losses.values()), losses


def source_update(model, teacher, optimizer, five_rgb_frames):
    """One source-only update with a provisional frozen FlowNet2 interpretation.

    Six encoder channels concatenate flow(0,1), flow(1,2), flow(2,3).
    Future target flow(3,4); predicted future flow(3,prediction).
    This interpretation is NOT specified by the paper and needs author confirmation.
    """
    if five_rgb_frames.ndim != 5 or tuple(five_rgb_frames.shape[1:3]) != (5, 3):
        raise ValueError("Expected B,5,3,H,W source-only RGB clip")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        motion = torch.cat([teacher(five_rgb_frames[:, i], five_rgb_frames[:, i + 1])
                            for i in range(3)], dim=1)
        future_motion = teacher(five_rgb_frames[:, 3], five_rgb_frames[:, 4])
    output = model.forward_train(five_rgb_frames[:, :4], motion)
    predicted_motion = teacher(five_rgb_frames[:, 3], output["prediction"])
    total, parts = five_losses(output, five_rgb_frames[:, 4], predicted_motion, future_motion)
    total.backward()
    gradient = model.zero.weight.grad
    motion_gradient = None if gradient is None else float(gradient.detach().norm())
    if motion_gradient is None or not torch.isfinite(gradient).all().item() or motion_gradient == 0:
        optimizer.zero_grad(set_to_none=True)
        raise RuntimeError("Provisional zero-conv/motion encoder has no finite nonzero gradient; "
                           "stop rather than label a dead-motion model a MoBA reproduction")
    optimizer.step()
    return {key: float(value.detach()) for key, value in parts.items()} | {
        "total": float(total.detach()), "zero_conv_grad_norm": motion_gradient}
