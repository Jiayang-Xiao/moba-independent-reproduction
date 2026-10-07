"""Eq17/18-based independent score; constants are frozen in config, not paper-confirmed."""
from __future__ import annotations
import numpy as np
import torch
from torch.nn import functional as F


def pyramid_psnr(prediction, target, levels, patch, stride):
    if levels is None or patch is None or stride is None:
        raise ValueError("Paper omits pyramid/patch settings: pass explicit documented choices")
    if levels < 1 or patch < 1 or stride < 1:
        raise ValueError("Positive levels, patch and stride required")
    # Prediction/target training range [-1,1]; score errors on [0,1].
    error = ((prediction - target) * 0.5).square().mean(dim=1, keepdim=True)
    peaks = []
    for i in range(levels):
        if patch > min(error.shape[-2:]):
            raise ValueError("Patch exceeds pyramid resolution")
        peak = F.avg_pool2d(error, patch, stride=stride).flatten(1).max(dim=1).values
        peaks.append(peak)
        if i + 1 != levels:
            error = F.avg_pool2d(error, 2, stride=2)
    return -10 * torch.log10(torch.stack(peaks).sum(dim=0).clamp_min(1e-12))


def video_anomaly_scores(psnr, smoothing_sigma=None):
    x = np.asarray(psnr, dtype=np.float64)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("Finite nonempty per-video PSNR required")
    span = x.max() - x.min()
    score = 1 - (x - x.min()) / span if span > 0 else np.full_like(x, 0.5)
    if smoothing_sigma is None:
        raise ValueError("Explicit Gaussian smoothing sigma required; no paper default available")
    if smoothing_sigma < 0:
        raise ValueError("Nonnegative sigma required")
    if smoothing_sigma > 0:
        radius = max(1, int(np.ceil(3 * smoothing_sigma)))
        positions = np.arange(-radius, radius + 1)
        kernel = np.exp(-0.5 * (positions / smoothing_sigma) ** 2)
        kernel /= kernel.sum()
        score = np.convolve(np.pad(score, radius, mode="edge"), kernel, mode="valid")
    return score


def auroc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Aligned finite frame-level labels/scores required")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary")
    positive, negative = (labels == 1).sum(), (labels == 0).sum()
    if not positive or not negative:
        raise ValueError("AUROC requires both normal and abnormal frames")
    order = np.argsort(scores, kind="stable")
    ranked = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(ranked)) + 1, len(ranked)]
    start, rank_sum = 0, 0.0
    for end in ends:
        rank_sum += (labels[order[start:end]] == 1).sum() * ((start + 1 + end) / 2)
        start = end
    return float((rank_sum - positive * (positive + 1) / 2) / (positive * negative))
