from __future__ import annotations

import torch
import torch.nn as nn


class FixedExtremeWeatherLoss(nn.Module):
    """
    Reviewer-safe weighted MSE using fixed training-period thresholds.

    This replaces batch-dependent percentile thresholds, which can introduce a
    non-stationary optimization objective.
    """

    def __init__(self, low_threshold: float, high_threshold: float, alpha_low: float = 2.0, alpha_high: float = 2.0, beta: float = 1.0):
        super().__init__()
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.alpha_low = float(alpha_low)
        self.alpha_high = float(alpha_high)
        self.beta = float(beta)

    def forward(self, predictions, targets):
        base = (predictions - targets) ** 2
        weights = torch.full_like(targets, self.beta)
        weights = torch.where(targets <= self.low_threshold, torch.full_like(weights, self.alpha_low), weights)
        weights = torch.where(targets >= self.high_threshold, torch.full_like(weights, self.alpha_high), weights)
        return torch.mean(weights * base)
