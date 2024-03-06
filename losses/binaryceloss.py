import torch.nn.functional as F
import torch.nn as nn
import torch

class BinaryCELoss(nn.Module):

    def __init__(self, weights: float | list | None = None, label_smoothing: float = 0.0, reduction: str = 'mean') -> None:

        super().__init__()
        if weights is not None:
            weights = torch.Tensor([1 - weights, weights]) if isinstance(weights, float) else torch.Tensor(weights)
            inv_weights = weights ** -1
            self.weights = inv_weights / (weights[0] * inv_weights[0] + weights[1] * inv_weights[1])
        else:
            self.weights = torch.Tensor([1, 1])
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:

        self.weights = self.weights.to(targets.device)
        wt = self.weights.gather(0, targets.long())
        if self.label_smoothing > 0:
            pos_smooth = 1.0 - self.label_smoothing
            neg_smooth = self.label_smoothing
            targets = targets * pos_smooth + (1 - targets) * neg_smooth
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = wt * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss