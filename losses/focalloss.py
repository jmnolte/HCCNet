import torch.nn.functional as F
import torch.nn as nn
import torch


class FocalLoss(nn.Module):
    "Focal loss implemented using F.cross_entropy"
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None, reduction: str = 'mean') -> None:
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction


    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        if self.weight is not None:
            self.weight = self.weight.to(targets.device)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.weight, reduction="none")
        p_t = torch.exp(-ce_loss)
        loss = (1 - p_t)**self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
