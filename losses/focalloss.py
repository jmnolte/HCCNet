import torch.nn.functional as F
import torch.nn as nn
import torch

class FocalLoss(nn.Module):

    def __init__(self, gamma: float = 2.0, alpha: float | list | None = None, reduction: str = 'mean') -> None:

        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            alpha = torch.Tensor([1 - alpha, alpha]) if isinstance(alpha, float) else torch.Tensor(alpha)
            inv_alpha = alpha ** -1
            self.alpha = inv_alpha / (alpha[0] * inv_alpha[0] + alpha[1] * inv_alpha[1])
        else:
            self.alpha = torch.Tensor([1, 1])
        self.reduction = reduction


    def forward(self, logits: torch.Tensor, targets: torch.Tensor):

        self.alpha = self.alpha.to(targets.device)
        at = self.alpha.gather(0, targets.long())
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        loss = at * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
