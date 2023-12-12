import torch
import torch.nn as nn
import torch.nn.functional as F


class RecallLoss(nn.Module):
    def __init__(self, ignore_index: int = 99):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor): 

        if logits.shape[1] == 1:
            sigmoid = nn.Sigmoid()
            preds = sigmoid(logits) > 0.5
            if labels.dim() == 1:
                labels = labels.unsqueeze(1)
        elif logits.shape[1] > 1:
            preds = logits.argmax(1)

        preds = preds > 0.5
        neg_idx, pos_idx = (labels == 0), (labels == 1)
        neg_count, pos_count = neg_idx.sum(), pos_idx.sum()
        fn_count = (preds != labels)[pos_idx].sum()
        fp_count = (preds != labels)[neg_idx].sum()
        fpr = fp_count / neg_count
        fnr = fn_count / pos_count
        weights = labels.float().clone()
        weights[neg_idx] = fpr.item()
        weights[pos_idx] = fnr.item()

        if logits.shape[1] == 1:
            ce_loss = F.binary_cross_entropy_with_logits(logits, labels.float(), reduction='none')
            recall_ce = weights.squeeze(1) * ce_loss.squeeze(1)
            return recall_ce.mean()

        elif logits.shape[1] > 1:
            ce_loss = F.cross_entropy(logits, labels, reduction='none')
            recall_ce = weights * ce_loss
            return recall_ce.mean()