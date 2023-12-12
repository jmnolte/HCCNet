import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveCenterLoss(nn.Module):

    def __init__(self, feat_dim, num_classes, lambda_c=1.0):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, feats, labels):
        batch_size = feats.size()[0]
        expanded_centers = self.centers.expand(batch_size, -1, -1).to(feats.device)
        expanded_feat = feats.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_feat - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, labels.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / (inter_distances + epsilon) / 0.1

        return loss


class ContrastiveCenterBCELoss(ContrastiveCenterLoss):

    def __init__(self, feat_dim, num_classes, lambda_c=1.0, pos_weight=3.0):
        super().__init__(
            feat_dim=feat_dim,
            num_classes=num_classes,
            lambda_c=lambda_c,
        )
        self.pos_weight = torch.Tensor([pos_weight])

    def forward(self, logits, feats, labels):
        cc_loss = super().forward(feats, labels)
        bce_loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float(), pos_weight=self.pos_weight)
        return bce_loss + cc_loss