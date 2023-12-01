import torch
import torch.nn as nn
import torch.nn.functional as F


class RecallLoss(nn.Module):
    def __init__(self, n_classes: int = 2, ignore_index: int = 99):
        super().__init__()
        self.n_classes = n_classes
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor): 
        # input (batch,n_classes,H,W)
        # target (batch,H,W)
        pred = logits.argmax(1)
        index = (pred != targets).view(-1) 
        
        #calculate ground truth counts
        gt_counter = torch.ones((self.n_classes,)).to(targets.device)
        gt_idx, gt_count = torch.unique(targets, return_counts=True)
        
        # map ignored label to an exisiting one
        gt_count[gt_idx==self.ignore_index] = gt_count[1]
        gt_idx[gt_idx==self.ignore_index] = 1 
        gt_counter[gt_idx] = gt_count.float()
        
        #calculate false negative counts
        fn_counter = torch.ones((self.n_classes)).to(targets.device)
        fn = targets.view(-1)[index]
        fn_idx, fn_count = torch.unique(fn, return_counts=True)
        
        # map ignored label to an exisiting one
        fn_count[fn_idx==self.ignore_index] = fn_count[1]
        fn_idx[fn_idx==self.ignore_index] = 1 
        fn_counter[fn_idx] = fn_count.float()

        weight = fn_counter / gt_counter
        
        ce_loss = F.cross_entropy(logits, targets, reduction='none', ignore_index=self.ignore_index)
        loss = weight[targets] * ce_loss
        return loss.mean()