import torch
import torch.nn as nn
import torch.nn.functional as F

class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07, mask_fp: bool = True, epsilon: float = 1e-8) -> None:
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super().__init__()
        self.temperature = temperature
        self.mask_fp = mask_fp
        self.epsilon = epsilon
        
    @staticmethod
    def mask_uncertain_positives(proba: torch.Tensor, bag_labels: torch.Tensor, rho: int) -> torch.Tensor:

        pos_proba = proba[bag_labels == 1]
        if rho > (pos_proba.shape[0] // 2):
            rho = (pos_proba.shape[0] // 2)
        top_k = torch.topk(pos_proba, rho, largest=True).values
        bottom_k = torch.topk(pos_proba, rho, largest=False).values
        mask = torch.zeros_like(proba)
        mask[torch.isin(proba, top_k)] = 1
        mask[torch.isin(proba, bottom_k)] = 1
        return mask
        
    @staticmethod
    def mask_similar_class(labels: torch.Tensor) -> torch.Tensor:

        return labels.unsqueeze(1).repeat(1, labels.shape[0]) == labels
    
    @staticmethod
    def mask_anchor_sample(dot_product: torch.Tensor) -> torch.Tensor:
        
        return 1 - torch.eye(dot_product.shape[0])

    def forward(self, projections: torch.Tensor, probas: torch.Tensor, bag_labels: torch.Tensor, rho: int = 0, warm_up: bool = True) -> torch.Tensor:
        """

        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        batch_size = bag_labels.shape[0]
        num_inst = projections.shape[0] // batch_size
        bag_labels = bag_labels.unsqueeze(-1).repeat(1, num_inst).flatten()
        probas = probas.reshape(-1).to(projections.device)

        labels = (probas > 0.5).float()
        labels = labels * bag_labels if self.mask_fp else labels
        
        if warm_up and rho == 0:
            samples = (bag_labels == 0).float()
        elif warm_up and rho > 0:
            negatives = (bag_labels == 0).float()
            certains = self.mask_uncertain_positives(probas, bag_labels, rho)
            samples = negatives + certains
        else:
            samples = (bag_labels == 1).float()

        dp_temp = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        edp_temp = (
            torch.exp(dp_temp - torch.max(dp_temp, dim=1, keepdim=True)[0]) + self.epsilon
        )

        mask_cls = torch.mm(samples.unsqueeze(-1), samples.unsqueeze(-1).T)
        mask_sim = self.mask_similar_class(labels)
        mask_anc = self.mask_anchor_sample(edp_temp)
        mask = mask_sim.to(projections.device) * mask_anc.to(projections.device) * mask_cls.to(projections.device)
        cardinality = torch.sum(mask, dim=1)

        log_prob = -torch.log(edp_temp / (torch.sum(edp_temp * mask_anc.to(projections.device), dim=1, keepdim=True)))
        sample_loss = torch.sum(log_prob * mask, dim=1) / (cardinality + self.epsilon)

        return torch.sum(sample_loss / (torch.sum(samples) + self.epsilon))

class SupervisedContrastiveLoss2(nn.Module):

    def __init__(self, temperature: float = 0.07, epsilon: float = 1e-8):
        """
        Implementation of the loss described in the paper Supervised Contrastive Learning :
        https://arxiv.org/abs/2004.11362

        :param temperature: int
        """
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon

    def forward(self, projections, targets):
        """

        :param projections: torch.Tensor, shape [batch_size, projection_dim]
        :param targets: torch.Tensor, shape [batch_size]
        :return: torch.Tensor, scalar
        """
        dp_temp = torch.mm(projections, projections.T) / self.temperature
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        edp_temp = (
            torch.exp(dp_temp - torch.max(dp_temp, dim=1, keepdim=True)[0]) + self.epsilon
        )

        mask_sim_cls = (targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets).to(projections.device)
        mask_anc_out = (1 - torch.eye(edp_temp.shape[0])).to(projections.device)
        mask = mask_sim_cls * mask_anc_out
        cardinality = torch.sum(mask, dim=1)

        log_prob = -torch.log(edp_temp / (torch.sum(edp_temp * mask_anc_out, dim=1, keepdim=True)))
        sample_loss = torch.sum(log_prob * mask, dim=1) / (cardinality + self.epsilon)
        loss = torch.mean(sample_loss)

        return loss