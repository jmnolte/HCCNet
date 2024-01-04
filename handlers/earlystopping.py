import torch.distributed as dist

class EarlyStopping:

    def __init__(
        self,
        patience: int = 10,
        warm_up: int = 20
    ) -> None:

    if patience < 1:
        raise ValueError("Patience argument should be a positive integer.")
    
    if warm_up < 1:
        raise ValueError("Warm-up argument should be a positive integer.")

    self.state_dict = {x: [] for x in ['counter','epoch','best_loss','best_score']}
    self.patience = patience
    self.warm_up = warm_up
    self.stop_criterion = torch.zeros(1)

    def __call__(
        self,
        epoch: int,
        loss: float,
        score: float
    ) -> dict:

    if epoch >= self.warm_up:
        if loss < self.state_dict['best_loss']:
            self.state_dict['counter'] = 0
            self.state_dict['epoch'] = epoch
            self.state_dict['best_loss'] = loss
            self.state_dict['best_score'] = score
            print(f'New best validation loss: {loss:.4f}. Current metric: {score:.4f}')
        else:
            self.state_dict['counter'] += 1

    if self.state_dict['counter'] >= self.patience:
        self.stop_criterion += 1

    dist.all_reduce(self.stop_criterion, op=dist.ReduceOp.AVG)
    if self.stop_criterion == 1:
        break

    return state_dict



