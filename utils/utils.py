import torch
import torch.nn as nn
import numpy as np

def cancel_gradients_last_layer(
        model: nn.Module, 
        step: int,
        warmup_steps: int
    ) -> None:

    '''
    Args:
        model (nn.Module): Pytorch module object.
        step (int): Current training step.
        warmup_step (int): Number of steps to wait before updating the model's last layer.
    '''

    if step >= warmup_steps:
        return
    for n, p in model.named_parameters():
        if "last_layer" in n:
            p.grad = None

def cosine_scheduler(
        base_value: float, 
        final_value: float, 
        steps: int, 
        warmup_steps: int = 0
    ) -> np.array:

    '''
    Args:
        base_value (float): Base value.
        final_value (float): Final value after cosine schedule.
        steps (int): Number of training steps.
        warmup_step (int): Number of steps to linearly increase the value to its base value. Defaults to 0.
    '''

    warmup_schedule = np.array([])
    if warmup_steps > 0:
        warmup_schedule = np.linspace(final_value, base_value, warmup_steps)

    iters = np.arange(steps - warmup_steps)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == steps
    return schedule

def get_params_groups(
        model: nn.Module
    ):

    '''
    Args:
        model (nn.Module): Pytorch module object.
    '''

    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if name.endswith(".bias") or len(param.shape) == 1:
            not_regularized.append(param)
        else:
            regularized.append(param)
    return [{'params': regularized}, {'params': not_regularized, 'weight_decay': 0.}]

def scale_learning_rate(
        batch_size: int
    ) -> float:

    '''
    Args:
        batch_size (int): Number of unique observations in a given batch.
    '''

    alpha = {8: 0.0001, 16: 0.000141, 32: 0.0002, 64: 0.000282, 128: 0.0004, 256: 0.000565, 512: 0.0008}
    return alpha[batch_size] * np.sqrt(batch_size) / np.sqrt(128)

def prep_batch(
        data: dict, 
        batch_size: int, 
        device: torch.device,
        pretrain: bool = False
    ) -> tuple:

    '''
    Args:
        data (dict): Batch obtained from a pytorch dataloader.
        batch_size (int): Number of unique observations in a given batch.
        device (torch.device): Pytorch device.
        pretrain (bool): Boolean flag to indicate the pretraining stage.
    '''

    B, C, H, W, D = data['image'].shape
    seq_length = B // batch_size
    for key in data:
        if key == 'image':
            data[key] = data[key].reshape(batch_size, seq_length, C, H, W, D)
        elif not isinstance(data[key], list):
            try:
                data[key] = data[key].reshape(batch_size, seq_length)
            except:
                pass
    padding_mask = torch.where(data['delta'] == 0.0, 1.0, 0.0)
    if not pretrain:
        data['label'] = torch.max(data['label'], dim=1).values
    return data['image'].to(device), data['label'].to(device), data['delta'].to(device), padding_mask.to(device)



