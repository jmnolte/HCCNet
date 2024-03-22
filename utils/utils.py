import torch
import torch.nn as nn
import numpy as np

def cancel_gradients_last_layer(
        model: nn.Module, 
        step: int,
        warmup_steps: int
    ) -> None:

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

    warmup_schedule = np.array([])
    if warmup_steps > 0:
        warmup_schedule = np.linspace(final_value, base_value, warmup_steps)

    iters = np.arange(steps - warmup_steps)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == steps
    return schedule

def get_params_groups(model):
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

    alpha = {8: 0.0001, 16: 0.000141, 32: 0.0002, 64: 0.000282, 128: 0.0004, 256: 0.000565, 512: 0.0008}
    return alpha[batch_size] * np.sqrt(batch_size) / np.sqrt(128)

class MultiCropWrapper(nn.Module):
    """
    Perform forward pass separately on each resolution input.
    The inputs corresponding to a single resolution are clubbed and single
    forward is run on the same resolution inputs. Hence we do several
    forward passes = number of different resolutions used. We then
    concatenate all the output features and run the head forward on these
    concatenated features.
    """
    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # disable layers dedicated to ImageNet labels classification
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        # convert to list
        if not isinstance(x, list):
            x = [x]
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),
            return_counts=True,
        )[1], 0)
        start_idx, output = 0, torch.empty(0).to(x[0].device)
        for end_idx in idx_crops:
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))
            # The output is a tuple with XCiT model. See:
            # https://github.com/facebookresearch/xcit/blob/master/xcit.py#L404-L405
            if isinstance(_out, tuple):
                _out = _out[0]
            # accumulate outputs
            output = torch.cat((output, _out))
            start_idx = end_idx
        # Run the head forward on the concatenated features.
        return self.head(output)



