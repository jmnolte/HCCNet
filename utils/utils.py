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

def prep_batch(
        data: dict, 
        batch_size: int, 
        pretrain: bool = False
    ) -> tuple:

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
    data['lirads'] = torch.where(data['delta'] == 0.0, 0.0, data['lirads'] + 1)
    pad_idx = torch.argmax(padding_mask, dim=1)
    pad_idx = torch.where(pad_idx == 0, seq_length, pad_idx)
    if pretrain:
        data['label'] = torch.zeros(batch_size)
        for i in range(batch_size):
            rand_gen = torch.rand(1)
            if rand_gen < 0.66:
                shuffled_idx = torch.randperm(pad_idx[i])
                sorted_idx = torch.sort(shuffled_idx).values
                data['image'][i, :pad_idx[i]] = data['image'][i, shuffled_idx]
                data['label'][i] = 0 if shuffled_idx.equal(sorted_idx) else 1
    else:
        data['label'] = torch.max(data['label'], dim=1).values
    pt_info = [data['delta'], data['lirads']]
    return data['image'], data['label'], pt_info, padding_mask

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



