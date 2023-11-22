from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet as _ResNet
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Any, Type, Callable, Union, List, Optional

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth'
}

class ResNet(_ResNet):

    def __init__(
            self,
            block: Type[Union[BasicBlock, Bottleneck]],
            layers: List[int],
            num_channels: int = 3,
            num_classes: int = 1000,
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 64,
            replace_stride_with_dilation: Optional[List[bool]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
        ) -> None:
        super().__init__(
            block,
            layers,
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer
        )

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)


def _load_pretrained_weights(
        model: ResNet, 
        url: str, 
        num_channels: int = 3, 
        progress: bool = True
    ) -> None:

    state_dict = torch.hub.load_state_dict_from_url(url, progress=progress)
    if num_channels == 1:
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
    elif num_channels != 3:
        assert False, "Invalid number of channels for pretrained weights"
    model.load_state_dict(state_dict)


def _truncate_model(
        model: ResNet, 
        trunc: int, 
        num_channels: int = 3
    ) -> ResNet:

    model = nn.Sequential(*list(model.children())[:4+trunc])

    test_input = torch.zeros(size=(1, num_channels, 224, 224), dtype=torch.float)
    linear_input_dim = np.prod(model(test_input).shape[:2])

    linear = nn.Linear(in_features=linear_input_dim, out_features=1000, bias=True)
    nn.init.kaiming_normal_(linear.weight) 
    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    model.fc = linear
    return model


def _resnet(
        arch: str,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        pretrained: bool = True,
        truncate_layer: int | None = None,
        progress: bool = True,
        **kwargs: Any
    ) -> ResNet:

    net = ResNet(block, layers, **kwargs)
    if pretrained:
        _load_pretrained_weights(net, model_urls[arch], kwargs['num_channels'], progress)

    if truncate_layer is not None:
        net = _truncate_model(net, truncate_layer, kwargs['num_channels'])
    return net


def resnet10(
        pretrained: bool = True,
        truncate_layer: int | None = None,
        progress: bool = True,
        **kwargs: Any
    ) -> ResNet:
    
    return _resnet('resnet10', BasicBlock, [1, 1, 1, 1], pretrained=pretrained, truncate_layer=truncate_layer, progress=progress, **kwargs)


def resnet18(
        pretrained: bool = True,
        truncate_layer: int | None = None,
        progress: bool = True,
        **kwargs: Any
    ) -> ResNet:
    
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained=pretrained, truncate_layer=truncate_layer, progress=progress, **kwargs)


def resnet34(
        pretrained: bool = True,
        truncate_layer: int | None = None,
        progress: bool = True,
        **kwargs: Any
    ) -> nn.Module:
    
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained=pretrained, truncate_layer=truncate_layer, progress=progress, **kwargs)


def resnet50(
        pretrained: bool = True,
        truncate_layer: int | None = None,
        progress: bool = True,
        **kwargs: Any
    ) -> nn.Module:
    
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained=pretrained, truncate_layer=truncate_layer, progress=progress, **kwargs)

if __name__ == '__main__':
    net = resnet18(pretrained=True, num_channels=1, truncate_layer=2)
    print(net)