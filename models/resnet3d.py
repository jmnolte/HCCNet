from __future__ import annotations

import torch
import torch.nn as nn
import os

from typing import Any
from monai.networks.nets import ResNet as ResNet3D
from monai.networks.nets import ResNetBlock, ResNetBottleneck


def get_inplanes():
    return [64, 128, 256, 512]
    

def _load_pretrained_weights(
        model: ResNet3D,
        weights_path: str,
        num_channels: int = 1
    ) -> None:

    weights_dict = torch.load(weights_path, map_location='cpu')
    weights_dict = {k.replace('module.', ''): v for k, v in weights_dict['state_dict'].items()}
    model_dict = model.state_dict()

    if num_channels > 1:
        weights_dict['conv1.weight'] = weights_dict['conv1.weight'].repeat(1, num_channels, 1, 1, 1)
        
    model_dict.update(weights_dict)
    model.load_state_dict(model_dict)


def _resnet(
        arch: str,
        block: type[ResNetBlock | ResNetBottleneck],
        layers: list[int],
        block_inplanes: list[int],
        pretrained: bool,
        progress: bool = True,
        **kwargs: Any
    ) -> ResNet3D:

    model = ResNet3D(block, layers, block_inplanes, **kwargs)
    if pretrained:
        weights_path = os.path.join('/home/x3007104/thesis/pretrained_models', 'resnet_' + arch[-2:] + '_23dataset.pth')
        _load_pretrained_weights(model, weights_path, num_channels=kwargs['n_input_channels'])
        if progress:
            print("Loaded pretrained weights from: " + str(weights_path))
    return model


def resnet10(
        pretrained: bool = False, 
        progress: bool = True, 
        **kwargs: Any
    ) -> ResNet3D:

    return _resnet("resnet10", ResNetBlock, [1, 1, 1, 1], get_inplanes(), pretrained, progress, **kwargs)


def resnet18(
        pretrained: bool = False, 
        progress: bool = True, 
        **kwargs: Any
    ) -> ResNet3D:
    
    return _resnet("resnet18", ResNetBlock, [2, 2, 2, 2], get_inplanes(), pretrained, progress, **kwargs)


def resnet34(
        pretrained: bool = False, 
        progress: bool = True, 
        **kwargs: Any
    ) -> ResNet3D:
    
    return _resnet("resnet34", ResNetBlock, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


def resnet50(
        pretrained: bool = False, 
        progress: bool = True, 
        **kwargs: Any
    ) -> ResNet3D:
    
    return _resnet("resnet50", ResNetBottleneck, [3, 4, 6, 3], get_inplanes(), pretrained, progress, **kwargs)


if __name__ == '__main__':
    net = resnet10(pretrained=True, n_input_channels=4, num_classes=2)
    print(net)