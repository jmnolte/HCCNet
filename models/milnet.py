from __future__ import annotations

from typing import cast
import torch
import torch.nn as nn
import numpy as np
from models.resnet import resnet18, resnet34, resnet50
from models.resnet3d import resnet10 as resnet10_3d
from models.resnet3d import resnet18 as resnet18_3d
from models.resnet3d import resnet34 as resnet34_3d
from models.resnet3d import resnet50 as resnet50_3d
from models.swinvit import swinvit_base

class MILNet(nn.Module):

    def __init__(
            self, 
            backbone: str = 'resnet50',
            mil_mode: str = 'att',
            pretrained: bool = True,
            num_channels: int = 3,
            num_spatial_dims: int = 2,
            num_classes: int = 2,
            truncate_layer: int | None = None,
            trans_blocks: int = 4,
            trans_dropout: float = 0.0
            ) -> None:
        super().__init__()
        '''
        Define the model's version and set the number of input channels.

        Args:
            version (str): ResNet model version. Can be 'resnet10', 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', or 'resnet200'.
            num_out_classes (int): Number of output classes.
            num_in_channels (int): Number of input channels.
            pretrained (bool): If True, pretrained weights are used.
            feature_extraction (bool): If True, only the last layer is updated during training. If False,
                all layers are updated.
            weights_path (str): Path to the pretrained weights.
        '''
        if mil_mode.lower() not in ['mean', 'max', 'att', 'att_trans', 'att_trans_pyramid']:
            raise ValueError('Unsupported mil_mode selected. Please choose from: mean, max, att, att_trans, att_trans_pyramid')
        
        self.extra_outputs: dict[str, torch.Tensor] = {}
        self.mil_mode = mil_mode.lower()
        self.attention = nn.Sequential()
        self.transformer: nn.Module | None = None

        if num_spatial_dims == 2:
            if backbone.lower() == 'resnet18':
                net = resnet18(pretrained=pretrained, num_channels=num_channels, truncate_layer=truncate_layer)
            elif backbone.lower() == 'resnet34':
                net = resnet34(pretrained=pretrained, num_channels=num_channels, truncate_layer=truncate_layer)
            elif backbone.lower() == 'resnet50':
                net = resnet50(pretrained=pretrained, num_channels=num_channels, truncate_layer=truncate_layer)
            else:
                raise ValueError('Unsupported backbone model selected.')
        elif num_spatial_dims == 3:
            if backbone.lower() == 'resnet10':
                net = resnet10_3d(pretrained=pretrained, n_input_channels=num_channels, shortcut_type='B')
            elif backbone.lower() == 'resnet18':
                net = resnet18_3d(pretrained=pretrained, n_input_channels=num_channels, shortcut_type='A')
            elif backbone.lower() == 'resnet34':
                net = resnet34_3d(pretrained=pretrained, n_input_channels=num_channels, shortcut_type='A')
            elif backbone.lower() == 'resnet50':
                net = resnet50_3d(pretrained=pretrained, n_input_channels=num_channels, shortcut_type='B')
            elif backbone.lower() == 'swinvit':
                net = swinvit_base(pretrained=pretrained, in_chans=num_channels, num_classes=num_classes)
            else:
                raise ValueError('Unsupported backbone model selected.')
        else:
            raise ValueError('Unsupported number of spatial dimensions selected. Please choose from: 2, 3')

        nfc = net.fc.in_features if backbone.lower() != 'swinvit' else net.head.in_features
        if backbone.lower() != 'swinvit':
            net.fc = nn.Identity()
        else:
            net.head = nn.Identity()

        if self.mil_mode in ["mean", "max"]:
            pass

        elif self.mil_mode == "att":
            self.attention = nn.Sequential(nn.Linear(nfc, nfc), nn.Tanh(), nn.Linear(nfc, 1))

        elif self.mil_mode == "att_trans":
            transformer = nn.TransformerEncoderLayer(d_model=nfc, nhead=8, dropout=trans_dropout)
            self.transformer = nn.TransformerEncoder(transformer, num_layers=trans_blocks)
            self.attention = nn.Sequential(nn.Linear(nfc, nfc), nn.Tanh(), nn.Linear(nfc, 1))

        elif self.mil_mode == "att_trans_pyramid":
            net.layer1.register_forward_hook(self.forward_hook("layer1"))
            net.layer2.register_forward_hook(self.forward_hook("layer2"))
            net.layer3.register_forward_hook(self.forward_hook("layer3"))
            net.layer4.register_forward_hook(self.forward_hook("layer4"))

            transformer_list = nn.ModuleList(
                [
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=nfc // 8, nhead=8, dropout=trans_dropout), 
                        num_layers=trans_blocks
                    ),
                    nn.Sequential(
                        nn.Linear(nfc // 4 + nfc // 8, nfc // 8),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=nfc // 8, nhead=8, dropout=trans_dropout),
                            num_layers=trans_blocks,
                        ),
                    ),
                    nn.Sequential(
                        nn.Linear(nfc // 2 + nfc // 8, nfc // 8),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=nfc // 8, nhead=8, dropout=trans_dropout),
                            num_layers=trans_blocks,
                        ),
                    ),
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=nfc + nfc // 8, nhead=8, dropout=trans_dropout),
                        num_layers=trans_blocks,
                    ),
                ]
            )
            self.transformer = transformer_list
            self.attention = nn.Sequential(nn.Linear(nfc + nfc // 8, nfc), nn.Tanh(), nn.Linear(nfc, 1))
            nfc = nfc + nfc // 8

        else:
            raise ValueError("Unsupported mil_mode: " + str(mil_mode))

        self.myfc = nn.Linear(nfc, num_classes)
        self.net = net
        self.nfc = nfc
            
    def forward_hook(self, layer_name):
        @staticmethod
        def hook(module, input, output):
            self.extra_outputs[layer_name] = output
        return hook

    def calc_head(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape

        if self.mil_mode == "mean":
            x = self.myfc(x)
            out = torch.mean(x, dim=1)

        elif self.mil_mode == "max":
            x = self.myfc(x)
            out, _ = torch.max(x, dim=1)

        elif self.mil_mode == "att":
            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            out = self.myfc(x)

        elif self.mil_mode == "att_trans" and self.transformer is not None:
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            out = self.myfc(x)

        elif self.mil_mode == "att_trans_pyramid" and self.transformer is not None:
            l1 = torch.mean(self.extra_outputs["layer1"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l2 = torch.mean(self.extra_outputs["layer2"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l3 = torch.mean(self.extra_outputs["layer3"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)
            l4 = torch.mean(self.extra_outputs["layer4"], dim=(2, 3)).reshape(sh[0], sh[1], -1).permute(1, 0, 2)

            transformer_list = cast(nn.ModuleList, self.transformer)

            x = transformer_list[0](l1)
            x = transformer_list[1](torch.cat((x, l2), dim=2))
            x = transformer_list[2](torch.cat((x, l3), dim=2))
            x = transformer_list[3](torch.cat((x, l4), dim=2))

            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            out = self.myfc(x)

        else:
            raise ValueError("Wrong model mode" + str(self.mil_mode))

        return x, out

    def forward(self, x: torch.Tensor, no_head: bool = False) -> torch.Tensor:
        sh = x.shape
        if len(sh) == 5:
            x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
        elif len(sh) == 6:
            x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4], sh[5])

        x = self.net(x)
        x = x.reshape(sh[0], sh[1], -1)

        if not no_head:
            feats, logits = self.calc_head(x)

        return feats, logits
