from __future__ import annotations

from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F
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

        fc_classes = num_classes if num_classes > 2 else 1
        self.myfc = nn.Linear(nfc, fc_classes)
        self.prototype = nn.Parameter(torch.randn(num_classes, nfc))
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
            x = torch.mean(x, dim=1)
            return x

        elif self.mil_mode == "max":
            x = self.myfc(x)
            x, _ = torch.max(x, dim=1)
            return x

        elif self.mil_mode == "att":
            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            x = self.myfc(x)
            return x, a

        elif self.mil_mode == "att_trans" and self.transformer is not None:
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            x = self.myfc(x)
            return x, a 

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
            x = self.myfc(x)
            return x, a

        else:
            raise ValueError("Wrong model mode" + str(self.mil_mode))

    def calc_euclidean(self, feats: torch.Tensor, att: torch.Tensor) -> torch.Tensor:
        
        sh = feats.shape
        prototype = self.prototype.expand(sh[0] * sh[1], -1, -1).to(feats.device)
        feats = feats.reshape(sh[0] * sh[1], -1)
        att = att.reshape(sh[0] * sh[1], -1)

        euc = torch.cdist(feats, prototype, p=2)
        euc_log = F.log_softmax(euc[[0]].squeeze(0), dim=1)

        att = torch.cat((1 - att, att), dim=1)
        att_log = F.log_softmax(att, dim=1)
        return euc_log, att_log

    def forward(self, x: torch.Tensor, no_head: bool = False, no_euclidean: bool = False) -> torch.Tensor:
        sh = x.shape
        if len(sh) == 5:
            x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])
        elif len(sh) == 6:
            x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4], sh[5])

        feats = self.net(x)
        feats = feats.reshape(sh[0], sh[1], -1)

        if not no_head:
            out, att = self.calc_head(feats)

        if not no_euclidean:
            euc_log, att_log = self.calc_euclidean(feats, att)

        return out, euc_log, att_log
