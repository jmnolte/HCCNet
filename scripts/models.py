from __future__ import annotations
from typing import cast
import torch
import torch.nn as nn
import os
from monai.networks.nets import milmodel
from torchvision.models import resnet50, densenet121, inception_v3, ResNet50_Weights, DenseNet121_Weights, Inception_V3_Weights

class MILNet(nn.Module):

    def __init__(
            self, 
            num_classes: int,
            mil_mode: str = 'att',
            pretrained: bool = True,
            backbone: str | None = None,
            trans_blocks: int = 4,
            trans_dropout: float = 0.0,
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

        if pretrained:
            weights = 'IMAGENET1K_V1'
        else:
            weights = None
        
        if backbone is None or backbone.lower() == 'resnet50':
            net = resnet50(weights=weights)
            nfc = net.fc.in_features
            net.fc = nn.Identity()
            if mil_mode == "att_trans_pyramid":
                net.layer1.register_forward_hook(self.forward_hook("layer1"))
                net.layer2.register_forward_hook(self.forward_hook("layer2"))
                net.layer3.register_forward_hook(self.forward_hook("layer3"))
                net.layer4.register_forward_hook(self.forward_hook("layer4"))

        elif backbone.lower() == 'densenet121':
            net = densenet121(weights=weights)
            nfc = net.classifier.in_features
            net.classifier = nn.Identity()
            if mil_mode == 'att_trans_pyramid':
                net.features._modules['denseblock1'].register_forward_hook(self.forward_hook('layer1'))
                net.features._modules['denseblock2'].register_forward_hook(self.forward_hook('layer2'))
                net.features._modules['denseblock3'].register_forward_hook(self.forward_hook('layer3'))
                net.features._modules['denseblock4'].register_forward_hook(self.forward_hook('layer4'))

        elif backbone.lower() == 'inceptionv3':
            net = inception_v3(weights=weights)
            nfc = net.fc.in_features
            net.fc = nn.Identity()
            if mil_mode == 'att_trans_pyramid':
                net.Mixed_5b.register_forward_hook(self.forward_hook("layer1"))
                net.Mixed_5c.register_forward_hook(self.forward_hook("layer2"))
                net.Mixed_5d.register_forward_hook(self.forward_hook("layer3"))
                net.Mixed_6a.register_forward_hook(self.forward_hook("layer4"))
                net.Mixed_6b.register_forward_hook(self.forward_hook("layer5"))
                net.Mixed_6c.register_forward_hook(self.forward_hook("layer6"))
                net.Mixed_6d.register_forward_hook(self.forward_hook("layer7"))
                net.Mixed_6e.register_forward_hook(self.forward_hook("layer8"))
                net.Mixed_7a.register_forward_hook(self.forward_hook("layer9"))
                net.Mixed_7b.register_forward_hook(self.forward_hook("layer10"))
                net.Mixed_7c.register_forward_hook(self.forward_hook("layer11"))
        else:
            raise ValueError('Unsupported backbone model selected. Please choose from: resnet50, dense121, inceptionv3. Defaults to resnet50.')
        
        if self.mil_mode in ["mean", "max"]:
            pass

        elif self.mil_mode == "att":
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        elif self.mil_mode == "att_trans":
            transformer = nn.TransformerEncoderLayer(d_model=nfc, nhead=8, dropout=trans_dropout)
            self.transformer = nn.TransformerEncoder(transformer, num_layers=trans_blocks)
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        elif self.mil_mode == "att_trans_pyramid":
            transformer_list = nn.ModuleList(
                [
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=trans_dropout), 
                        num_layers=trans_blocks
                    ),
                    nn.Sequential(
                        nn.Linear(768, 256),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=trans_dropout),
                            num_layers=trans_blocks,
                        ),
                    ),
                    nn.Sequential(
                        nn.Linear(1280, 256),
                        nn.TransformerEncoder(
                            nn.TransformerEncoderLayer(d_model=256, nhead=8, dropout=trans_dropout),
                            num_layers=trans_blocks,
                        ),
                    ),
                    nn.TransformerEncoder(
                        nn.TransformerEncoderLayer(d_model=nfc + 256, nhead=8, dropout=trans_dropout),
                        num_layers=trans_blocks,
                    ),
                ]
            )
            self.transformer = transformer_list
            nfc = nfc + 256
            self.attention = nn.Sequential(nn.Linear(nfc, 2048), nn.Tanh(), nn.Linear(2048, 1))

        else:
            raise ValueError("Unsupported mil_mode: " + str(mil_mode))

        self.myfc = nn.Linear(nfc, num_classes)
        self.net = net

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

        elif self.mil_mode == "max":
            x = self.myfc(x)
            x, _ = torch.max(x, dim=1)

        elif self.mil_mode == "att":
            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            x = self.myfc(x)

        elif self.mil_mode == "att_trans" and self.transformer is not None:
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)

            x = self.myfc(x)

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

        else:
            raise ValueError("Wrong model mode" + str(self.mil_mode))

        return x

    def forward(self, x: torch.Tensor, no_head: bool = False) -> torch.Tensor:
        sh = x.shape
        x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4])

        x = self.net(x)
        x = x.reshape(sh[0], sh[1], -1)

        if not no_head:
            x = self.calc_head(x)

        return x


    # def calc_head(self, x: torch.Tensor) -> torch.Tensor:
    #     return super().calc_head(x)
    
    # def forward(self, x: torch.Tensor, no_head: bool = False) -> torch.Tensor:
    #     return super().forward(x, no_head)