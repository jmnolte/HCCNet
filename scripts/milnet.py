from __future__ import annotations
from typing import cast
import torch
import torch.nn as nn
from resnet import resnet50

class MILNet(nn.Module):

    def __init__(
            self, 
            num_classes: int,
            mil_mode: str = 'att',
            backbone: str | None = None,
            pretrained: bool = True,
            tl_strategy: str = 'finetune',
            shrink_coefficient: int = 1,
            load_up_to: int = -1,
            trans_blocks: int = 4,
            trans_dropout: float = 0.0
            ) -> None:
        super().__init__()
        '''
        Define the model's version and set the number of input channels.

        Args:
            num_classes (int): Integer specifying the number of classes.
            mil_mode (str): Mutiple instance learning (MIL) algorithm. Can be one of the following (defaults to 'att'):
                'mean': takes the average over all instances (decision boundary = 0.5).
                'max': retains the maximum value over all instances.
                'att': attention based MIL https://arxiv.org/abs/1802.04712.
                'att_trans': transformer MIL https://arxiv.org/abs/2111.01556.
                'att_trans_pyramid': transformer pyramid MIL https://arxiv.org/abs/2111.01556.
            backbone (str): backbone (str): Backbone architecture to use. Has to be 'resnet50', or 'densenet121'.
            pretrained (bool): Flag to initialize model using pretrained weights.
            tl_strategy (str): String specifying the transfer learning strategy. Can be one of the following (defaults to 'finetune'):
                'finetune': finetunes the model, updating all model weights.
                'lw_finetune': finetunes a subset of the model, freezing the reamining layer until a given cutoff point k.
                'transfusion': reduces the dimensionality of k top layers, initializing their weights randomly, and updating all model weights.
            shrink_coefficient (int): Integer specifying the factor by which the dimensionality of the model should be reduced in the top layers. Only applies if 'transfusion' is selcted as the model's transfer learning strategy.
            load_up_to (int): Integer specifying the cutoff point k, when 'lw_finetune' or 'transfusion' is used as the model's transfer learning strategy.
        '''
        if mil_mode.lower() not in ['mean', 'max', 'att', 'att_trans', 'att_trans_pyramid']:
            raise ValueError('Unsupported mil_mode selected. Please choose from: mean, max, att, att_trans, att_trans_pyramid')
        if tl_strategy.lower() not in ['finetune', 'lw_finetune', 'transfusion']:
            raise ValueError('Unsupported transfer learning strategy selected. Please choose from: finetuning (finetune), layer-wise finetuning (lw_finetune), or TransFusion (transfusion). Defaults to finetuning.')
        
        self.extra_outputs: dict[str, torch.Tensor] = {}
        self.mil_mode = mil_mode.lower()
        self.attention = nn.Sequential()
        self.transformer: nn.Module | None = None
        
        if backbone is None or backbone.lower() == 'resnet50':
            net = resnet50(pretrained=pretrained, shrink_coefficient=shrink_coefficient, load_up_to=load_up_to)
            nfc = net.fc.in_features
            net.fc = nn.Identity()
            if mil_mode == "att_trans_pyramid":
                net.layer1.register_forward_hook(self.forward_hook("layer1"))
                net.layer2.register_forward_hook(self.forward_hook("layer2"))
                net.layer3.register_forward_hook(self.forward_hook("layer3"))
                net.layer4.register_forward_hook(self.forward_hook("layer4"))

        # elif backbone.lower() == 'densenet121':
        #     net = models.densenet121(weights=weights)
        #     nfc = net.classifier.in_features
        #     net.classifier = nn.Identity()
        #     if mil_mode == 'att_trans_pyramid':
        #         net.features._modules['denseblock1'].register_forward_hook(self.forward_hook("layer1"))
        #         net.features._modules['denseblock2'].register_forward_hook(self.forward_hook("layer2"))
        #         net.features._modules['denseblock3'].register_forward_hook(self.forward_hook("layer3"))
        #         net.features._modules['denseblock4'].register_forward_hook(self.forward_hook("layer4"))
        else:
            raise ValueError('Unsupported backbone model selected. Please choose from: resnet50, or densenet121. Defaults to resnet50.')
        
        if tl_strategy == 'lw_finetune':

            frozen_param = ['conv1.weight', 'bn1.weight', 'bn1.bias']
            for name, param in net.named_parameters():
                param.requires_grad = False if name in frozen_param else True
            param_count = 0
            for layer in net.children():
                for bottleneck in layer.children():
                    param_count += 1
                    for name, param in bottleneck.named_parameters():
                        if param_count <= load_up_to:
                            param.requires_grad = False

        if self.mil_mode in ["mean", "max"]:
            pass

        elif self.mil_mode == "att":
            self.attention = nn.Sequential(nn.Linear(nfc, nfc), nn.Tanh(), nn.Linear(nfc, 1))

        elif self.mil_mode == "att_trans":
            transformer = nn.TransformerEncoderLayer(d_model=nfc, nhead=8, dropout=trans_dropout)
            self.transformer = nn.TransformerEncoder(transformer, num_layers=trans_blocks)
            self.attention = nn.Sequential(nn.Linear(nfc, nfc), nn.Tanh(), nn.Linear(nfc, 1))

        elif self.mil_mode == "att_trans_pyramid":
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

if __name__ == '__main__':
    milnet = MILNet(num_classes=2, mil_mode='att_trans_pyramid', pretrained=True, backbone='resnet50', tl_strategy='TF', shrink_coefficient=2, load_up_to=7)
    print(milnet)