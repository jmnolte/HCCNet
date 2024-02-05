from __future__ import annotations

from typing import cast
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
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

        self.bag_head = nn.Linear(nfc, num_classes if num_classes > 2 else 1)
        # self.projection = nn.Sequential(nn.Linear(nfc, nfc), nn.ReLU(), nn.Linear(nfc, nfc // 8))
        self.ins_head = nn.Linear(nfc, num_classes if num_classes > 2 else 1)
        self.register_buffer('prototypes', torch.zeros(num_classes, nfc))
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
            x = self.bag_head(x)
            x = torch.mean(x, dim=1)

        elif self.mil_mode == "max":
            x = self.bag_head(x)
            x, _ = torch.max(x, dim=1)

        elif self.mil_mode == "att":
            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            x = self.bag_head(x)

        elif self.mil_mode == "att_trans" and self.transformer is not None:
            x = x.permute(1, 0, 2)
            x = self.transformer(x)
            x = x.permute(1, 0, 2)

            a = self.attention(x)
            a = torch.softmax(a, dim=1)
            x = torch.sum(x * a, dim=1)
            x = self.bag_head(x)

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
            x = self.bag_head(x)

        else:
            raise ValueError("Wrong model mode" + str(self.mil_mode))
        
        return x

    def update_prototypes(self, x: torch.Tensor, logits: torch.Tensor, curr_labels: torch.Tensor, warm_up: bool, alpha: float = 0.95) -> None:

        bag_labels = (curr_labels.reshape(-1, 1) > 0.0).float()
        bag_labels = torch.cat([1 - bag_labels, bag_labels], dim=1)
        probs = F.sigmoid(logits.reshape(-1, 1))
        if warm_up:
            probs = torch.cat([1 - probs, probs], dim=1) * bag_labels
        else:
            probs = torch.cat([1 - probs, probs * bag_labels[:,1].unsqueeze(-1)], dim=1)
        topk = torch.sum(bag_labels, dim=0) / (bag_labels.shape[0] / 8)
        pos_idx = torch.topk(probs[:,1], int(topk[1])).indices
        neg_idx = torch.topk(probs[:,0], int(topk[0])).indices
        for feat in x[pos_idx]:
            self.prototypes[1] = alpha * self.prototypes[1] + (1 - alpha) * feat
        for feat in x[neg_idx]:
            self.prototypes[0] = alpha * self.prototypes[0] + (1 - alpha) * feat

    def update_instance_labels(self, x: torch.Tensor, curr_labels: torch.Tensor, warm_up: bool) -> torch.Tensor:

        beta = 1.0 if warm_up else 0.99
        bag_labels = (curr_labels.reshape(-1) > 0.0).float()
        logits = torch.mm(x, self.prototypes.T)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        ins_labels = beta * curr_labels.reshape(-1) + (1 - beta) * (preds * bag_labels)
        return ins_labels.detach().clone()

    def forward(self, x: torch.Tensor, ins_labels: torch.Tensor, warm_up: bool, no_update: bool = False, alpha: float = 0.95) -> torch.Tensor:

        # extract features
        sh = x.shape
        x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4], sh[5])
        x = self.net(x)
        x = x.reshape(sh[0], sh[1], -1)

        # compute instance logits
        ins_logits = self.ins_head(x)

        # update prototypes and instance labels
        x = x.reshape(sh[0] * sh[1], -1)
        if not no_update:
            self.update_prototypes(x, ins_logits, ins_labels, warm_up, alpha)
        new_ins_labels = self.update_instance_labels(x, ins_labels, warm_up)
        x = x.reshape(sh[0], sh[1], -1)

        # aggregate instance features into bag features
        bag_logits = self.calc_head(x)
        return ins_logits, bag_logits, new_ins_labels

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class DecoderBlock(nn.Module):

    def __init__(self, d_model: int, n_heads:int, dropout: float = 0.1) -> None:

        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, src: torch.Tensor | None = None) -> torch.Tensor:

        if src is None:
            q = k = v = x
        else:
            q = x
            k = v = src
        a = self.attention(q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1))[0].transpose(0, 1)
        x = x + self.dropout(a)
        x = self.norm(x)
        return x
    

class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.1, activation: str = "relu") -> None:

        super().__init__()
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "gelu":
            act_fn = nn.GELU()
        self.feed_forward = nn.Sequential(nn.Linear(d_model, d_ffn), act_fn, nn.Dropout(dropout), nn.Linear(d_ffn, d_model))
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.dropout(self.feed_forward(x))
        x = self.norm(x)
        return x
    

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_heads=8,
    ):
        super().__init__()
        self.self_attn = DecoderBlock(d_model, n_heads, dropout=dropout)
        self.cross_attn = DecoderBlock(d_model, n_heads, dropout=dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ffn, dropout=dropout, activation=activation)

    def forward(self, tgt, src):

        tgt = self.self_attn(tgt)
        tgt = self.cross_attn(tgt, src)
        tgt = self.feed_forward(tgt)
        return tgt


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers):

        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)

    def forward(self, tgt, src):
        output = tgt
        for _, layer in enumerate(self.layers):
            output = layer(output, src)

        return output
    

class LinearBlock(nn.Module):

    def __init__(self, input_dim, dim, dropout) -> None:

        super().__init__()
        self.linear = nn.Linear(input_dim, dim)
        self.activation = nn.Tanh()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x
    

class TransformerBlock(nn.Module):

    def __init__(self, input_dim, dim, dropout) -> None:

        super().__init__()
        self.linear_block1 = LinearBlock(input_dim, dim * 2, dropout)
        self.linear_block2 = LinearBlock(dim * 2, dim * 4, dropout)
        self.linear_block3 = LinearBlock(dim * 4, dim * 2, dropout)
        self.linear_block4 = LinearBlock(dim * 2, dim * 2, dropout)
        self.linear_block5 = LinearBlock(dim * 2, dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear_block1(x)
        x = self.linear_block2(x)
        x = self.linear_block3(x)
        x = self.linear_block4(x)
        x = self.linear_block5(x)
        return x
    

class SkipConnection(nn.Module):

    def __init__(self, input_dim, dim) -> None:

        super().__init__()
        self.linear = nn.Linear(input_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear(x)
        return x

    
class MLPHead(nn.Module):

    def __init__(self, input_dim, num_classes) -> None:

        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.norm(x)
        x = self.linear(x)
        return x
    

class EncoderLayer(nn.Module):

    def __init__(self, input_dim: int, dim: int, skip: bool, init_layer: bool, dropout: float = 0.1) -> None:

        super().__init__()
        self.skip = skip
        trans_dim = input_dim if init_layer else dim
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=2, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.transformer_block = TransformerBlock(trans_dim, dim, dropout)
        self.residual_block = SkipConnection(input_dim, dim) if skip else nn.Identity()

    def forward(self, x: torch.Tensor, x_orig: torch.Tensor) -> torch.Tensor:

        x = self.transformer_block(x)
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        if self.skip:
            x = x + torch.relu(self.residual_block(x_orig))
        return x
    

class FeatureEncoder(nn.Module):

    def __init__(
        self,
        patch_dim,
        dim,
        depth,
        dropout=0.1
    ):
        super().__init__()
        for i_layer in range(depth):
            layer = EncoderLayer(
                input_dim=patch_dim, 
                dim=dim, 
                skip=False if i_layer == 3 else True, 
                init_layer=True if i_layer == 0 else False,
                dropout=dropout)
            if i_layer == 0:
                self.layer1 = layer
            elif i_layer == 1:
                self.layer2 = layer
            elif i_layer == 2:
                self.layer3 = layer
            elif i_layer == 3:
                self.layer4 = layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x_orig = x.clone()
        x = self.layer1(x, x_orig)
        x = self.layer2(x, x_orig)
        x = self.layer3(x, x_orig)
        x = self.layer4(x, x_orig)
        return x


# Decoder
class BagDecoder(nn.Module):

    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_decoder_layers=6,
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            d_model,
            dim_feedforward,
            dropout,
            activation,
            nhead,
        )
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers)

    def forward(self, tgt, src):
        x = self.decoder(tgt, src)
        return x
    

class IIBMIL(nn.Module):
    def __init__(
        self,
        num_classes,
        patch_dim,
        dim,
        depth,
        num_queries=5,
    ):
        super().__init__()
        backbone = resnet50_3d(pretrained=True, n_input_channels=1)
        nfc = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.encoder = FeatureEncoder(
            patch_dim=nfc,
            dim=dim,
            depth=depth
        )
        self.decoder = BagDecoder(
            d_model=dim,
            nhead=4,
            num_decoder_layers=2,
            dim_feedforward=dim * 2,
            dropout=0.1,
            activation="relu",
        )
        self.query_embed = nn.Embedding(num_queries, dim)
        self.bag_head = nn.Linear(dim * num_queries, num_classes if num_classes > 2 else 1)
        self.ins_head = MLPHead(dim, num_classes if num_classes > 2 else 1)
        self.register_buffer('prototypes', torch.zeros(num_classes, dim))

    def update_prototypes(self, x: torch.Tensor, logits: torch.Tensor, curr_labels: torch.Tensor, warm_up: bool, alpha: float = 0.95) -> None:

        bag_labels = (curr_labels.reshape(-1, 1) > 0.0).float()
        bag_labels = torch.cat([1 - bag_labels, bag_labels], dim=1)
        probs = F.sigmoid(logits.reshape(-1, 1))
        if warm_up:
            probs = torch.cat([1 - probs, probs], dim=1) * bag_labels
        else:
            probs = torch.cat([1 - probs, probs * bag_labels[:,1].unsqueeze(-1)], dim=1)
        topk = torch.sum(bag_labels, dim=0) / (bag_labels.shape[0] / 8)
        pos_idx = torch.topk(probs[:,1], int(topk[1])).indices
        neg_idx = torch.topk(probs[:,0], int(topk[0])).indices
        for feat in x[pos_idx]:
            self.prototypes[1] = alpha * self.prototypes[1] + (1 - alpha) * feat
        for feat in x[neg_idx]:
            self.prototypes[0] = alpha * self.prototypes[0] + (1 - alpha) * feat

    def update_instance_labels(self, x: torch.Tensor, curr_labels: torch.Tensor, warm_up: bool) -> torch.Tensor:

        beta = 1.0 if warm_up else 0.99
        bag_labels = (curr_labels.reshape(-1) > 0.0).float()
        logits = torch.mm(x, self.prototypes.T)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        ins_labels = beta * curr_labels.reshape(-1) + (1 - beta) * (preds * bag_labels)
        return ins_labels.detach().clone()

    def forward(self, x: torch.Tensor, ins_labels: torch.Tensor, warm_up: bool, no_update: bool = False, alpha: float = 0.95) -> torch.Tensor:

        # extract features
        sh = x.shape
        x = x.reshape(sh[0] * sh[1], sh[2], sh[3], sh[4], sh[5])
        x = self.backbone(x)
        x = x.reshape(sh[0], sh[1], -1)

        # encode features
        x = self.encoder(x)

        # compute instance logits
        ins_logits = self.ins_head(x)

        # update prototypes and instance labels
        x = x.reshape(sh[0] * sh[1], -1)
        if not no_update:
            self.update_prototypes(x, ins_logits, ins_labels, warm_up, alpha)
        new_ins_labels = self.update_instance_labels(x, ins_labels, warm_up)
        x = x.reshape(sh[0], sh[1], -1)

        # aggregate instance features into bag features
        t = self.query_embed.weight
        t = t.unsqueeze(0).expand(x.shape[0], -1, -1)
        x = self.decoder(t, x)
        bag_logits = self.bag_head(x.view(x.shape[0], -1))
        return ins_logits, bag_logits, new_ins_labels