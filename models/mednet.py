from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_

def _remove_last_layer(backbone) -> None:
        return setattr(backbone, list(backbone._modules)[-1], nn.Identity())
    
def _extract_num_features(backbone) -> int:
    for layer in backbone.children():
        if isinstance(layer, nn.Linear):
            num_features = layer.in_features
    return num_features

class CLSPooling(nn.Module):

    def __init__(
            self,
            d_model: int
        ) -> None:

        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(
            self, 
            hidden_states: torch.Tensor, 
            padding_mask: torch.Tensor | None
        ) -> torch.Tensor:

        cls_token = hidden_states[:, 0]
        out = self.linear(cls_token)
        out = self.activation(out)
        return out

class MeanPooling(nn.Module):

    def __init__(
            self, 
            d_model: int
        ) -> None:

        super().__init__()

    def forward(
            self, 
            hidden_states: torch.Tensor, 
            padding_mask: torch.Tensor | None
        ) -> torch.Tensor:

        if padding_mask is None:
            padding_mask = torch.ones_like(hidden_states[:, :, 0])
        else:
            padding_mask = torch.where(padding_mask == 1, 0, 1)
        padding_mask = padding_mask.unsqueeze(-1)
        hidden_states = torch.mul(hidden_states, padding_mask)
        out = hidden_states.sum(dim=1) / padding_mask.sum(dim=1)
        return out

class AttentionPooling(nn.Module):

    def __init__(
            self, 
            d_model: int
        ) -> None:

        super().__init__()
        self.attention = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh(), nn.Linear(d_model, 1))

    def forward(
            self, 
            hidden_states: torch.Tensor, 
            padding_mask: torch.Tensor | None
        ) -> torch.Tensor:

        if padding_mask is None:
            padding_mask = torch.zeros_like(hidden_states[:, :, 0])
        else:
            padding_mask = torch.where(padding_mask == 1, float('-inf'), 0)
        padding_mask = padding_mask.unsqueeze(-1)
        a = self.attention(hidden_states)
        a = torch.softmax(a + padding_mask, dim=1)
        out = torch.sum(hidden_states * a, dim=1)
        return out

class Classifier(nn.Module):

    def __init__(
            self, 
            d_model: int, 
            num_classes: int = 1000,
            eps: float = 1e-6
        ) -> None:

        super().__init__()
        self.norm = nn.LayerNorm([d_model], eps=eps)
        self.head = nn.Linear(d_model, num_classes)

    def forward(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:

        out = self.head(self.norm(x))
        return out
    
class PositionalEncoding(nn.Module):

    def __init__(
            self, 
            d_model: int, 
            max_len: int = 512,
            dropout: float = 0.1
        ) -> None:

        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(p=dropout)

    def forward(
            self, 
            x: torch.Tensor, 
            pos: torch.Tensor
        ) -> torch.Tensor:

        pos_encod = self.pe[:, pos.long(), :].squeeze(0)
        return self.dropout(x + pos_encod)

class MedNet(nn.Module):

    '''
    CNN-Transformer model where the CNN extracts image features and the Transformer models the sequence of feature
    vectors.
    '''

    def __init__(
            self, 
            backbone: nn.Module,
            num_classes: int = 1000,
            pretrain: bool = False,
            max_len: int = 512,
            num_layers: int = 6,
            dropout: float = 0.1,
            activation: str = 'gelu',
            eps: float = 1e-6,
            norm_first: bool = True
        ) -> None:

        '''
        Args:
            backbone (nn.Module): The CNN backbone model.
            num_classes (int): The number of classes.
            pretrain (bool): Whether to perform classification on downstream task or masked pretraining.
            max_len (int): The maximum sequence length.
            num_layers (int): The number of transformer encoder layers.
            droput (float): The dropout rate.
            activation (str): The activation function in the encoder layers.
            eps (float): Epsilon to stabilize training. Default: 1e-6.
            norm_first (bool): Whether the use the Pre-LN transformer encoder.
        '''

        super().__init__()
        self.d_model = _extract_num_features(backbone)
        _remove_last_layer(backbone)
        self.backbone = backbone
        self.pretrain = pretrain
        self.positional_encoding = PositionalEncoding(
            d_model=self.d_model, 
            max_len=max_len, 
            dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.d_model // 128, 
            dim_feedforward=self.d_model * 4, 
            dropout=dropout,
            activation=activation,
            layer_norm_eps=eps,
            norm_first=norm_first)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers, 
            enable_nested_tensor=False)
        self.pooler = CLSPooling(d_model=self.d_model)
        self.head = Classifier(self.d_model, num_classes=num_classes, eps=eps)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model))
        self.apply(self._init_weights)

    def _init_weights(
            self, 
            m: nn.Module
        ) -> None:

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        trunc_normal_(self.cls_token, std=0.02)
    
    def add_cls_token(
            self,
            x: torch.Tensor,
            pad_mask: torch.Tensor,
            pos: torch.Tensor
        ) -> torch.Tensor:

        B, S, _ = x.shape
        pad_idx = torch.argmax(pad_mask, dim=1)
        pad_idx = torch.where(pad_idx == 0, S, pad_idx)

        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_token, x], dim=1)
        cls_token_pad = torch.zeros(B, 1).to(x.device)
        pad_mask = torch.cat([cls_token_pad, pad_mask], dim=1)
        pos = torch.cat([cls_token_pad, pos], dim=1)
        return x, pad_mask, pos

    def shuffle_sequence(
            self,
            x: torch.Tensor,
            pad_mask: torch.Tensor,
            prob: float = 0.5
        ) -> torch.Tensor:

        B, S, _ = x.shape
        pad_idx = torch.argmax(pad_mask, dim=1)
        pad_idx = torch.where(pad_idx == 0, S, pad_idx)
        prob_mask = torch.rand(B) < prob
        for i in range(B):
            if prob_mask[i]:
                if pad_idx[i] - 1 == 1:
                    prob_mask[i] = False
                rand_idx = torch.randperm(pad_idx[i] - 1)
                rand_idx += 1
                x[i, 1:pad_idx[i]] = x[i, rand_idx]
        labels = prob_mask.float()
        return x, labels.to(x.device)

    def extract_features(
            self,
            x: torch.Tensor,
        ) -> torch.Tensor:

        B, S, C, H, W, D = x.shape
        x = x.reshape(B * S, C, H, W, D)
        x = self.backbone(x)
        x = x.reshape(B, S, self.d_model)
        return x
    
    def forward(
            self, 
            x: torch.Tensor,
            pad_mask: torch.Tensor,
            pos: torch.Tensor
        ) -> torch.Tensor:

        x = self.extract_features(x)
        x, pad_mask, pos = self.add_cls_token(x, pad_mask, pos)
        if self.pretrain:
            x, labels = self.shuffle_sequence(x, pad_mask, prob=0.66)
        x = self.positional_encoding(x, pos)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        x = x.permute(1, 0, 2)
        x = self.pooler(x, padding_mask=pad_mask)
        x = self.head(x)
        if self.pretrain:
            return x, labels
        else:
            return x