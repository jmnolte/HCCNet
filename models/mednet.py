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
    
class PositionalEncoding(nn.Module):

    def __init__(
            self, 
            d_model: int, 
            dropout: float = 0.1
        ) -> None:

        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
        self.d_model = d_model

    def forward(
            self, 
            x: torch.Tensor, 
            pos_token: torch.Tensor
        ) -> torch.Tensor:

        device, dtype = x.device, x.dtype
        pe = torch.zeros(x.shape, device=device, dtype=dtype)
        pos_token = pos_token.unsqueeze(-1).expand(-1, -1, int(self.d_model / 2))
        pe[:, :, 0::2] = torch.sin(pos_token * self.div_term.expand_as(pos_token))
        pe[:, :, 1::2] = torch.cos(pos_token * self.div_term.expand_as(pos_token))
        x = x + pe
        return self.dropout(x)

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
            max_len: int = 512
        ) -> None:

        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_length, embedding_dim]``
        """
        if self.training:
            pos_encod = torch.zeros_like(x, device=x.device)
            pos_sampled = torch.zeros(x.size(0), x.size(1), device=x.device)
            for i in range(x.size(0)):
                pos_perm = torch.randperm(self.max_len)
                pos_sampled[i] = pos_perm[:x.size(1)].sort()[0]
                pos_encod[i] = self.pe[:, pos_sampled[i].long(), :]
            return pos_encod
        else:
            return self.pe[:, :x.size(1), :]

    
class Embedding(nn.Module):

    def __init__(
            self, 
            d_model: int, 
            max_len: int = 512,
            dropout: float = 0.1,
            eps: float = 1e-6
        ) -> None:

        super().__init__()
        self.pos_encod = PositionalEncoding(d_model, max_len)
        self.age_embed = nn.Embedding(100, d_model, padding_idx=0)
        self.eti_embed = nn.Embedding(10, d_model)
        self.sex_embed = nn.Embedding(2, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(
            self, 
            x: torch.Tensor,
            pt_age: torch.Tensor,
            pt_etiology: torch.Tensor,
            pt_sex: torch.Tensor
        ) -> torch.Tensor:

        embed = x + self.pos_encod(x) \
            + self.age_embed(pt_age.long()) \
            + self.eti_embed(pt_etiology.long()) \
            + self.sex_embed(pt_sex.long())
        embed = self.norm(embed)
        return self.dropout(embed)

class MedNet(nn.Module):

    def __init__(
            self, 
            backbone: nn.Module,
            num_classes: int = 1000,
            max_len: int = 512,
            num_layers: int = 6,
            dropout: float = 0.1,
            activation: str = 'gelu',
            eps: float = 1e-6,
            norm_first: bool = True
        ) -> None:

        super().__init__()
        self.d_model = _extract_num_features(backbone)
        _remove_last_layer(backbone)
        self.backbone = backbone
        self.embedding = Embedding(
            d_model=self.d_model,
            max_len=max_len,
            dropout=dropout,
            eps=eps)
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
            pt_info: List[torch.Tensor],
        ) -> torch.Tensor:

        B, _, _ = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        pad_mask = torch.cat([pad_mask[:, 0].unsqueeze(-1), pad_mask], dim=1)
        for idx, info in enumerate(pt_info):
            if idx == 0:
                age_cls = torch.max(info, dim=1).values + 1
                pt_info[idx] = torch.cat([age_cls.unsqueeze(-1), info], dim=1)
            else:
                pt_info[idx] = torch.cat([info[:, 0].unsqueeze(-1), info], dim=1)
        return x, pad_mask, pt_info

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
            pad_mask: torch.Tensor | None = None,
            pt_info: List[torch.Tensor] | None = None
        ) -> torch.Tensor:

        x = self.extract_features(x)
        x, pad_mask, pt_info = self.add_cls_token(x, pad_mask, pt_info)
        pt_age, pt_etiology, pt_sex = pt_info[0], pt_info[1], pt_info[2]
        x = self.embedding(x, pt_age, pt_etiology, pt_sex)
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=pad_mask)
        x = x.permute(1, 0, 2)
        x = self.pooler(x, padding_mask=pad_mask)
        return self.head(x)