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
            padding_mask = torch.ones_like(hidden_states[:, :, 1])
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
            padding_mask = torch.zeros_like(hidden_states[:, :, 1])
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
            num_classes: int = 1000
        ) -> None:

        super().__init__()
        self.norm = nn.LayerNorm([d_model])
        self.head = nn.Linear(d_model, num_classes)

    def forward(
            self, 
            x: torch.Tensor
        ) -> torch.Tensor:

        out = self.head(self.norm(x))
        return out

class MedNet(nn.Module):

    def __init__(
            self, 
            backbone: nn.Module,
            num_classes: int = 1000,
            num_layers: int = 4,
            dropout: float = 0.1,
            pooling_mode: str = 'cls',
            activation: str = 'gelu',
            eps: float = 1e-5,
            norm_first: bool = True
        ) -> None:

        super().__init__()
        self.d_model = _extract_num_features(backbone)
        _remove_last_layer(backbone)
        self.backbone = backbone
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=self.d_model // 64, 
            dim_feedforward=self.d_model * 4, 
            dropout=dropout,
            activation=activation,
            layer_norm_eps=eps,
            norm_first=norm_first)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        if pooling_mode == 'cls':
            self.pooler = CLSPooling(d_model=self.d_model)
        elif pooling_mode == 'att':
            self.pooler = AttentionPooling(d_model=self.d_model)
        elif pooling_mode == 'mean':
            self.pooler = MeanPooling(d_model=self.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.d_model)) if pooling_mode == 'cls' else None
        self.pooling_mode = pooling_mode
        self.head = Classifier(self.d_model, num_classes)
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
        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)
    
    def forward(
            self, 
            x: torch.Tensor, 
            pos_token: torch.Tensor,
            padding_mask: torch.Tensor | None
        ) -> torch.Tensor:
        
        B, S, C, H, W, D = x.shape
        x = x.reshape(B * S, C, H, W, D)
        x = self.backbone(x)
        x = x.reshape(B, S, self.d_model)
        cls_pos_mask = (torch.zeros((B, 1))).to(pos_token.device)
        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_token, x], dim=1)
            pos_token = torch.hstack([cls_pos_mask, pos_token])
        x = self.positional_encoding(x, pos_token)
        if padding_mask is not None:
            padding_mask = torch.hstack([cls_pos_mask, padding_mask]) if self.pooling_mode == 'cls' else padding_mask
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        x = x.permute(1, 0, 2)
        x = self.pooler(x, padding_mask=padding_mask)
        out = self.head(x)
        return out