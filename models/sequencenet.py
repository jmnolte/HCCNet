import torch
import torch.nn as nn
import numpy as np
import math

class SequenceNet(nn.Module):

    def __init__(
            self, 
            model: nn.Module,
            num_classes: int = 2,
            nhead: int = 8, 
            nlayers: int = 4, 
            dropout: float = 0.1
        ) -> None:
        super().__init__()
        self.embed_dim = self._extract_num_features(model)
        self._remove_last_layer(model)
        self.model = model
        self.positional_encoding = AbsTimeEncoding(d_model=self.embed_dim)
        transformer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(transformer, num_layers=nlayers)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def _remove_last_layer(self, model) -> None:
        return setattr(model, list(model._modules)[-1], nn.Identity())
    
    def _extract_num_features(self, model) -> int:
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                num_features = layer.in_features
        return num_features

    def forward(
            self, 
            x: torch.Tensor, 
            x_mask: torch.Tensor = None, 
            encodings: torch.Tensor = None,
            batch_size: int = 4
        ) -> torch.Tensor:
        
        x = self.model(x)
        sh = x.shape
        x = x.reshape(batch_size, sh[0] // batch_size, sh[1]) # B S C
        if encodings is not None:
            encodings = encodings.reshape(batch_size, -1)
            x = self.positional_encoding(x, encodings)
        if x_mask is not None:
            x_mask = x_mask.reshape(batch_size, -1)
        x = self.transformer(x, src_key_padding_mask=x_mask)
        x = self.fc(x)
        sh = x.shape
        out = x.reshape(sh[0] * sh[1], sh[2])
        return out


class AbsTimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Compute the positional encodings once in log space.
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
        self.register_buffer('div_term', div_term)
        self.d_model = d_model

    def forward(self, x, t):
        device, dtype = x.device, x.dtype
        pe = torch.zeros(x.shape, device=device, dtype=dtype)

        # repeat times into shape [b, t, dim]
        time_position = t.unsqueeze(-1).expand(-1, -1, int(self.d_model / 2))
        pe[:, :, 0::2] = torch.sin(time_position * self.div_term.expand_as(time_position))
        pe[:, :, 1::2] = torch.cos(time_position * self.div_term.expand_as(time_position))
        x = x + pe
        return self.dropout(x)