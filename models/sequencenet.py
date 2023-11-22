import torch
import torch.nn as nn
import numpy as np


class SequenceNet(nn.Module):

    def __init__(
            self, 
            model: nn.Module,
            num_classes: int = 2,
            nhead: int = 4, 
            nlayers: int = 6, 
            dropout: float = 0.5
        ) -> None:
        super().__init__()
        self.embed_dim = self._extract_num_features(model)
        self.model = self._remove_last_layer(model)
        self.model = model
        transformer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=nhead, dropout=dropout)
        self.transformer = nn.TransformerEncoder(transformer, num_layers=nlayers)
        self.fc = nn.Linear(self.embed_dim, num_classes)

    def _remove_last_layer(self, model) -> None:
        return setattr(model, list(model._modules)[-1], nn.Identity())
    
    def _extract_num_features(self, model) -> int:
        for layer in model.children():
            if isinstance(layer, nn.Linear):
                num_features = layer.in_features
        return num_features

    def generate_positional_encodings(self, encodings: torch.Tensor) -> torch.Tensor:
        B, L = encodings.shape
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * -(np.log(10000.0) / self.embed_dim))
        pos_encodings = torch.zeros(B, L, self.embed_dim)
        pos_encodings[:, :, 0::2] = torch.sin(encodings.unsqueeze(2) * div_term)
        pos_encodings[:, :, 1::2] = torch.cos(encodings.unsqueeze(2) * div_term)
        return pos_encodings

    def forward(
            self, 
            x: torch.Tensor, 
            x_mask: torch.Tensor = None, 
            encodings: torch.Tensor = None,
            batch_size: int = 4
        ) -> torch.Tensor:
        
        x = self.model(x)

        sh = x.shape
        x = x.reshape(batch_size, int(sh[0] / batch_size), sh[1])
        if encodings is not None:
            x = x + self.generate_positional_encodings(encodings)
        x = x.permute(1, 0, 2)
        if x_mask is not None:
            x_mask = x_mask.permute(1, 0, 2)
        x = self.transformer(x, x_mask)
        x = self.fc(x)
        x = x.permute(1, 0, 2)
        sh = x.shape
        out = x.reshape(sh[0] * sh[1], sh[2])
        return out