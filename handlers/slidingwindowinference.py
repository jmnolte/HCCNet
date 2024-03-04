from typing import Any
import torch.nn as nn
import torch

class SlidingWindowInferer:

    def __init__(
            self,
            model: nn.Module,
            device: torch.device,
            max_length: int = 4,
            window_shift: int = 1
        ) -> None:
        
        self.model = model
        self.device = device
        self.max_length = max_length
        self.window_shift = window_shift

    def __call__(
            self,
            x: torch.Tensor,
            pos_token: torch.Tensor
        ) -> torch.Tensor:

        seq_len = x.shape[1]
        if seq_len <= self.max_length:
            return self.model(x.to(self.device), padding_mask=None, pos_token=pos_token.to(self.device))
        else: 
            count = torch.zeros(1).to(self.device)
            logits_sum = torch.zeros(1, 1).to(self.device)
            for start in range(0, seq_len - self.max_length + 1, self.window_shift):
                count += 1
                end = start + self.max_length
                x_window = x[start:end]
                pos_token_window = pos_token[start:end]
                logits_sum += self.model(x_window.to(self.device), mask=None, pos_token=pos_token_window.to(self.device))

            logits_avg = logits_sum / count
            return logits_avg



