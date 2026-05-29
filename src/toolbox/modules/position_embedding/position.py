import math

import torch
import torch.nn as nn
from einops import rearrange


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, device, max_len = 16384):
        super().__init__()
        self.device = device

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, device = self.device)
        pe.require_grad = False

        position = torch.arange(0, max_len, device = self.device).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, device = self.device) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, length_idx = -1):
        length = x.shape[length_idx]
        return self.pe[:, :length]


class BiasedPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len, device):
        super().__init__()
        self.device = device
        self.d_model = d_model

        position = torch.arange(0, max_len, device = self.device).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, device = self.device) * -(math.log(10000.0) / d_model)).exp()
        self.register_buffer('position', position)
        self.register_buffer('div_term', div_term)

        self.Wt = nn.Linear(1, d_model // 2 + (1 if d_model % 2 else 0), bias = False, device = self.device)


    def forward(self, seq_len, interval, position_start_index = 0):
        phi = self.Wt(interval.unsqueeze(-1))                                  # [..., d_model // 2 + 1 if d_model % 2 else 0]

        arc = (self.position[position_start_index:seq_len + position_start_index] * self.div_term)
                                                                               # [seq_len, d_model // 2 + 1 if d_model % 2 else 0]
        einop = f'... -> {"() " * (len(phi.shape) - 2)}...'
        arc = rearrange(arc, einop)                                            # [..., seq_len, d_model // 2 + 1 if d_model % 2 else 0]

        pe_cos = torch.cos(arc + phi)                                          # [..., seq_len, d_model // 2 + 1 if d_model % 2 else 0]
        pe_sin = torch.sin(arc + phi)                                          # [..., seq_len, d_model // 2 + 1 if d_model % 2 else 0]
        if self.d_model % 2 == 1:
            pe_sin = pe_sin[..., :-1]                                          # [..., seq_len, d_model // 2]
        return torch.cat([pe_sin, pe_cos], dim=-1)                             # [..., seq_len, d_model // 2]
