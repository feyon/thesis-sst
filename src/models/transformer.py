"""Model Transformer encoder standalone (baseline deep learning).

Berperan sebagai pembanding untuk mengukur kontribusi komponen LSTM.
"""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerRegressor(nn.Module):
    def __init__(self, input_size: int = 1, d_model: int = 64, nhead: int = 4,
                 num_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x):                 # x: [B, N, 1]
        h = self.input_proj(x)            # [B, N, d_model]
        h = self.pos(h)
        h = self.encoder(h)               # [B, N, d_model]
        pooled = h.mean(dim=1)            # global average pooling
        return self.head(pooled).squeeze(-1)
