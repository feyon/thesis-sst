"""Model LSTM standalone (baseline deep learning).

Berperan sebagai pembanding untuk mengukur kontribusi komponen Transformer.
Input univariat: [batch, lookback, 1].
"""
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int = 1, hidden_size: int = 50,
                 num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size,
            num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):                 # x: [B, N, 1]
        out, _ = self.lstm(x)             # out: [B, N, H]
        last = out[:, -1, :]              # hidden state terakhir
        return self.head(last).squeeze(-1)  # [B]
