"""Model Hibrida LSTM-Transformer (sekuensial).

Alur: LSTM menangkap pola temporal lokal -> seluruh urutan hidden state
diproyeksikan linear ke proj_dim (kelipatan nhead) -> Transformer encoder
menangkap dependensi jangka panjang -> pooling -> prediksi 1 nilai SST.

Catatan: lstm_hidden=50 tidak habis dibagi nhead=4. Diselesaikan dengan
lapisan proyeksi linear 50 -> 52 (52 % 4 == 0) sebelum masuk Transformer.
"""
import torch
import torch.nn as nn
from .transformer import PositionalEncoding


class HybridLSTMTransformer(nn.Module):
    def __init__(self, input_size: int = 1, lstm_hidden: int = 50,
                 proj_dim: int = 52, nhead: int = 4,
                 num_transformer_layers: int = 2, dim_feedforward: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        assert proj_dim % nhead == 0, "proj_dim harus habis dibagi nhead"

        self.lstm = nn.LSTM(input_size, lstm_hidden, batch_first=True)
        self.proj = nn.Linear(lstm_hidden, proj_dim)   # 50 -> 52
        self.pos = PositionalEncoding(proj_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=proj_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_transformer_layers)
        self.head = nn.Linear(proj_dim, 1)

    def forward(self, x):                 # x: [B, N, 1]
        seq, _ = self.lstm(x)             # [B, N, lstm_hidden]  (seluruh urutan)
        h = self.proj(seq)                # [B, N, proj_dim]
        h = self.pos(h)
        h = self.encoder(h)               # [B, N, proj_dim]
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(-1)
