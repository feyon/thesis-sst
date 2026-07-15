"""Factory: bangun model deep learning dari konfigurasi."""
from .lstm import LSTMRegressor
from .transformer import TransformerRegressor
from .hybrid import HybridLSTMTransformer


def build_model(name: str, cfg: dict):
    m = cfg["model"]
    if name == "lstm":
        return LSTMRegressor(input_size=1, **m["lstm"])
    if name == "transformer":
        return TransformerRegressor(input_size=1, **m["transformer"])
    if name == "hybrid":
        return HybridLSTMTransformer(input_size=1, **m["hybrid"])
    raise ValueError(f"Model tidak dikenal: {name}")
