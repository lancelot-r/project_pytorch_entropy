import torch
import torch.nn as nn


class MyTransformerEstimator(nn.Module):
    """
    Minimal Transformer model for entropy estimation from datasets.
    """

    def __init__(self, d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.readout = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(x.size(1), device=x.device)[None, :] >= lengths[:, None]
        h = self.input_proj(x)
        h = self.encoder(h, src_key_padding_mask=mask)
        pooled = h.mean(dim=1)
        return self.readout(pooled).squeeze(-1)
