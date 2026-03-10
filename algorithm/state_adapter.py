import torch.nn as nn


class StateAdapter(nn.Module):
    """Adapt raw base-state vectors into a compact feature space."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.input_norm(x)
        return self.net(x)
