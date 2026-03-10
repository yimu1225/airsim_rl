import torch.nn as nn


class StateAdapter(nn.Module):
    """Adapt raw base-state vectors into a compact feature space."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
