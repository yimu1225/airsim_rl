import torch
import torch.nn as nn

from algorithm.VMSAC.networks import Actor, Critic, STVimEncoder


class SafetyConstraintHead(nn.Module):
    def __init__(self, latent_dim, action_dim):
        super().__init__()
        self.fc_g = nn.Linear(latent_dim, action_dim)
        self.fc_h = nn.Linear(latent_dim, 1)

    def forward(self, latent_state):
        g = self.fc_g(latent_state)
        h = self.fc_h(latent_state)
        return g, h


def safety_project_actions(a_raw, g, h, eps=1e-6):
    constraint_val = (g * a_raw).sum(dim=-1, keepdim=True) + h
    correction = torch.clamp(constraint_val, min=0.0) / (
        torch.norm(g, dim=-1, keepdim=True).pow(2) + eps
    )
    a_safe = a_raw - correction * g
    return a_safe, constraint_val


__all__ = [
    "Actor",
    "Critic",
    "STVimEncoder",
    "SafetyConstraintHead",
    "safety_project_actions",
]
