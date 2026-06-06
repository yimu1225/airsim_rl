import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..cnn_modules import CNN


# ================================================================
#  SubNetwork1 - Actor 视觉分支 (perception-related state so)
#  输入: 4帧深度图 (B, 4, H, W) → CNN 逐帧编码 → 拼接 → (B, 256)
#  输出: ao = MLP(so_repr) → Tanh → (-1, 1)
#  说明: 拥有独立 CNN，不与 Critic 共享
# ================================================================

class SubNetwork1(nn.Module):
    """Actor 视觉子网络：4 帧深度图通过共享 CNN 逐帧编码，拼接后经 MLP 输出 ao。"""
    def __init__(self, depth_shape, hidden_dims, out_dim, encoder_output_dim=64):
        super().__init__()
        _, H, W = depth_shape
        # 逐帧编码: (B, 4, H, W) → (B*4, 1, H, W) → (B*4, 64) → (B, 256)
        self.encoder = CNN(
            input_height=H, input_width=W,
            input_channels=4, output_dim=encoder_output_dim,
            frame_wise=True, flatten_all_tokens=True,
        )
        self.cat_repr_dim = self.encoder.repr_dim  # 4 * 64 = 256

        self.input_norm = nn.LayerNorm(self.cat_repr_dim)
        h1, h2 = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(self.cat_repr_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, out_dim),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, depth):
        """逐帧编码深度图。

        Args:
            depth: (B, 4, H, W)  4帧堆叠深度图
        Returns:
            ao:      (B, out_dim)           经 Tanh 映射到 (-1, 1)
            so_repr: (B, 4*encoder_output_dim)  4帧视觉特征拼接
        """
        so_repr = self.encoder(depth)             # CNN 内部逐帧处理 → (B, 256)
        ao = self.mlp(self.input_norm(so_repr))
        return ao, so_repr


# ================================================================
#  SubNetwork2 - Actor 目标分支 (target-related state sg)
#  输入: sg = [dh, dv, phi_h, phi_v] (B, 4)
#  输出: ag = MLP(sg) → Tanh → (-1, 1)
# ================================================================

class SubNetwork2(nn.Module):
    """Actor 目标子网络：将 sg 映射为与目标相关的动作分量 ag。"""
    def __init__(self, sg_dim, hidden_dims, out_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(sg_dim)
        h1, h2 = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(sg_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, out_dim),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, sg):
        """Args: sg (B, 4)  返回: ag (B, out_dim)"""
        return self.mlp(self.input_norm(sg))


# ================================================================
#  GlobalActor - 融合 ao、ag 与完整状态 S，生成最终动作
#  输入: concat[ao, ag, so_repr, sg, su]
#        其中 S = [so_repr(256), sg(4), su(7)] = 267 维
#  输出: action = MLP → Tanh → (-1, 1)^action_dim
# ================================================================

class GlobalActor(nn.Module):
    """全局 Actor：融合子网络输出 (ao, ag) 与完整状态表征 S 输出最终动作。"""
    def __init__(self, input_dim, hidden_dims, action_dim):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_dim)
        h1, h2 = hidden_dims
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, action_dim),
            nn.Tanh(),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, ao, ag, state_repr):
        """Args:
            ao, ag: 子网络输出
            state_repr: 完整状态 S = concat[so_repr, sg, su]
        Returns: action (B, action_dim) in (-1, 1)
        """
        x = torch.cat([ao, ag, state_repr], dim=-1)
        return self.mlp(self.input_norm(x))


# ================================================================
#  Critic (Q-network) — 拥有独立 CNN，不与 Actor 共享
#  架构: CNN(depth) → so_repr(256)
#        concat[so_repr(256), sg(4), su(7), action] → LayerNorm → [400,300] → Q
# ================================================================

class Critic(nn.Module):
    """Critic 网络，拥有独立的视觉编码器（不与 Actor 共享 CNN）。"""
    def __init__(self, depth_shape, sg_dim, su_dim,
                 action_dim, hidden_dims, encoder_output_dim=64):
        super().__init__()
        _, H, W = depth_shape
        # 独立 CNN（不与 SubNetwork1 共享参数）
        self.encoder = CNN(
            input_height=H, input_width=W,
            input_channels=4, output_dim=encoder_output_dim,
            frame_wise=True, flatten_all_tokens=True,
        )
        so_dim = self.encoder.repr_dim  # 256
        total_input_dim = so_dim + sg_dim + su_dim + action_dim

        self.input_norm = nn.LayerNorm(total_input_dim)
        h1, h2 = hidden_dims
        self.q_net = nn.Sequential(
            nn.Linear(total_input_dim, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, depth, sg, su, action):
        """计算 Q(s, a)。

        Args:
            depth:  (B, 4, H, W)  4帧堆叠深度图
            sg:     (B, 4)        目标相关状态
            su:     (B, 7)        无人机自身状态
            action: (B, action_dim)
        Returns:
            q: (B, 1)
        """
        so_repr = self.encoder(depth)               # (B, 256)
        x = torch.cat([so_repr, sg, su, action], dim=-1)
        q = self.q_net(self.input_norm(x))
        return q, None  # 保留第二个返回值以保持接口兼容
