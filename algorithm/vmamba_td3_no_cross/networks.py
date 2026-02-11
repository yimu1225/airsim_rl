import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..cnn_modules import CNN

# Import VMamba from virtual environment
from vmamba.vmamba import VSSBlock, VSSM as VMambaModel, SS2D

# Create compatibility classes for VMamba
class PatchPartition(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4, channel_first=False):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.channel_first = channel_first
        
    def forward(self, x):
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        if not self.channel_first:
            x = x.permute(0, 2, 3, 1)  # (B, H/patch_size, W/patch_size, embed_dim)
        return x

class DownsampleV3(nn.Module):
    def __init__(self, dim=96, out_dim=192):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(dim, out_dim, kernel_size=2, stride=2),
        )
        self.norm = nn.LayerNorm(out_dim)
        
    def forward(self, x):
        # x is (B, H, W, C) - NHWC format
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        x = self.down(x)  # (B, out_dim, H/2, W/2)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # (B, H/2, W/2, out_dim)
        x = self.norm(x)  # Apply normalization
        return x

# Compatibility aliases
VMamba = VMambaModel
Downsample = DownsampleV3
GlobalAvgPool = nn.AdaptiveAvgPool2d


class VMambaRLTiny(nn.Module):
    """
    轻量级 VMamba 模型，专为强化学习设计
    参数量大幅减少，适合在线训练
    修改版：移除GlobalAvgPool，输出(B, L, C)保留空间信息用于后续处理
    仅使用2个stage以降低计算复杂度
    """
    def __init__(self, 
                 in_chans=4,  # 修改为支持4帧堆叠输入
                 hidden_dim=64,
                 num_vss_blocks=[2, 2],  # 动态 Stage 数量
                 drop_path_rate=0.1,
                 layer_scale_init=1e-6,
                 patch_size=4,
                 ssm_d_state=16,
                 ssm_ratio=2.0,
                 mlp_ratio=4.0,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        total_depth = sum(num_vss_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        
        # Stem: 直接接受4通道输入，输出 (B, H/patch_size, W/patch_size, hidden_dim)
        self.stem = PatchPartition(
            in_chans=in_chans, 
            embed_dim=hidden_dim, 
            patch_size=patch_size,
            channel_first=False
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        
        current_dim = hidden_dim
        offset = 0

        # 动态构建 Stages
        for i, num_blocks in enumerate(num_vss_blocks):
            # 构建当前 Stage 的 Blocks
            stage_blocks = nn.Sequential(
                *[VSSBlock(
                    hidden_dim=current_dim, 
                    drop_path=dpr[offset + j], 
                    layer_scale_init=layer_scale_init,
                    ssm_d_state=ssm_d_state,
                    ssm_ratio=ssm_ratio,
                    mlp_ratio=mlp_ratio,
                    forward_type="v05"
                ) for j in range(num_blocks)]
            )
            self.stages.append(stage_blocks)
            offset += num_blocks
            
            # 构建 Downsample层 (除了最后一个 Stage)
            if i < len(num_vss_blocks) - 1:
                down = DownsampleV3(dim=current_dim, out_dim=current_dim * 2)
                self.downsamples.append(down)
                current_dim = current_dim * 2
        
        self.feature_dim = current_dim
    
    def forward(self, x):
        # x: (B, C, H, W) 其中 C=4 (4帧堆叠)
        x = self.stem(x)            # (B, H/4, W/4, C)
        
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        
        # 将特征图转换为序列格式：(B, H, W, C) -> (B, H*W, C)
        B, H, W, C = x.shape
        x = x.view(B, H * W, C)  # (B, L, C) 其中 L=H*W 是patch数量
        
        return x


class VMambaVisualEncoder(nn.Module):
    """
    Visual Encoder using VMamba (Pure PyTorch Implementation).
    Processes 4-frame stacked images and outputs spatial feature sequence.
    修改版：支持4帧通道堆叠输入，输出(B, L, C)用于后续处理
    """
    def __init__(self, input_height, input_width,  input_channels=4, args=None):
        super().__init__()
        
        # 从args获取配置参数
        patch_size = args.vmamba_patch_size
        hidden_dim = args.vmamba_hidden_dim
        num_vss_blocks = args.vmamba_num_vss_blocks
        drop_path_rate = args.vmamba_drop_path_rate
        layer_scale_init = args.vmamba_layer_scale_init
        ssm_d_state = args.vmamba_ssm_d_state
        ssm_ratio = args.vmamba_ssm_ratio
        mlp_ratio = args.vmamba_mlp_ratio
        
        self.backbone = VMambaRLTiny(
            in_chans=input_channels,  # 4通道输入
            hidden_dim=hidden_dim,
            num_vss_blocks=num_vss_blocks,
            drop_path_rate=drop_path_rate,
            layer_scale_init=layer_scale_init,
            patch_size=patch_size,
            ssm_d_state=ssm_d_state,
            ssm_ratio=ssm_ratio,
            mlp_ratio=mlp_ratio
        )
        
        # 输出特征维度是VMamba的feature_dim
        self.feature_dim = self.backbone.feature_dim
        self.repr_dim = self.feature_dim

    def forward(self, x):
        """
        x: (B, C, H, W) 其中 C=4 (4帧堆叠)
        Returns: (B, L, C) where L is number of patches, C is feature dimension
        """
        x = self.backbone(x)  # (B, L, C)
        return x


class StateMLP(nn.Module):
    """
    Encode low-dimensional state sequence or single state with an MLP.
    Input: (B, K, state_dim) or (B, state_dim)
    Output: (B, K, state_feature_dim) or (B, state_feature_dim)
    """
    def __init__(self, state_dim, state_feature_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, state_feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(state_feature_dim, state_feature_dim),
        )

    def forward(self, x):
        # 统一处理 2D/3D 或任意维度输入，nn.Linear 会自动作用于最后一维
        return self.net(x)


class SimpleFusionEncoder(nn.Module):
    """
    简单的融合编码器：不使用交叉注意力，使用全局平均池化和简单拼接
    Input: visual_feat (B, L, C) - VMamba输出的空间特征序列
           current_state (B, S) - 当前无人机状态向量
    Output: fused_context (B, H) - 融合后的上下文向量
    """
    def __init__(self, visual_dim, state_dim, hidden_dim):
        super().__init__()
        # 状态特征扩展网络：将状态映射到与视觉特征相同的维度
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 视觉特征投影：将视觉特征映射到hidden_dim
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 融合后的处理网络
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, visual_feat, current_state):
        # visual_feat: (B, L, C) 来自VMamba的空间特征序列
        # current_state: (B, S) 当前时刻的状态向量
        
        # 对视觉特征进行全局平均池化
        visual_pooled = torch.mean(visual_feat, dim=1)  # (B, C)
        
        # 将视觉特征投影到hidden_dim
        visual_feat_proj = self.visual_proj(visual_pooled)  # (B, hidden_dim)
        
        # 将状态特征投影到hidden_dim
        state_feat = self.state_proj(current_state)  # (B, hidden_dim)
        
        # 简单拼接
        fused_feat = torch.cat([visual_feat_proj, state_feat], dim=-1)  # (B, hidden_dim * 2)
        
        # 通过融合网络
        fused_context = self.fusion_net(fused_feat)  # (B, hidden_dim)
        
        return fused_context


class ConcatFusionEncoder(nn.Module):
    """
    拼接融合编码器：将视觉特征和状态特征直接拼接后通过MLP
    Input: visual_feat (B, L, C) - VMamba输出的空间特征序列
           current_state (B, S) - 当前无人机状态向量
    Output: fused_context (B, H) - 融合后的上下文向量
    """
    def __init__(self, visual_dim, state_dim, hidden_dim):
        super().__init__()
        # 状态特征扩展网络：将状态映射到与视觉特征相同的维度
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 视觉特征投影：将视觉特征映射到hidden_dim
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 融合后的处理网络
        self.fusion_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, visual_feat, current_state):
        # visual_feat: (B, L, C) 来自VMamba的空间特征序列
        # current_state: (B, S) 当前时刻的状态向量
        
        # 对视觉特征进行全局平均池化
        visual_pooled = torch.mean(visual_feat, dim=1)  # (B, C)
        
        # 将视觉特征投影到hidden_dim
        visual_feat_proj = self.visual_proj(visual_pooled)  # (B, hidden_dim)
        
        # 将状态特征投影到hidden_dim
        state_feat = self.state_proj(current_state)  # (B, hidden_dim)
        
        # 简单拼接
        fused_feat = torch.cat([visual_feat_proj, state_feat], dim=-1)  # (B, hidden_dim * 2)
        
        # 通过融合网络
        fused_context = self.fusion_net(fused_feat)  # (B, hidden_dim)
        
        return fused_context


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.LayerNorm(repr_dim),
                                    nn.Linear(repr_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = (2.0 / m.in_features) ** 0.5
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, std=None):
        mu = self.policy(obs)
        mu = torch.tanh(mu)
        return mu


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, hidden_dim):
        super().__init__()

        self.Q1 = nn.Sequential(
            nn.LayerNorm(repr_dim + action_shape[0]),
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.LayerNorm(repr_dim + action_shape[0]),
            nn.Linear(repr_dim + action_shape[0], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            std = (2.0 / m.in_features) ** 0.5
            nn.init.trunc_normal_(m.weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, obs, action):
        h_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)
        return q1, q2
