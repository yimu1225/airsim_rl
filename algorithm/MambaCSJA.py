import torch
import torch.nn as nn


def _require_mamba():
    try:
        from mamba_ssm import Mamba
    except Exception as exc:
        raise ImportError(
            "mamba_ssm is required to instantiate MambaAttentionCNN."
        ) from exc
    return Mamba


class MambaCSJA(nn.Module):
    """
    Mamba-based Channel-Spatial Joint Attention.

    Channel branch models channels as a sequence. Spatial branch scans reduced
    spatial features in multiple directions. The two attention maps are fused
    explicitly instead of using a purely serial CBAM-style product.
    """
    def __init__(
        self,
        channels,
        d_state=16,
        d_conv=4,
        expand=2,
        spatial_dim=None,
        reduction=4,
        use_four_direction=True,
    ):
        super().__init__()
        channels = int(channels)
        if channels <= 0:
            raise ValueError("channels must be positive.")
        reduction = max(1, int(reduction))
        hidden = int(spatial_dim) if spatial_dim is not None else max(channels // reduction, 16)
        if hidden <= 0:
            raise ValueError("spatial_dim must be positive.")

        mamba_cls = _require_mamba()
        self.channels = channels
        self.spatial_dim = hidden
        self.use_four_direction = bool(use_four_direction)

        self.channel_embed = nn.Linear(2, hidden)
        self.channel_mamba = mamba_cls(d_model=hidden, d_state=d_state, d_conv=d_conv, expand=expand)
        self.channel_proj = nn.Linear(hidden, 1)

        self.spatial_reduce = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.spatial_norm = nn.GroupNorm(1, hidden)
        self.spatial_act = nn.SiLU(inplace=True)
        self.spatial_mamba = mamba_cls(d_model=hidden, d_state=d_state, d_conv=d_conv, expand=expand)
        self.spatial_proj = nn.Conv2d(hidden, 1, kernel_size=1, bias=True)

        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, kernel_size=1, bias=False),
            nn.GroupNorm(1, channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def channel_attention(self, x):
        batch_size, channels, _, _ = x.shape
        avg_pool = torch.mean(x, dim=(2, 3))
        max_pool = torch.amax(x, dim=(2, 3))
        tokens = torch.stack([avg_pool, max_pool], dim=-1)
        tokens = self.channel_embed(tokens)
        tokens = self.channel_mamba(tokens)
        attn = torch.sigmoid(self.channel_proj(tokens))
        return attn.view(batch_size, channels, 1, 1)

    def spatial_scan(self, feat):
        batch_size, dim, height, width = feat.shape

        seq_lr = feat.flatten(2).transpose(1, 2).contiguous()
        out_lr = self.spatial_mamba(seq_lr)
        out_lr = out_lr.transpose(1, 2).contiguous().view(batch_size, dim, height, width)

        if not self.use_four_direction:
            return out_lr

        seq_rl = torch.flip(seq_lr, dims=[1]).contiguous()
        out_rl = self.spatial_mamba(seq_rl)
        out_rl = torch.flip(out_rl, dims=[1]).contiguous()
        out_rl = out_rl.transpose(1, 2).contiguous().view(batch_size, dim, height, width)

        feat_t = feat.transpose(2, 3).contiguous()
        seq_tb = feat_t.flatten(2).transpose(1, 2).contiguous()
        out_tb = self.spatial_mamba(seq_tb)
        out_tb = out_tb.transpose(1, 2).contiguous().view(batch_size, dim, width, height)
        out_tb = out_tb.transpose(2, 3).contiguous()

        seq_bt = torch.flip(seq_tb, dims=[1]).contiguous()
        out_bt = self.spatial_mamba(seq_bt)
        out_bt = torch.flip(out_bt, dims=[1]).contiguous()
        out_bt = out_bt.transpose(1, 2).contiguous().view(batch_size, dim, width, height)
        out_bt = out_bt.transpose(2, 3).contiguous()

        return (out_lr + out_rl + out_tb + out_bt) / 4.0

    def spatial_attention(self, x):
        feat = self.spatial_reduce(x)
        feat = self.spatial_norm(feat)
        feat = self.spatial_act(feat)
        feat = self.spatial_scan(feat)
        return torch.sigmoid(self.spatial_proj(feat))

    def forward(self, x):
        ac = self.channel_attention(x)
        as_ = self.spatial_attention(x)

        xc = x * ac
        xs = x * as_
        xcs = x * ac * as_
        attn = self.fusion(torch.cat([xc, xs, xcs], dim=1))
        return x + self.gamma * x * attn


class MambaAttentionCNN(nn.Module):
    """
    Nature-style CNN for stacked depth frames with Mamba-CSJA attention.

    Frames are treated as input channels directly, not encoded one by one.
    """
    def __init__(
        self,
        input_height,
        input_width,
        input_channels=4,
        output_dim=64,
        d_state=16,
        d_conv=4,
        expand=2,
        spatial_dim=None,
        reduction=4,
        use_four_direction=True,
        attention_stages=("conv2",),
    ):
        super().__init__()
        self.input_height = int(input_height)
        self.input_width = int(input_width)
        self.input_channels = int(input_channels)
        self.repr_dim = 64
        if int(output_dim) != self.repr_dim:
            raise ValueError("MambaAttentionCNN outputs GAP features with fixed output_dim=64.")

        self.attention_stages = tuple(attention_stages)
        unsupported = set(self.attention_stages) - {"conv2"}
        if unsupported:
            raise ValueError(
                f"Unsupported attention_stages={sorted(unsupported)}. Only 'conv2' is implemented."
            )

        self.conv1 = nn.Conv2d(self.input_channels, 32, kernel_size=8, stride=4, padding=0)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.relu3 = nn.ReLU()

        self.csja_conv2 = MambaCSJA(
            channels=64,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            spatial_dim=spatial_dim,
            reduction=reduction,
            use_four_direction=use_four_direction,
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def _prepare_stacked_input(self, x):
        """
        Convert depth input to (B, input_channels, H, W).

        Supported inputs:
        - (H, W)
        - (T, H, W)
        - (B, T, H, W)
        - (B, T, 1, H, W)
        - (B, T, C, H, W), flattened to (B, T*C, H, W)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            if x.size(0) == self.input_channels:
                x = x.unsqueeze(0)
            elif self.input_channels == 1:
                x = x.unsqueeze(1)
            else:
                raise ValueError(
                    f"Expected first dim to be input_channels={self.input_channels}, got {x.size(0)}"
                )
        elif x.dim() == 5:
            batch_size, seq_len, channels, height, width = x.shape
            x = x.reshape(batch_size, seq_len * channels, height, width)

        if x.dim() != 4:
            raise ValueError(f"Unsupported MambaAttentionCNN input shape: {tuple(x.shape)}")
        if x.size(1) != self.input_channels:
            raise ValueError(f"Expected input_channels={self.input_channels}, got {x.size(1)}")
        if x.size(-2) != self.input_height or x.size(-1) != self.input_width:
            raise ValueError(
                f"Expected spatial size {(self.input_height, self.input_width)}, got {tuple(x.shape[-2:])}"
            )
        return x.float()

    def _forward_conv(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        if "conv2" in self.attention_stages:
            x = self.csja_conv2(x)
        x = self.relu3(self.conv3(x))
        return x

    def forward(self, x):
        x = self._prepare_stacked_input(x)
        x = self._forward_conv(x)
        return self.gap(x).squeeze(-1).squeeze(-1)
