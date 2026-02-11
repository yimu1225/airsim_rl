import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class D_LGMD(nn.Module):
    """
    D-LGMD 通用版 (Generic Implementation)
    支持任意长度的时间序列输入 (T >= 4)。
    利用并行计算一次性输出有效时间步的所有 S 层特征图。
    """
    def __init__(self, input_height=224, input_width=224, device='cuda'):
        super().__init__()
        self.device = device
        
        # --- 参数设置 (基于论文 Set 7) ---
        self.sigma_e = 1.5
        self.sigma_i = 5.0
        self.alpha_inhib = 1.0 # 抑制权重
        self.tau_params = {'alpha': -0.1, 'beta': 0.5, 'lambda': 0.7} 

        # --- 预计算卷积核 ---
        self.kernel_e = self._create_gaussian_kernel(self.sigma_e).to(device)
        self.kernel_i = self._create_gaussian_kernel(self.sigma_i).to(device)
        
        # 卷积层 (使用 Group Conv 实现时间维度的并行处理)
        self.conv_e = self._make_fixed_conv(self.kernel_e)
        self.conv_i = self._make_fixed_conv(self.kernel_i)

        # --- 预计算延时掩码 ---
        # 扩展维度以支持广播: (1, 1, 1, H, W) -> (Batch, Time, Channel, H, W)
        mask_1, mask_2 = self._create_latency_masks(input_height, input_width)
        self.mask_delay_1 = mask_1.to(device)
        self.mask_delay_2 = mask_2.to(device)

    def _make_fixed_conv(self, kernel):
        k_size = kernel.shape[2]
        conv = nn.Conv2d(1, 1, kernel_size=k_size, padding=k_size//2, bias=False)
        conv.weight.data = kernel
        conv.weight.requires_grad = False
        return conv

    def _create_gaussian_kernel(self, sigma):
        k_size = math.ceil(sigma * 6)
        if k_size % 2 == 0: k_size += 1
        center = k_size // 2
        x = torch.arange(k_size, dtype=torch.float32) - center
        y = torch.arange(k_size, dtype=torch.float32) - center
        y, x = torch.meshgrid(y, x, indexing='ij')
        kernel = (1 / (2 * math.pi * sigma**2)) * torch.exp(-(x**2 + y**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def _create_latency_masks(self, H, W):
        x = torch.linspace(-3, 3, W)
        y = torch.linspace(-3, 3, H)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        r2 = grid_x**2 + grid_y**2
        tau_map = self.tau_params['alpha'] + 1.0 / (self.tau_params['beta'] + torch.exp(- (self.tau_params['lambda']**2) * r2))
        tau_int = torch.clamp(torch.round(tau_map).long(), 1, 2)
        
        # 增加维度 (1, 1, 1, H, W) 用于后续与 (B, T, 1, H, W) 进行广播乘法
        mask_1 = (tau_int == 1).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask_2 = (tau_int == 2).float().unsqueeze(0).unsqueeze(0).unsqueeze(0)
        return mask_1, mask_2

    def forward(self, frames):
        """
        输入: 
            frames: (B, T, C, H, W) 或 (B, T, H, W)
            要求 T >= 4
        输出:
            S: (B, T-3, 1, H, W) 
            返回有效时间步序列。例如输入10帧，输出后7帧的特征图。
        """
        # --- 1. 格式统一 (转灰度) ---
        if frames.dim() == 5:
            B, T, C, H, W = frames.shape
            if C == 3:
                frames = 0.299 * frames[:, :, 0] + 0.587 * frames[:, :, 1] + 0.114 * frames[:, :, 2]
            elif C == 1:
                frames = frames.squeeze(2)
        else:
            B, T, H, W = frames.shape
            
        if T < 4:
            raise ValueError(f"D-LGMD 需要至少4帧来计算历史延时，当前输入仅 {T} 帧。")

        # --- 2. P Layer (批量差分) ---
        # frames: [F0, F1, F2, ..., FT-1]
        # P_seq:  [P0, P1, P2, ..., PT-2] (长度 T-1)
        # P0=|F1-F0|, P1=|F2-F1| ...
        P_seq = torch.abs(frames[:, 1:] - frames[:, :-1]).unsqueeze(2) # (B, T-1, 1, H, W)
        
        # --- 3. 时间步对齐 (Slicing) ---
        # 我们需要计算 S(t) 对应 P_current(t)
        # 依赖关系: S(t) 需要 P(t) [兴奋], P(t-1) [延时1], P(t-2) [延时2]
        # 因此，有效的 P 序列起始点必须是 index 2 (即 P2)
        
        # P_curr (兴奋源): 从 P2 开始到最后 -> [P2, P3, ..., PT-2]
        P_curr = P_seq[:, 2:] 
        
        # P_delay1 (抑制源1): 从 P1 开始到倒数第1个 -> [P1, P2, ..., PT-3]
        P_delay1 = P_seq[:, 1:-1]
        
        # P_delay2 (抑制源2): 从 P0 开始到倒数第2个 -> [P0, P1, ..., PT-4]
        P_delay2 = P_seq[:, :-2]
        
        # 此时三个张量的形状都是: (B, T_valid, 1, H, W)，其中 T_valid = T - 3

        # --- 4. E Layer (并行卷积) ---
        # 为了高效计算，将 Batch 和 Time 维度合并进行 2D 卷积
        # (B, T_valid, 1, H, W) -> (B*T_valid, 1, H, W)
        T_valid = P_curr.shape[1]
        P_curr_flat = P_curr.reshape(-1, 1, H, W)
        
        E_flat = self.conv_e(P_curr_flat)
        E = E_flat.reshape(B, T_valid, 1, H, W)

        # --- 5. I Layer (分布式混合 + 并行卷积) ---
        # 先混合 (Mixed P)
        # 掩码 mask_1, mask_2 形状为 (1, 1, 1, H, W)，会自动广播到 Batch 和 Time
        P_mixed = P_delay1 * self.mask_delay_1 + P_delay2 * self.mask_delay_2
        
        # 再卷积
        P_mixed_flat = P_mixed.reshape(-1, 1, H, W)
        I_flat = self.conv_i(P_mixed_flat)
        I = I_flat.reshape(B, T_valid, 1, H, W)

        # --- 6. S Layer ---
        S_raw = E - self.alpha_inhib * I
        S = F.relu(S_raw)
        
        return S
