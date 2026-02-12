#!/usr/bin/env python3
import torch
import torch.nn as nn

# 模拟CNN架构来计算输出维度
class CNN(nn.Module):
    def __init__(self, input_height, input_width, input_channels=1):
        super().__init__()

        # 特征扩展因子
        f1 = 4
        f2 = 8
        f3 = 16
        f4 = 32
        f5 = 48

        self.net = nn.Sequential(
            # 第一层: Conv2d + MaxPool(2x2) -> 输入高宽减半
            nn.Conv2d(input_channels, f1, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
           
            # 第二层: Conv2d + MaxPool(2x2) -> 输入高宽再减半
            nn.Conv2d(f1, f2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三层: Conv2d + MaxPool(2x2) -> 输入高宽再减半
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四层
            nn.Conv2d(f3, f4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f4),
            nn.ReLU(inplace=True),
            
            # 第五层 1x1 conv
            nn.Conv2d(f4, f5, kernel_size=1),
            nn.BatchNorm2d(f5),
            nn.ReLU(inplace=True),
        )

        # 计算展平后的尺寸
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, input_height, input_width)
            dummy_output = self.net(dummy_input)
            self.n_flatten = dummy_output.view(1, -1).size(1)
            self.repr_dim = self.n_flatten

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # 添加通道维度如果需要
        x = self.net(x)
        x = x.view(x.size(0), -1)  # 直接展平
        return x

def calculate_dimensions():
    # 从配置中获取的参数
    STATE_RGB_H, STATE_RGB_W = 128, 128
