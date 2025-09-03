#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力机制模块 - 数字人唇形优化
让模型更专注于嘴部关键区域，提升唇形同步精度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple

class SpatialAttentionModule(nn.Module):
    """
    空间注意力模块
    专注于图像中的重要区域（特别是嘴部区域）
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 8):
        super(SpatialAttentionModule, self).__init__()
        
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        # 通道注意力分支
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 空间注意力分支
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
        # 嘴部区域增强分支
        self.mouth_enhancement = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, mouth_region_hint: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            mouth_region_hint: 嘴部区域提示 [B, 1, H, W]，可选
        
        Returns:
            attended_features: 注意力加权后的特征图 [B, C, H, W]
            attention_map: 注意力图 [B, 1, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 通道注意力
        channel_att = self.channel_attention(x)  # [B, C, 1, 1]
        x_channel = x * channel_att
        
        # 2. 空间注意力
        # 计算通道维度的平均值和最大值
        avg_pool = torch.mean(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        max_pool, _ = torch.max(x_channel, dim=1, keepdim=True)  # [B, 1, H, W]
        spatial_input = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, H, W]
        spatial_att = self.spatial_attention(spatial_input)  # [B, 1, H, W]
        
        # 3. 嘴部区域增强注意力
        mouth_att = self.mouth_enhancement(x_channel)  # [B, 1, H, W]
        
        # 4. 如果提供了嘴部区域提示，融合到注意力中
        if mouth_region_hint is not None:
            # 将嘴部提示与学习到的注意力结合
            mouth_att = mouth_att * 0.7 + mouth_region_hint * 0.3
        
        # 5. 组合所有注意力
        # 空间注意力和嘴部注意力的加权组合
        combined_spatial_att = spatial_att * 0.6 + mouth_att * 0.4
        
        # 最终的注意力图
        final_attention = combined_spatial_att
        
        # 应用注意力
        attended_features = x_channel * final_attention
        
        return attended_features, final_attention

class MouthFocusedAttention(nn.Module):
    """
    嘴部专注注意力模块
    专门设计用于增强嘴部区域的特征表示
    """
    
    def __init__(self, in_channels: int, mouth_region_size: Tuple[float, float] = (0.4, 0.3)):
        super(MouthFocusedAttention, self).__init__()
        
        self.in_channels = in_channels
        self.mouth_region_size = mouth_region_size  # (width_ratio, height_ratio)
        
        # 嘴部区域检测网络
        self.mouth_detector = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, in_channels // 4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1),
            nn.Sigmoid()
        )
        
        # 嘴部特征增强网络
        self.mouth_enhancer = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels // 4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )
        
        # 全局上下文网络
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )
    
    def create_mouth_prior(self, feature_shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """
        创建嘴部区域先验掩码
        
        Args:
            feature_shape: 特征图形状 (B, C, H, W)
        
        Returns:
            mouth_prior: 嘴部先验掩码 [B, 1, H, W]
        """
        B, C, H, W = feature_shape
        
        # 创建高斯分布的嘴部先验
        mouth_prior = torch.zeros(B, 1, H, W)
        
        # 嘴部中心位置（假设在图像下半部分中央）
        center_y = int(H * 0.75)  # 嘴部通常在脸部下方
        center_x = int(W * 0.5)   # 水平居中
        
        # 嘴部区域大小
        mouth_h = int(H * self.mouth_region_size[1])
        mouth_w = int(W * self.mouth_region_size[0])
        
        # 创建2D高斯分布
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, dtype=torch.float32),
            torch.arange(W, dtype=torch.float32),
            indexing='ij'
        )
        
        # 计算到嘴部中心的距离
        dist_y = (y_coords - center_y) / (mouth_h / 2)
        dist_x = (x_coords - center_x) / (mouth_w / 2)
        dist_sq = dist_y ** 2 + dist_x ** 2
        
        # 高斯权重
        gaussian_weight = torch.exp(-dist_sq / 2)
        
        # 扩展到batch维度
        mouth_prior = gaussian_weight.unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        return mouth_prior
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            enhanced_features: 增强后的特征图 [B, C, H, W]
            mouth_attention: 嘴部注意力图 [B, 1, H, W]
            mouth_features: 嘴部特征图 [B, C, H, W]
        """
        B, C, H, W = x.shape
        
        # 1. 检测嘴部区域
        mouth_attention = self.mouth_detector(x)  # [B, 1, H, W]
        
        # 2. 创建嘴部先验并融合
        mouth_prior = self.create_mouth_prior(x.shape).to(x.device)
        mouth_attention_combined = mouth_attention * 0.7 + mouth_prior * 0.3
        
        # 3. 提取嘴部特征
        mouth_features = self.mouth_enhancer(x * mouth_attention_combined)
        
        # 4. 全局上下文调制
        global_context = self.global_context(x)
        mouth_features = mouth_features * global_context
        
        # 5. 特征融合
        # 使用注意力权重融合原始特征和嘴部增强特征
        enhanced_features = x * (1 - mouth_attention_combined) + mouth_features * mouth_attention_combined
        
        return enhanced_features, mouth_attention_combined, mouth_features

class MultiScaleAttention(nn.Module):
    """
    多尺度注意力模块
    在不同尺度上捕获嘴部特征，提升细节表现
    """
    
    def __init__(self, in_channels: int, scales: list = [1, 2, 4]):
        super(MultiScaleAttention, self).__init__()
        
        self.in_channels = in_channels
        self.scales = scales
        
        # 多尺度特征提取分支
        self.scale_branches = nn.ModuleList()
        for scale in scales:
            branch = nn.Sequential(
                nn.Conv2d(in_channels, in_channels // len(scales), 3, 
                         padding=scale, dilation=scale, bias=False),
                nn.BatchNorm2d(in_channels // len(scales)),
                nn.ReLU(inplace=True)
            )
            self.scale_branches.append(branch)
        
        # 特征融合网络
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        # 注意力生成网络
        self.attention_gen = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 1, 1, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            attended_features: 多尺度注意力加权后的特征图 [B, C, H, W]
            attention_map: 多尺度注意力图 [B, 1, H, W]
        """
        # 1. 多尺度特征提取
        scale_features = []
        for branch in self.scale_branches:
            scale_feat = branch(x)
            scale_features.append(scale_feat)
        
        # 2. 特征拼接和融合
        multi_scale_features = torch.cat(scale_features, dim=1)  # [B, C, H, W]
        fused_features = self.fusion_conv(multi_scale_features)
        
        # 3. 生成注意力图
        attention_map = self.attention_gen(fused_features)  # [B, 1, H, W]
        
        # 4. 应用注意力
        attended_features = x * attention_map + fused_features * (1 - attention_map)
        
        return attended_features, attention_map

class AdaptiveAttentionFusion(nn.Module):
    """
    自适应注意力融合模块
    动态融合多种注意力机制的结果
    """
    
    def __init__(self, in_channels: int):
        super(AdaptiveAttentionFusion, self).__init__()
        
        self.in_channels = in_channels
        
        # 空间注意力
        self.spatial_attention = SpatialAttentionModule(in_channels)
        
        # 嘴部专注注意力
        self.mouth_attention = MouthFocusedAttention(in_channels)
        
        # 多尺度注意力
        self.multiscale_attention = MultiScaleAttention(in_channels)
        
        # 注意力权重学习网络
        self.attention_weights = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, 3, 1),  # 3个注意力分支的权重
            nn.Softmax(dim=1)
        )
        
        # 最终特征融合
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            final_features: 最终融合的特征图 [B, C, H, W]
            attention_info: 注意力信息字典
        """
        B, C, H, W = x.shape
        
        # 1. 应用不同的注意力机制
        spatial_features, spatial_att = self.spatial_attention(x)
        mouth_features, mouth_att, _ = self.mouth_attention(x)
        multiscale_features, multiscale_att = self.multiscale_attention(x)
        
        # 2. 学习注意力权重
        attention_weights = self.attention_weights(x)  # [B, 3, 1, 1]
        w_spatial = attention_weights[:, 0:1, :, :].expand(-1, -1, H, W)
        w_mouth = attention_weights[:, 1:2, :, :].expand(-1, -1, H, W)
        w_multiscale = attention_weights[:, 2:3, :, :].expand(-1, -1, H, W)
        
        # 3. 加权融合特征
        weighted_spatial = spatial_features * w_spatial.expand(-1, C, -1, -1)
        weighted_mouth = mouth_features * w_mouth.expand(-1, C, -1, -1)
        weighted_multiscale = multiscale_features * w_multiscale.expand(-1, C, -1, -1)
        
        # 4. 特征拼接和最终融合
        concatenated_features = torch.cat([weighted_spatial, weighted_mouth, weighted_multiscale], dim=1)
        final_features = self.final_fusion(concatenated_features)
        
        # 5. 组织注意力信息
        attention_info = {
            'spatial_attention': spatial_att,
            'mouth_attention': mouth_att,
            'multiscale_attention': multiscale_att,
            'attention_weights': attention_weights,
            'final_attention': w_spatial * spatial_att + w_mouth * mouth_att + w_multiscale * multiscale_att
        }
        
        return final_features, attention_info

# 使用示例和配置说明
"""
注意力机制模块使用说明：

1. 基础空间注意力使用：
   spatial_att = SpatialAttentionModule(in_channels=256)
   attended_features, attention_map = spatial_att(input_features)

2. 嘴部专注注意力使用：
   mouth_att = MouthFocusedAttention(in_channels=256, mouth_region_size=(0.4, 0.3))
   enhanced_features, mouth_attention, mouth_features = mouth_att(input_features)

3. 多尺度注意力使用：
   multiscale_att = MultiScaleAttention(in_channels=256, scales=[1, 2, 4])
   attended_features, attention_map = multiscale_att(input_features)

4. 自适应注意力融合使用（推荐）：
   adaptive_att = AdaptiveAttentionFusion(in_channels=256)
   final_features, attention_info = adaptive_att(input_features)
   
   # 可以访问各种注意力信息
   spatial_att = attention_info['spatial_attention']
   mouth_att = attention_info['mouth_attention']
   final_att = attention_info['final_attention']

5. 在DINet模型中集成：
   class DINetWithAttention(nn.Module):
       def __init__(self, ...):
           super().__init__()
           self.backbone = DINet_five_Ref(...)
           self.attention = AdaptiveAttentionFusion(in_channels=256)
       
       def forward(self, x):
           features = self.backbone.extract_features(x)
           attended_features, att_info = self.attention(features)
           output = self.backbone.decode(attended_features)
           return output, att_info

特性说明：
- 空间注意力：结合通道注意力和空间注意力，全面提升特征表示
- 嘴部专注注意力：专门针对嘴部区域设计，包含先验知识和学习机制
- 多尺度注意力：在不同感受野下捕获特征，提升细节表现
- 自适应融合：动态学习不同注意力机制的权重，自适应融合

推荐配置：
- in_channels: 根据网络层的通道数设置（通常128-512）
- mouth_region_size: (0.4, 0.3) 适合大多数人脸图像
- scales: [1, 2, 4] 提供良好的多尺度覆盖
- reduction_ratio: 8 在性能和计算量之间平衡
"""