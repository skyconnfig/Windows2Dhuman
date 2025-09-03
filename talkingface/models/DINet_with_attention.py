#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DINet模型集成注意力机制版本
在原有DINet_five_Ref基础上集成多种注意力机制，提升唇形同步精度
"""

import torch
from torch import nn
import torch.nn.functional as F
import math
import cv2
import numpy as np
from torch.nn import BatchNorm2d
from torch.nn import BatchNorm1d
from typing import Optional, Tuple, Dict

# 导入原有DINet组件
from .DINet import (
    make_coordinate_grid_3d, ResBlock1d, ResBlock2d, UpBlock2d, 
    DownBlock1d, DownBlock2d, SameBlock1d, SameBlock2d, AdaAT
)

# 导入注意力机制模块
from .attention_modules import (
    SpatialAttentionModule, MouthFocusedAttention, 
    MultiScaleAttention, AdaptiveAttentionFusion
)

class AttentionEnhancedBlock(nn.Module):
    """
    注意力增强的基础块
    在原有ResBlock基础上添加注意力机制
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 kernel_size: int = 3, padding: int = 1, 
                 use_attention: bool = True, attention_type: str = 'spatial'):
        super(AttentionEnhancedBlock, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.use_attention = use_attention
        
        # 原有ResBlock组件
        self.conv1 = nn.Conv2d(in_channels=in_features, out_channels=in_features, 
                              kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_channels=in_features, out_channels=out_features, 
                              kernel_size=kernel_size, padding=padding)
        
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features, out_features, 1)
        
        self.norm1 = BatchNorm2d(in_features)
        self.norm2 = BatchNorm2d(in_features)
        self.relu = nn.ReLU()
        
        # 注意力机制
        if self.use_attention:
            if attention_type == 'spatial':
                self.attention = SpatialAttentionModule(out_features)
            elif attention_type == 'mouth':
                self.attention = MouthFocusedAttention(out_features)
            elif attention_type == 'multiscale':
                self.attention = MultiScaleAttention(out_features)
            elif attention_type == 'adaptive':
                self.attention = AdaptiveAttentionFusion(out_features)
            else:
                self.attention = SpatialAttentionModule(out_features)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
        
        Returns:
            out: 输出特征图 [B, C, H, W]
            attention_map: 注意力图（如果使用注意力）
        """
        # 原有ResBlock前向传播
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        
        # 残差连接
        if self.in_features != self.out_features:
            out += self.channel_conv(x)
        else:
            out += x
        
        # 应用注意力机制
        attention_map = None
        if self.use_attention:
            if hasattr(self.attention, 'forward'):
                if isinstance(self.attention, AdaptiveAttentionFusion):
                    out, attention_info = self.attention(out)
                    attention_map = attention_info.get('final_attention', None)
                elif isinstance(self.attention, MouthFocusedAttention):
                    out, attention_map, _ = self.attention(out)
                else:
                    out, attention_map = self.attention(out)
        
        return out, attention_map

class DINet_five_Ref_with_Attention(nn.Module):
    """
    集成注意力机制的DINet_five_Ref模型
    在关键位置添加注意力模块，专注于嘴部区域优化
    """
    
    def __init__(self, source_channel: int, ref_channel: int, 
                 cuda: bool = True, attention_config: Optional[Dict] = None):
        super(DINet_five_Ref_with_Attention, self).__init__()
        
        # 默认注意力配置
        default_attention_config = {
            'use_source_attention': True,
            'use_ref_attention': True,
            'use_merge_attention': True,
            'source_attention_type': 'mouth',  # 源图像使用嘴部专注注意力
            'ref_attention_type': 'spatial',   # 参考图像使用空间注意力
            'merge_attention_type': 'adaptive' # 融合特征使用自适应注意力
        }
        
        if attention_config is None:
            attention_config = default_attention_config
        else:
            default_attention_config.update(attention_config)
            attention_config = default_attention_config
        
        self.attention_config = attention_config
        
        # 源图像编码器（添加注意力）
        self.source_in_conv = nn.Sequential(
            SameBlock2d(source_channel, 32, kernel_size=7, padding=3),
            DownBlock2d(32, 64, kernel_size=3, padding=1),
            DownBlock2d(64, 128, kernel_size=3, padding=1)
        )
        
        # 源图像注意力模块
        if attention_config['use_source_attention']:
            self.source_attention = self._create_attention_module(
                128, attention_config['source_attention_type']
            )
        
        # 参考图像编码器（添加注意力）
        self.ref_in_conv = nn.Sequential(
            SameBlock2d(ref_channel, 64, kernel_size=7, padding=3),
            DownBlock2d(64, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 256, kernel_size=3, padding=1),
        )
        
        # 参考图像注意力模块
        if attention_config['use_ref_attention']:
            self.ref_attention = self._create_attention_module(
                256, attention_config['ref_attention_type']
            )
        
        # 变换卷积层
        self.trans_conv = nn.Sequential(
            # 20 →10
            SameBlock2d(384, 128, kernel_size=3, padding=1),
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 10 →5
            SameBlock2d(128, 128, kernel_size=7, padding=3),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 5 →3
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
            # 3 →2
            SameBlock2d(128, 128, kernel_size=3, padding=1),
            DownBlock2d(128, 128, kernel_size=3, padding=1),
        )
        
        # 外观卷积层（使用注意力增强块）
        appearance_conv_list = []
        for i in range(2):
            if i == 0:  # 第一层使用嘴部专注注意力
                appearance_conv_list.append(
                    nn.Sequential(
                        AttentionEnhancedBlock(256, 256, 3, 1, 
                                             use_attention=True, attention_type='mouth'),
                        AttentionEnhancedBlock(256, 256, 3, 1, 
                                             use_attention=True, attention_type='spatial'),
                    )
                )
            else:  # 第二层使用多尺度注意力
                appearance_conv_list.append(
                    nn.Sequential(
                        AttentionEnhancedBlock(256, 256, 3, 1, 
                                             use_attention=True, attention_type='multiscale'),
                        AttentionEnhancedBlock(256, 256, 3, 1, 
                                             use_attention=True, attention_type='adaptive'),
                    )
                )
        
        self.appearance_conv_list = nn.ModuleList(appearance_conv_list)
        
        # AdaAT模块
        self.adaAT = AdaAT(128, 256, cuda)
        
        # 融合特征注意力模块
        if attention_config['use_merge_attention']:
            self.merge_attention = self._create_attention_module(
                384, attention_config['merge_attention_type']
            )
        
        # 输出卷积层
        self.out_conv = nn.Sequential(
            SameBlock2d(384, 128, kernel_size=3, padding=1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            ResBlock2d(128, 128, 3, 1),
            UpBlock2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # 全局池化层
        self.global_avg2d = nn.AdaptiveAvgPool2d(1)
        self.global_avg1d = nn.AdaptiveAvgPool1d(1)
        
        # 存储注意力信息
        self.attention_maps = {}
    
    def _create_attention_module(self, in_channels: int, attention_type: str):
        """
        创建指定类型的注意力模块
        
        Args:
            in_channels: 输入通道数
            attention_type: 注意力类型
        
        Returns:
            attention_module: 注意力模块
        """
        if attention_type == 'spatial':
            return SpatialAttentionModule(in_channels)
        elif attention_type == 'mouth':
            return MouthFocusedAttention(in_channels)
        elif attention_type == 'multiscale':
            return MultiScaleAttention(in_channels)
        elif attention_type == 'adaptive':
            return AdaptiveAttentionFusion(in_channels)
        else:
            return SpatialAttentionModule(in_channels)
    
    def ref_input(self, ref_img: torch.Tensor) -> None:
        """
        处理参考图像输入
        
        Args:
            ref_img: 参考图像 [B, C, H, W]
        """
        # 参考图像编码
        self.ref_in_feature = self.ref_in_conv(ref_img)
        
        # 应用参考图像注意力
        if self.attention_config['use_ref_attention']:
            if isinstance(self.ref_attention, AdaptiveAttentionFusion):
                self.ref_in_feature, attention_info = self.ref_attention(self.ref_in_feature)
                self.attention_maps['ref_attention'] = attention_info.get('final_attention', None)
            elif isinstance(self.ref_attention, MouthFocusedAttention):
                self.ref_in_feature, attention_map, _ = self.ref_attention(self.ref_in_feature)
                self.attention_maps['ref_attention'] = attention_map
            else:
                self.ref_in_feature, attention_map = self.ref_attention(self.ref_in_feature)
                self.attention_maps['ref_attention'] = attention_map
        
        # 使用AdaAT进行空间变形
        self.ref_trans_feature0, attention_map_0 = self.appearance_conv_list[0][0](self.ref_in_feature)
        self.ref_trans_feature0, attention_map_1 = self.appearance_conv_list[0][1](self.ref_trans_feature0)
        
        # 存储注意力图
        self.attention_maps['appearance_0_0'] = attention_map_0
        self.attention_maps['appearance_0_1'] = attention_map_1
    
    def interface(self, source_img: torch.Tensor, source_prompt: torch.Tensor) -> torch.Tensor:
        """
        主要推理接口
        
        Args:
            source_img: 源图像 [B, C, H, W]
            source_prompt: 源提示 [B, C, H, W]
        
        Returns:
            out: 输出图像 [B, C, H, W]
        """
        # 拼接源图像和提示
        self.source_img = torch.cat([source_img, source_prompt], dim=1)
        
        # 源图像编码
        source_in_feature = self.source_in_conv(self.source_img)
        
        # 应用源图像注意力
        if self.attention_config['use_source_attention']:
            if isinstance(self.source_attention, AdaptiveAttentionFusion):
                source_in_feature, attention_info = self.source_attention(source_in_feature)
                self.attention_maps['source_attention'] = attention_info.get('final_attention', None)
            elif isinstance(self.source_attention, MouthFocusedAttention):
                source_in_feature, attention_map, _ = self.source_attention(source_in_feature)
                self.attention_maps['source_attention'] = attention_map
            else:
                source_in_feature, attention_map = self.source_attention(source_in_feature)
                self.attention_maps['source_attention'] = attention_map
        
        # 对齐编码器
        img_para = self.trans_conv(torch.cat([source_in_feature, self.ref_in_feature], 1))
        img_para = self.global_avg2d(img_para).squeeze(3).squeeze(2)
        
        # 变换参数
        trans_para = img_para
        
        # AdaAT变换
        ref_trans_feature = self.adaAT(self.ref_trans_feature0, trans_para)
        
        # 第二层外观卷积（带注意力）
        ref_trans_feature, attention_map_2 = self.appearance_conv_list[1][0](ref_trans_feature)
        ref_trans_feature, attention_map_3 = self.appearance_conv_list[1][1](ref_trans_feature)
        
        # 存储注意力图
        self.attention_maps['appearance_1_0'] = attention_map_2
        self.attention_maps['appearance_1_1'] = attention_map_3
        
        # 特征融合
        merge_feature = torch.cat([source_in_feature, ref_trans_feature], 1)
        
        # 应用融合特征注意力
        if self.attention_config['use_merge_attention']:
            if isinstance(self.merge_attention, AdaptiveAttentionFusion):
                merge_feature, attention_info = self.merge_attention(merge_feature)
                self.attention_maps['merge_attention'] = attention_info.get('final_attention', None)
            elif isinstance(self.merge_attention, MouthFocusedAttention):
                merge_feature, attention_map, _ = self.merge_attention(merge_feature)
                self.attention_maps['merge_attention'] = attention_map
            else:
                merge_feature, attention_map = self.merge_attention(merge_feature)
                self.attention_maps['merge_attention'] = attention_map
        
        # 特征解码
        out = self.out_conv(merge_feature)
        
        return out
    
    def forward(self, source_img: torch.Tensor, source_prompt: torch.Tensor, 
                ref_img: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        前向传播
        
        Args:
            source_img: 源图像 [B, C, H, W]
            source_prompt: 源提示 [B, C, H, W]
            ref_img: 参考图像 [B, C, H, W]
        
        Returns:
            out: 输出图像 [B, C, H, W]
            attention_maps: 注意力图字典
        """
        # 清空之前的注意力图
        self.attention_maps = {}
        
        # 处理参考图像
        self.ref_input(ref_img)
        
        # 主要推理
        out = self.interface(source_img, source_prompt)
        
        return out, self.attention_maps
    
    def get_attention_visualization(self, attention_name: str = 'merge_attention') -> Optional[torch.Tensor]:
        """
        获取指定注意力的可视化图
        
        Args:
            attention_name: 注意力名称
        
        Returns:
            attention_vis: 注意力可视化图 [B, 1, H, W]
        """
        if attention_name in self.attention_maps:
            attention_map = self.attention_maps[attention_name]
            if attention_map is not None:
                # 归一化到0-1范围
                attention_vis = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)
                return attention_vis
        return None
    
    def set_attention_config(self, new_config: Dict) -> None:
        """
        更新注意力配置
        
        Args:
            new_config: 新的注意力配置
        """
        self.attention_config.update(new_config)
        print(f"注意力配置已更新: {self.attention_config}")

# 使用示例和配置说明
"""
DINet集成注意力机制使用说明：

1. 基础使用（默认配置）：
   model = DINet_five_Ref_with_Attention(source_channel=6, ref_channel=30)
   output, attention_maps = model(source_img, source_prompt, ref_img)

2. 自定义注意力配置：
   attention_config = {
       'use_source_attention': True,
       'use_ref_attention': True,
       'use_merge_attention': True,
       'source_attention_type': 'adaptive',  # 可选: spatial, mouth, multiscale, adaptive
       'ref_attention_type': 'mouth',
       'merge_attention_type': 'adaptive'
   }
   model = DINet_five_Ref_with_Attention(6, 30, attention_config=attention_config)

3. 获取注意力可视化：
   attention_vis = model.get_attention_visualization('merge_attention')
   # 可用的注意力名称: source_attention, ref_attention, merge_attention, 
   #                  appearance_0_0, appearance_0_1, appearance_1_0, appearance_1_1

4. 动态调整注意力配置：
   model.set_attention_config({'source_attention_type': 'mouth'})

5. 训练时使用：
   model.train()
   optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
   
   for batch in dataloader:
       source_img, source_prompt, ref_img, target = batch
       output, attention_maps = model(source_img, source_prompt, ref_img)
       
       # 可以使用注意力图进行额外的损失计算
       attention_loss = compute_attention_loss(attention_maps, target_attention)
       main_loss = criterion(output, target)
       total_loss = main_loss + 0.1 * attention_loss
       
       optimizer.zero_grad()
       total_loss.backward()
       optimizer.step()

特性说明：
- 多层次注意力：在源图像、参考图像和融合特征上分别应用注意力
- 可配置性：支持不同类型的注意力机制组合
- 可视化支持：提供注意力图的可视化接口
- 向后兼容：保持与原DINet相同的接口
- 性能优化：注意力计算经过优化，对推理速度影响最小

推荐配置：
- 嘴部专注场景：source_attention_type='mouth', merge_attention_type='adaptive'
- 高质量渲染：所有注意力类型设为'adaptive'
- 实时应用：使用'spatial'注意力以降低计算量
- 训练阶段：使用完整的注意力配置以获得最佳效果
"""