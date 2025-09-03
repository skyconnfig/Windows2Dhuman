#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
时序一致性损失函数 - 数字人唇形优化
确保连续帧之间的唇形平滑过渡，减少闪烁和不连续现象
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple

class TemporalConsistencyLoss(nn.Module):
    """
    时序一致性损失函数
    
    主要功能：
    1. 帧间差异损失：确保相邻帧之间的变化平滑
    2. 光流一致性损失：基于光流的运动一致性
    3. 嘴部区域时序损失：专门针对嘴部区域的时序平滑
    4. 长期时序损失：确保较长时间窗口内的一致性
    """
    
    def __init__(self, 
                 lambda_frame_diff=5.0,
                 lambda_optical_flow=3.0, 
                 lambda_mouth_temporal=8.0,
                 lambda_long_term=2.0,
                 window_size=5):
        super(TemporalConsistencyLoss, self).__init__()
        
        self.lambda_frame_diff = lambda_frame_diff
        self.lambda_optical_flow = lambda_optical_flow
        self.lambda_mouth_temporal = lambda_mouth_temporal
        self.lambda_long_term = lambda_long_term
        self.window_size = window_size
        
        # 损失函数
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        
        # Sobel算子用于边缘检测
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
    
    def compute_optical_flow_loss(self, pred_frames: torch.Tensor, target_frames: torch.Tensor) -> torch.Tensor:
        """
        计算基于光流的一致性损失
        
        Args:
            pred_frames: 预测帧序列 [B, T, C, H, W]
            target_frames: 目标帧序列 [B, T, C, H, W]
        
        Returns:
            optical_flow_loss: 光流一致性损失
        """
        if pred_frames.size(1) < 2:
            return torch.tensor(0.0, device=pred_frames.device)
        
        # 计算相邻帧之间的差异（简化的光流近似）
        pred_flow = pred_frames[:, 1:] - pred_frames[:, :-1]  # [B, T-1, C, H, W]
        target_flow = target_frames[:, 1:] - target_frames[:, :-1]  # [B, T-1, C, H, W]
        
        # 光流一致性损失
        flow_loss = self.l1_loss(pred_flow, target_flow)
        
        return flow_loss
    
    def compute_frame_difference_loss(self, pred_frames: torch.Tensor, target_frames: torch.Tensor) -> torch.Tensor:
        """
        计算帧间差异损失
        
        Args:
            pred_frames: 预测帧序列 [B, T, C, H, W]
            target_frames: 目标帧序列 [B, T, C, H, W]
        
        Returns:
            frame_diff_loss: 帧间差异损失
        """
        if pred_frames.size(1) < 2:
            return torch.tensor(0.0, device=pred_frames.device)
        
        # 计算相邻帧之间的差异
        pred_diff = torch.abs(pred_frames[:, 1:] - pred_frames[:, :-1])
        target_diff = torch.abs(target_frames[:, 1:] - target_frames[:, :-1])
        
        # 帧间差异损失
        diff_loss = self.l1_loss(pred_diff, target_diff)
        
        return diff_loss
    
    def compute_mouth_temporal_loss(self, pred_frames: torch.Tensor, target_frames: torch.Tensor, 
                                   mouth_masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算嘴部区域的时序一致性损失
        
        Args:
            pred_frames: 预测帧序列 [B, T, C, H, W]
            target_frames: 目标帧序列 [B, T, C, H, W]
            mouth_masks: 嘴部掩码 [B, T, 1, H, W]，可选
        
        Returns:
            mouth_temporal_loss: 嘴部时序损失
        """
        if pred_frames.size(1) < 2:
            return torch.tensor(0.0, device=pred_frames.device)
        
        # 如果没有提供嘴部掩码，创建默认掩码（假设嘴部在图像下半部分中央）
        if mouth_masks is None:
            B, T, C, H, W = pred_frames.shape
            mouth_masks = torch.zeros(B, T, 1, H, W, device=pred_frames.device)
            # 嘴部区域大致在图像的下半部分中央
            mouth_y_start = int(H * 0.6)
            mouth_y_end = int(H * 0.9)
            mouth_x_start = int(W * 0.3)
            mouth_x_end = int(W * 0.7)
            mouth_masks[:, :, :, mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end] = 1.0
        
        # 应用嘴部掩码
        mouth_masks_expanded = mouth_masks.expand_as(pred_frames)
        pred_mouth = pred_frames * mouth_masks_expanded
        target_mouth = target_frames * mouth_masks_expanded
        
        # 计算嘴部区域的时序变化
        pred_mouth_diff = pred_mouth[:, 1:] - pred_mouth[:, :-1]
        target_mouth_diff = target_mouth[:, 1:] - target_mouth[:, :-1]
        
        # 嘴部时序损失
        mouth_temporal_loss = self.l2_loss(pred_mouth_diff, target_mouth_diff)
        
        # 添加嘴部区域的边缘时序一致性
        pred_mouth_gray = torch.mean(pred_mouth, dim=2, keepdim=True)  # 转为灰度
        target_mouth_gray = torch.mean(target_mouth, dim=2, keepdim=True)
        
        # 计算边缘
        pred_edges = self.compute_edge_map(pred_mouth_gray.view(-1, 1, H, W)).view(B, T, 1, H, W)
        target_edges = self.compute_edge_map(target_mouth_gray.view(-1, 1, H, W)).view(B, T, 1, H, W)
        
        # 边缘时序损失
        pred_edge_diff = pred_edges[:, 1:] - pred_edges[:, :-1]
        target_edge_diff = target_edges[:, 1:] - target_edges[:, :-1]
        edge_temporal_loss = self.l1_loss(pred_edge_diff, target_edge_diff)
        
        return mouth_temporal_loss + edge_temporal_loss * 0.5
    
    def compute_edge_map(self, image: torch.Tensor) -> torch.Tensor:
        """
        计算图像的边缘图
        
        Args:
            image: 输入图像 [B, C, H, W]
        
        Returns:
            edges: 边缘图 [B, C, H, W]
        """
        # 如果是多通道图像，转换为灰度
        if image.size(1) > 1:
            image = torch.mean(image, dim=1, keepdim=True)
        
        # 应用Sobel算子
        grad_x = F.conv2d(image, self.sobel_x, padding=1)
        grad_y = F.conv2d(image, self.sobel_y, padding=1)
        
        # 计算梯度幅值
        edges = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
        return edges
    
    def compute_long_term_consistency_loss(self, pred_frames: torch.Tensor, target_frames: torch.Tensor) -> torch.Tensor:
        """
        计算长期时序一致性损失
        
        Args:
            pred_frames: 预测帧序列 [B, T, C, H, W]
            target_frames: 目标帧序列 [B, T, C, H, W]
        
        Returns:
            long_term_loss: 长期时序一致性损失
        """
        T = pred_frames.size(1)
        if T < self.window_size:
            return torch.tensor(0.0, device=pred_frames.device)
        
        long_term_loss = 0.0
        num_windows = T - self.window_size + 1
        
        for i in range(num_windows):
            # 提取时间窗口
            pred_window = pred_frames[:, i:i+self.window_size]  # [B, window_size, C, H, W]
            target_window = target_frames[:, i:i+self.window_size]
            
            # 计算窗口内的平均帧
            pred_mean = torch.mean(pred_window, dim=1)  # [B, C, H, W]
            target_mean = torch.mean(target_window, dim=1)
            
            # 计算每帧与平均帧的差异
            pred_var = torch.mean((pred_window - pred_mean.unsqueeze(1)) ** 2)
            target_var = torch.mean((target_window - target_mean.unsqueeze(1)) ** 2)
            
            # 方差一致性损失
            long_term_loss += self.l1_loss(pred_var, target_var)
        
        return long_term_loss / num_windows
    
    def forward(self, pred_frames: torch.Tensor, target_frames: torch.Tensor, 
                mouth_masks: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, dict]:
        """
        前向传播计算时序一致性损失
        
        Args:
            pred_frames: 预测帧序列 [B, T, C, H, W] 或 [B, C, H, W]（单帧）
            target_frames: 目标帧序列 [B, T, C, H, W] 或 [B, C, H, W]（单帧）
            mouth_masks: 嘴部掩码 [B, T, 1, H, W]，可选
        
        Returns:
            total_loss: 总时序一致性损失
            loss_dict: 各项损失的字典
        """
        # 如果输入是单帧，添加时间维度
        if pred_frames.dim() == 4:
            pred_frames = pred_frames.unsqueeze(1)  # [B, 1, C, H, W]
            target_frames = target_frames.unsqueeze(1)
            if mouth_masks is not None:
                mouth_masks = mouth_masks.unsqueeze(1)
        
        loss_dict = {}
        
        # 1. 帧间差异损失
        frame_diff_loss = self.compute_frame_difference_loss(pred_frames, target_frames)
        loss_dict['frame_diff_loss'] = frame_diff_loss
        
        # 2. 光流一致性损失
        optical_flow_loss = self.compute_optical_flow_loss(pred_frames, target_frames)
        loss_dict['optical_flow_loss'] = optical_flow_loss
        
        # 3. 嘴部时序损失
        mouth_temporal_loss = self.compute_mouth_temporal_loss(pred_frames, target_frames, mouth_masks)
        loss_dict['mouth_temporal_loss'] = mouth_temporal_loss
        
        # 4. 长期时序一致性损失
        long_term_loss = self.compute_long_term_consistency_loss(pred_frames, target_frames)
        loss_dict['long_term_loss'] = long_term_loss
        
        # 5. 计算总损失
        total_loss = (self.lambda_frame_diff * frame_diff_loss +
                     self.lambda_optical_flow * optical_flow_loss +
                     self.lambda_mouth_temporal * mouth_temporal_loss +
                     self.lambda_long_term * long_term_loss)
        
        loss_dict['temporal_total_loss'] = total_loss
        
        return total_loss, loss_dict

class TemporalFrameBuffer:
    """
    时序帧缓冲器
    用于在训练过程中维护帧序列，支持时序一致性损失计算
    """
    
    def __init__(self, buffer_size=5):
        self.buffer_size = buffer_size
        self.pred_buffer = []
        self.target_buffer = []
        self.mask_buffer = []
    
    def update(self, pred_frame: torch.Tensor, target_frame: torch.Tensor, 
               mouth_mask: Optional[torch.Tensor] = None):
        """
        更新帧缓冲器
        
        Args:
            pred_frame: 预测帧 [B, C, H, W]
            target_frame: 目标帧 [B, C, H, W]
            mouth_mask: 嘴部掩码 [B, 1, H, W]，可选
        """
        self.pred_buffer.append(pred_frame.detach())
        self.target_buffer.append(target_frame.detach())
        
        if mouth_mask is not None:
            self.mask_buffer.append(mouth_mask.detach())
        else:
            self.mask_buffer.append(None)
        
        # 保持缓冲器大小
        if len(self.pred_buffer) > self.buffer_size:
            self.pred_buffer.pop(0)
            self.target_buffer.pop(0)
            self.mask_buffer.pop(0)
    
    def get_sequence(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        获取当前帧序列
        
        Returns:
            pred_sequence: 预测帧序列 [B, T, C, H, W]
            target_sequence: 目标帧序列 [B, T, C, H, W]
            mask_sequence: 掩码序列 [B, T, 1, H, W]
        """
        if len(self.pred_buffer) < 2:
            return None, None, None
        
        pred_sequence = torch.stack(self.pred_buffer, dim=1)  # [B, T, C, H, W]
        target_sequence = torch.stack(self.target_buffer, dim=1)
        
        # 处理掩码序列
        if any(mask is not None for mask in self.mask_buffer):
            # 如果有掩码，创建掩码序列
            mask_sequence = []
            for mask in self.mask_buffer:
                if mask is not None:
                    mask_sequence.append(mask)
                else:
                    # 创建默认掩码
                    B, C, H, W = pred_sequence.shape[0], 1, pred_sequence.shape[3], pred_sequence.shape[4]
                    default_mask = torch.zeros(B, C, H, W, device=pred_sequence.device)
                    mask_sequence.append(default_mask)
            mask_sequence = torch.stack(mask_sequence, dim=1)  # [B, T, 1, H, W]
        else:
            mask_sequence = None
        
        return pred_sequence, target_sequence, mask_sequence
    
    def clear(self):
        """清空缓冲器"""
        self.pred_buffer.clear()
        self.target_buffer.clear()
        self.mask_buffer.clear()

# 使用示例和配置说明
"""
时序一致性损失函数使用说明：

1. 基础使用（单帧训练时）：
   temporal_loss_fn = TemporalConsistencyLoss(
       lambda_frame_diff=5.0,
       lambda_optical_flow=3.0,
       lambda_mouth_temporal=8.0,
       lambda_long_term=2.0
   )
   
   # 在训练循环中
   frame_buffer = TemporalFrameBuffer(buffer_size=5)
   
   for batch in dataloader:
       pred = model(batch)
       target = batch['target']
       
       # 更新帧缓冲器
       frame_buffer.update(pred, target)
       
       # 计算时序损失
       pred_seq, target_seq, mask_seq = frame_buffer.get_sequence()
       if pred_seq is not None:
           temporal_loss, temporal_loss_dict = temporal_loss_fn(pred_seq, target_seq, mask_seq)
       else:
           temporal_loss = 0

2. 序列训练使用：
   # 如果直接有帧序列数据
   pred_frames = model(batch_sequence)  # [B, T, C, H, W]
   target_frames = batch_sequence['target']  # [B, T, C, H, W]
   
   temporal_loss, loss_dict = temporal_loss_fn(pred_frames, target_frames)

3. 与其他损失函数结合：
   total_loss = pixel_loss + perceptual_loss + 0.3 * temporal_loss

特性说明：
- 帧间差异损失：确保相邻帧之间的变化平滑自然
- 光流一致性损失：基于运动信息的时序约束
- 嘴部时序损失：专门针对嘴部区域的时序平滑，包括边缘一致性
- 长期时序损失：确保较长时间窗口内的整体一致性
- 帧缓冲器：支持单帧训练时的时序损失计算

推荐权重配置：
- lambda_frame_diff: 5.0 (帧间差异损失权重)
- lambda_optical_flow: 3.0 (光流一致性损失权重)
- lambda_mouth_temporal: 8.0 (嘴部时序损失权重，最重要)
- lambda_long_term: 2.0 (长期时序损失权重)
"""