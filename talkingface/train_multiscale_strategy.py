#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多尺度训练策略实现
支持从64px到160px的渐进式分辨率提升训练
结合注意力机制和时序一致性损失的完整训练方案
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import time
from collections import defaultdict
import logging

# 导入项目模块
from models.DINet_with_attention import DINet_five_Ref_with_Attention
from util.temporal_consistency_loss import TemporalConsistencyLoss, TemporalFrameBuffer
from models.attention_modules import (
    SpatialAttentionModule,
    MouthFocusedAttention, 
    MultiScaleAttention,
    AdaptiveAttentionFusion
)

class MultiScaleTrainingStrategy:
    """
    多尺度训练策略类
    实现渐进式分辨率提升和自适应训练调度
    """
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 多尺度配置
        self.scale_stages = [
            {'resolution': 64, 'epochs': 20, 'batch_size': 16, 'lr': 0.0001},
            {'resolution': 96, 'epochs': 15, 'batch_size': 12, 'lr': 0.00008},
            {'resolution': 128, 'epochs': 15, 'batch_size': 8, 'lr': 0.00006},
            {'resolution': 160, 'epochs': 10, 'batch_size': 6, 'lr': 0.00004}
        ]
        
        self.current_stage = 0
        self.total_epochs = 0
        
        # 初始化模型和损失函数
        self._initialize_model()
        self._initialize_losses()
        self._setup_logging()
        
    def _initialize_model(self):
        """初始化带注意力机制的DINet模型"""
        attention_config = {
            'spatial_attention': True,
            'mouth_focused_attention': True,
            'multi_scale_attention': True,
            'adaptive_fusion': True,
            'mouth_region_weight': 10.0
        }
        
        self.model = DINet_five_Ref_with_Attention(
            source_channel=3,
            ref_channel=15,
            attention_config=attention_config
        ).to(self.device)
        
        # 初始化优化器（将在每个阶段重新配置）
        self.optimizer = None
        self.scheduler = None
        
    def _initialize_losses(self):
        """初始化多种损失函数"""
        # 主要损失函数
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        
        # 时序一致性损失
        self.temporal_loss = TemporalConsistencyLoss(
            frame_diff_weight=8.0,
            optical_flow_weight=6.0,
            mouth_temporal_weight=10.0,
            long_term_weight=4.0
        ).to(self.device)
        
        # 时序帧缓冲
        self.frame_buffer = TemporalFrameBuffer(window_size=5)
        
        # 感知损失权重（随分辨率调整）
        self.perceptual_weights = {
            64: {'pixel': 1.0, 'perceptual': 0.1, 'temporal': 0.5},
            96: {'pixel': 1.0, 'perceptual': 0.15, 'temporal': 0.7},
            128: {'pixel': 1.0, 'perceptual': 0.2, 'temporal': 0.8},
            160: {'pixel': 1.0, 'perceptual': 0.25, 'temporal': 1.0}
        }
        
    def _setup_logging(self):
        """设置训练日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('multiscale_training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def _configure_stage(self, stage_idx):
        """配置当前训练阶段"""
        stage = self.scale_stages[stage_idx]
        resolution = stage['resolution']
        
        self.logger.info(f"配置训练阶段 {stage_idx + 1}: 分辨率 {resolution}px")
        
        # 重新配置优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=stage['lr'],
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=stage['epochs'],
            eta_min=stage['lr'] * 0.1
        )
        
        # 更新数据加载器（需要根据实际数据集实现）
        # self.train_loader = self._get_dataloader(resolution, stage['batch_size'])
        
        return stage
        
    def _compute_multiscale_loss(self, pred, target, resolution):
        """计算多尺度损失"""
        weights = self.perceptual_weights[resolution]
        
        # 基础像素损失
        pixel_loss = self.mse_loss(pred, target) + 0.1 * self.l1_loss(pred, target)
        
        # 时序一致性损失（如果有帧序列）
        temporal_loss = 0.0
        if self.frame_buffer.has_sequence():
            # 添加当前帧到缓冲区
            self.frame_buffer.add_frame(pred.detach())
            
            # 计算时序损失
            if len(self.frame_buffer.frames) >= 2:
                temporal_loss = self.temporal_loss(
                    pred, 
                    self.frame_buffer.frames[-2],  # 前一帧
                    target
                )
        
        # 注意力正则化损失
        attention_loss = 0.0
        if hasattr(self.model, 'get_attention_maps'):
            attention_maps = self.model.get_attention_maps()
            if attention_maps:
                # 鼓励注意力集中在嘴部区域
                for att_map in attention_maps:
                    # 简单的注意力集中度损失
                    attention_loss += torch.mean(torch.var(att_map, dim=[2, 3]))
        
        # 总损失
        total_loss = (
            weights['pixel'] * pixel_loss +
            weights['temporal'] * temporal_loss +
            0.01 * attention_loss
        )
        
        return {
            'total': total_loss,
            'pixel': pixel_loss,
            'temporal': temporal_loss,
            'attention': attention_loss
        }
        
    def train_stage(self, stage_idx, train_loader, val_loader=None):
        """训练单个阶段"""
        stage = self._configure_stage(stage_idx)
        resolution = stage['resolution']
        epochs = stage['epochs']
        
        self.logger.info(f"开始训练阶段 {stage_idx + 1}: {resolution}px, {epochs} epochs")
        
        best_val_loss = float('inf')
        stage_stats = defaultdict(list)
        
        for epoch in range(epochs):
            # 训练模式
            self.model.train()
            epoch_losses = defaultdict(float)
            num_batches = 0
            
            for batch_idx, batch_data in enumerate(train_loader):
                # 假设batch_data包含source, reference, target
                source = batch_data['source'].to(self.device)
                reference = batch_data['reference'].to(self.device)
                target = batch_data['target'].to(self.device)
                
                # 前向传播
                self.optimizer.zero_grad()
                output = self.model(source, reference)
                
                # 计算损失
                losses = self._compute_multiscale_loss(output, target, resolution)
                
                # 反向传播
                losses['total'].backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # 统计损失
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                num_batches += 1
                
                # 打印训练进度
                if batch_idx % 50 == 0:
                    self.logger.info(
                        f"Stage {stage_idx+1}, Epoch {epoch+1}/{epochs}, "
                        f"Batch {batch_idx}, Loss: {losses['total'].item():.6f}"
                    )
            
            # 计算平均损失
            avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
            
            # 验证
            val_loss = 0.0
            if val_loader:
                val_loss = self._validate(val_loader, resolution)
                
            # 学习率调度
            self.scheduler.step()
            
            # 记录统计信息
            stage_stats['train_loss'].append(avg_losses['total'])
            stage_stats['val_loss'].append(val_loss)
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(
                    stage_idx, epoch, resolution, 
                    avg_losses['total'], val_loss
                )
            
            self.logger.info(
                f"Stage {stage_idx+1}, Epoch {epoch+1}/{epochs} - "
                f"Train Loss: {avg_losses['total']:.6f}, Val Loss: {val_loss:.6f}, "
                f"LR: {self.scheduler.get_last_lr()[0]:.8f}"
            )
        
        self.total_epochs += epochs
        return stage_stats
        
    def _validate(self, val_loader, resolution):
        """验证模型性能"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                source = batch_data['source'].to(self.device)
                reference = batch_data['reference'].to(self.device)
                target = batch_data['target'].to(self.device)
                
                output = self.model(source, reference)
                losses = self._compute_multiscale_loss(output, target, resolution)
                
                total_loss += losses['total'].item()
                num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
        
    def _save_checkpoint(self, stage_idx, epoch, resolution, train_loss, val_loss):
        """保存模型检查点"""
        checkpoint = {
            'stage': stage_idx,
            'epoch': epoch,
            'resolution': resolution,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'total_epochs': self.total_epochs
        }
        
        # 保存当前最佳模型
        checkpoint_path = f'checkpoints/multiscale_stage_{stage_idx+1}_{resolution}px_best.pth'
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        self.logger.info(f"保存检查点: {checkpoint_path}")
        
    def train_all_stages(self, train_loader, val_loader=None):
        """训练所有阶段"""
        self.logger.info("开始多尺度训练策略")
        self.logger.info(f"总共 {len(self.scale_stages)} 个训练阶段")
        
        all_stats = {}
        
        for stage_idx in range(len(self.scale_stages)):
            stage_stats = self.train_stage(stage_idx, train_loader, val_loader)
            all_stats[f'stage_{stage_idx+1}'] = stage_stats
            
            # 阶段间的模型适应（可选）
            if stage_idx < len(self.scale_stages) - 1:
                self._adapt_model_for_next_stage(stage_idx)
        
        self.logger.info("多尺度训练完成")
        return all_stats
        
    def _adapt_model_for_next_stage(self, current_stage_idx):
        """为下一阶段调整模型（可选的模型适应策略）"""
        # 可以在这里实现模型权重的微调或结构调整
        # 例如：调整注意力机制的权重、更新批归一化统计等
        pass
        
    def load_checkpoint(self, checkpoint_path):
        """加载训练检查点"""
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.current_stage = checkpoint['stage']
            self.total_epochs = checkpoint['total_epochs']
            
            self.logger.info(f"加载检查点: {checkpoint_path}")
            self.logger.info(f"恢复到阶段 {self.current_stage + 1}, 总训练轮数 {self.total_epochs}")
            
            return True
        return False


def main():
    """主训练函数"""
    # 配置参数
    config = {
        'data_root': 'path/to/your/dataset',
        'num_workers': 4,
        'pin_memory': True
    }
    
    # 创建多尺度训练器
    trainer = MultiScaleTrainingStrategy(config)
    
    # 这里需要根据实际数据集实现数据加载器
    # train_loader = create_train_dataloader(config)
    # val_loader = create_val_dataloader(config)
    
    print("多尺度训练策略已初始化")
    print("训练阶段配置:")
    for i, stage in enumerate(trainer.scale_stages):
        print(f"  阶段 {i+1}: {stage['resolution']}px, {stage['epochs']} epochs, "
              f"batch_size={stage['batch_size']}, lr={stage['lr']}")
    
    # 开始训练（需要实际的数据加载器）
    # stats = trainer.train_all_stages(train_loader, val_loader)
    
    print("\n使用说明:")
    print("1. 实现数据加载器以支持不同分辨率")
    print("2. 根据GPU内存调整batch_size")
    print("3. 可以通过load_checkpoint恢复训练")
    print("4. 训练日志保存在multiscale_training.log")


if __name__ == '__main__':
    main()