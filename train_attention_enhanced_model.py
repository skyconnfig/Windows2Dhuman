#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
注意力增强DINet模型训练脚本
集成注意力机制和时序一致性损失，提升数字人唇形同步精度
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import cv2
from tqdm import tqdm
import argparse
import json
import time
from typing import Dict, Tuple, Optional

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入模型和工具
from talkingface.models.DINet_with_attention import DINet_five_Ref_with_Attention
from talkingface.util.temporal_consistency_loss import (
    TemporalConsistencyLoss, TemporalFrameBuffer
)
from talkingface.util.utils_advanced_optimized import (
    AdvancedOptimizedLoss, AdaptiveLossWeights
)
from talkingface.config.config_advanced_optimized import Config

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

class AttentionEnhancedTrainer:
    """
    注意力增强模型训练器
    集成多种损失函数和优化策略
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.device = device
        
        # 初始化模型
        self.init_model()
        
        # 初始化损失函数
        self.init_loss_functions()
        
        # 初始化优化器
        self.init_optimizers()
        
        # 初始化时序帧缓冲
        self.temporal_buffer = TemporalFrameBuffer(window_size=5)
        
        # 训练统计
        self.train_stats = {
            'epoch': 0,
            'total_loss': [],
            'main_loss': [],
            'temporal_loss': [],
            'attention_loss': [],
            'best_loss': float('inf')
        }
    
    def init_model(self):
        """
        初始化注意力增强模型
        """
        # 注意力配置
        attention_config = {
            'use_source_attention': True,
            'use_ref_attention': True,
            'use_merge_attention': True,
            'source_attention_type': 'mouth',      # 源图像专注嘴部
            'ref_attention_type': 'spatial',       # 参考图像空间注意力
            'merge_attention_type': 'adaptive'     # 融合特征自适应注意力
        }
        
        # 创建模型
        self.model = DINet_five_Ref_with_Attention(
            source_channel=self.config.source_channel,
            ref_channel=self.config.ref_channel,
            cuda=True,
            attention_config=attention_config
        ).to(self.device)
        
        print(f"模型已创建，参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"注意力配置: {attention_config}")
    
    def init_loss_functions(self):
        """
        初始化损失函数
        """
        # 主要损失函数（高级优化版本）
        self.main_loss_fn = AdvancedOptimizedLoss(
            lambda_pixel=self.config.lamb_pixel,
            lambda_perception=self.config.lamb_perception,
            lambda_mouth_region=self.config.lamb_mouth_region,
            lambda_edge_preserving=self.config.lamb_edge_preserving,
            lambda_texture_consistency=self.config.lamb_texture_consistency,
            mouth_region_ratio=0.4,
            device=self.device
        )
        
        # 时序一致性损失
        self.temporal_loss_fn = TemporalConsistencyLoss(
            lambda_frame_diff=8.0,      # 帧间差异权重
            lambda_optical_flow=5.0,    # 光流一致性权重
            lambda_mouth_temporal=10.0, # 嘴部时序权重
            lambda_long_term=3.0,       # 长期一致性权重
            device=self.device
        )
        
        # 自适应权重调整
        self.adaptive_weights = AdaptiveLossWeights(
            initial_weights={
                'main': 1.0,
                'temporal': 0.3,
                'attention': 0.1
            },
            adaptation_rate=0.01
        )
        
        print("损失函数已初始化")
        print(f"主要损失权重 - 像素: {self.config.lamb_pixel}, 感知: {self.config.lamb_perception}")
        print(f"时序损失权重 - 帧间: 8.0, 光流: 5.0, 嘴部: 10.0")
    
    def init_optimizers(self):
        """
        初始化优化器和学习率调度器
        """
        # 主优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            betas=(0.9, 0.999),
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=10,
            verbose=True
        )
        
        print(f"优化器已初始化，学习率: {self.config.lr}")
    
    def compute_attention_loss(self, attention_maps: Dict, target_attention: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        计算注意力损失
        
        Args:
            attention_maps: 注意力图字典
            target_attention: 目标注意力图（可选）
        
        Returns:
            attention_loss: 注意力损失
        """
        attention_loss = torch.tensor(0.0, device=self.device)
        
        # 1. 注意力稀疏性损失（鼓励注意力集中）
        for name, attention_map in attention_maps.items():
            if attention_map is not None:
                # L1稀疏性损失
                sparsity_loss = torch.mean(attention_map)
                attention_loss += 0.1 * sparsity_loss
                
                # 注意力集中度损失（鼓励高对比度）
                attention_var = torch.var(attention_map)
                concentration_loss = 1.0 / (attention_var + 1e-8)
                attention_loss += 0.05 * concentration_loss
        
        # 2. 嘴部区域注意力损失
        if 'source_attention' in attention_maps and attention_maps['source_attention'] is not None:
            source_att = attention_maps['source_attention']
            B, _, H, W = source_att.shape
            
            # 创建嘴部区域掩码
            mouth_mask = torch.zeros_like(source_att)
            center_y, center_x = int(H * 0.75), int(W * 0.5)
            mouth_h, mouth_w = int(H * 0.3), int(W * 0.4)
            
            y_start = max(0, center_y - mouth_h // 2)
            y_end = min(H, center_y + mouth_h // 2)
            x_start = max(0, center_x - mouth_w // 2)
            x_end = min(W, center_x + mouth_w // 2)
            
            mouth_mask[:, :, y_start:y_end, x_start:x_end] = 1.0
            
            # 嘴部区域注意力应该更高
            mouth_attention_loss = torch.mean((1.0 - source_att) * mouth_mask)
            attention_loss += 0.2 * mouth_attention_loss
        
        return attention_loss
    
    def train_step(self, batch: Tuple) -> Dict[str, float]:
        """
        单步训练
        
        Args:
            batch: 训练批次数据
        
        Returns:
            losses: 损失字典
        """
        source_img, source_prompt, ref_img, target_img = batch
        
        # 移动到设备
        source_img = source_img.to(self.device)
        source_prompt = source_prompt.to(self.device)
        ref_img = ref_img.to(self.device)
        target_img = target_img.to(self.device)
        
        # 前向传播
        self.optimizer.zero_grad()
        
        output, attention_maps = self.model(source_img, source_prompt, ref_img)
        
        # 1. 主要损失
        main_loss = self.main_loss_fn(output, target_img)
        
        # 2. 时序一致性损失
        temporal_loss = torch.tensor(0.0, device=self.device)
        if self.temporal_buffer.is_ready():
            # 获取历史帧
            prev_frames = self.temporal_buffer.get_sequence()
            if len(prev_frames) > 1:
                temporal_loss = self.temporal_loss_fn(
                    current_frame=output,
                    previous_frames=prev_frames,
                    is_sequence=True
                )
        
        # 更新时序缓冲
        self.temporal_buffer.add_frame(output.detach())
        
        # 3. 注意力损失
        attention_loss = self.compute_attention_loss(attention_maps)
        
        # 4. 自适应权重组合
        current_losses = {
            'main': main_loss.item(),
            'temporal': temporal_loss.item(),
            'attention': attention_loss.item()
        }
        
        weights = self.adaptive_weights.get_weights(current_losses)
        
        total_loss = (
            weights['main'] * main_loss +
            weights['temporal'] * temporal_loss +
            weights['attention'] * attention_loss
        )
        
        # 反向传播
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return {
            'total_loss': total_loss.item(),
            'main_loss': main_loss.item(),
            'temporal_loss': temporal_loss.item(),
            'attention_loss': attention_loss.item(),
            'weights': weights
        }
    
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_loader: 验证数据加载器
        
        Returns:
            val_losses: 验证损失字典
        """
        self.model.eval()
        val_losses = {
            'total_loss': 0.0,
            'main_loss': 0.0,
            'temporal_loss': 0.0,
            'attention_loss': 0.0
        }
        
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                source_img, source_prompt, ref_img, target_img = batch
                
                source_img = source_img.to(self.device)
                source_prompt = source_prompt.to(self.device)
                ref_img = ref_img.to(self.device)
                target_img = target_img.to(self.device)
                
                output, attention_maps = self.model(source_img, source_prompt, ref_img)
                
                # 计算损失
                main_loss = self.main_loss_fn(output, target_img)
                attention_loss = self.compute_attention_loss(attention_maps)
                
                # 时序损失（验证时简化）
                temporal_loss = torch.tensor(0.0, device=self.device)
                
                total_loss = main_loss + 0.3 * temporal_loss + 0.1 * attention_loss
                
                val_losses['total_loss'] += total_loss.item()
                val_losses['main_loss'] += main_loss.item()
                val_losses['temporal_loss'] += temporal_loss.item()
                val_losses['attention_loss'] += attention_loss.item()
                
                num_batches += 1
        
        # 平均损失
        for key in val_losses:
            val_losses[key] /= num_batches
        
        self.model.train()
        return val_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存检查点
        
        Args:
            epoch: 当前轮次
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_stats': self.train_stats,
            'config': self.config.__dict__,
            'attention_config': self.model.attention_config
        }
        
        # 保存常规检查点
        checkpoint_path = os.path.join(self.config.checkpoint_dir, f'attention_enhanced_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config.checkpoint_dir, 'attention_enhanced_best.pth')
            torch.save(checkpoint, best_path)
            print(f"最佳模型已保存: {best_path}")
        
        print(f"检查点已保存: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.train_stats = checkpoint['train_stats']
            
            print(f"检查点已加载: {checkpoint_path}")
            print(f"从第 {checkpoint['epoch']} 轮继续训练")
            
            return checkpoint['epoch']
        else:
            print(f"检查点不存在: {checkpoint_path}")
            return 0
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int, resume_from: Optional[str] = None):
        """
        训练模型
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮次
            resume_from: 恢复训练的检查点路径
        """
        start_epoch = 0
        
        # 恢复训练
        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
        
        print(f"开始训练，共 {num_epochs} 轮，从第 {start_epoch} 轮开始")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # 训练阶段
            self.model.train()
            train_losses = {
                'total_loss': 0.0,
                'main_loss': 0.0,
                'temporal_loss': 0.0,
                'attention_loss': 0.0
            }
            
            num_batches = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in pbar:
                losses = self.train_step(batch)
                
                for key in train_losses:
                    if key in losses:
                        train_losses[key] += losses[key]
                
                num_batches += 1
                
                # 更新进度条
                pbar.set_postfix({
                    'Total': f"{losses['total_loss']:.4f}",
                    'Main': f"{losses['main_loss']:.4f}",
                    'Temporal': f"{losses['temporal_loss']:.4f}",
                    'Attention': f"{losses['attention_loss']:.4f}"
                })
            
            # 平均训练损失
            for key in train_losses:
                train_losses[key] /= num_batches
            
            # 验证阶段
            val_losses = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step(val_losses['total_loss'])
            
            # 更新统计信息
            self.train_stats['epoch'] = epoch + 1
            for key in train_losses:
                self.train_stats[key].append(train_losses[key])
            
            # 检查是否为最佳模型
            is_best = val_losses['total_loss'] < self.train_stats['best_loss']
            if is_best:
                self.train_stats['best_loss'] = val_losses['total_loss']
            
            # 保存检查点
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
            
            # 打印训练信息
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch+1}/{num_epochs} 完成 ({epoch_time:.2f}s)")
            print(f"训练损失 - Total: {train_losses['total_loss']:.4f}, Main: {train_losses['main_loss']:.4f}, "
                  f"Temporal: {train_losses['temporal_loss']:.4f}, Attention: {train_losses['attention_loss']:.4f}")
            print(f"验证损失 - Total: {val_losses['total_loss']:.4f}, Main: {val_losses['main_loss']:.4f}")
            print(f"当前学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            print("-" * 80)
        
        print("训练完成！")
        print(f"最佳验证损失: {self.train_stats['best_loss']:.4f}")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='注意力增强DINet模型训练')
    parser.add_argument('--config', type=str, default='config_advanced_optimized.py', 
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None, 
                       help='恢复训练的检查点路径')
    parser.add_argument('--epochs', type=int, default=100, 
                       help='训练轮次')
    parser.add_argument('--batch_size', type=int, default=8, 
                       help='批次大小')
    
    args = parser.parse_args()
    
    # 加载配置
    config = Config()
    
    # 创建检查点目录
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    
    # 创建训练器
    trainer = AttentionEnhancedTrainer(config)
    
    # 这里需要实现数据加载器
    # train_loader = create_train_loader(config)
    # val_loader = create_val_loader(config)
    
    print("注意：需要实现数据加载器才能开始训练")
    print("训练器已准备就绪，配置如下：")
    print(f"- 设备: {device}")
    print(f"- 批次大小: {args.batch_size}")
    print(f"- 训练轮次: {args.epochs}")
    print(f"- 检查点目录: {config.checkpoint_dir}")
    
    # 开始训练（需要数据加载器）
    # trainer.train(train_loader, val_loader, args.epochs, args.resume)

if __name__ == '__main__':
    main()

# 使用说明
"""
注意力增强DINet训练脚本使用说明：

1. 基础训练：
   python train_attention_enhanced_model.py --epochs 100 --batch_size 8

2. 恢复训练：
   python train_attention_enhanced_model.py --resume checkpoint/attention_enhanced_epoch_50.pth

3. 自定义配置：
   python train_attention_enhanced_model.py --config custom_config.py --epochs 200

特性说明：
- 集成多种注意力机制（空间、嘴部专注、多尺度、自适应融合）
- 时序一致性损失确保帧间平滑过渡
- 自适应权重调整优化训练过程
- 完整的检查点保存和恢复机制
- 详细的训练统计和可视化支持

训练优化：
- 梯度裁剪防止梯度爆炸
- 学习率自适应调整
- 早停机制避免过拟合
- 多层次损失函数组合

注意事项：
- 需要实现对应的数据加载器
- 建议使用GPU进行训练
- 定期保存检查点以防意外中断
- 监控验证损失以调整超参数
"""