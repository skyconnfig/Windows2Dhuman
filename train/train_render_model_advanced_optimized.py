#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级优化训练脚本 - 数字人唇形同步
支持160px高分辨率、渐进式训练、自适应权重调整
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import cv2
from tqdm import tqdm
import time
import logging
from datetime import datetime

from talkingface.config.config_advanced_optimized import DINetAdvancedTrainingOptions
from talkingface.models.DINet import DINet_five_Ref
from talkingface.models.common.Discriminator import Discriminator
from talkingface.util.utils_advanced_optimized import AdvancedOptimizedLoss, AdaptiveLossWeights, ProgressiveTrainingScheduler
from talkingface.data.few_shot_dataset import FewShotDataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AdvancedTrainer:
    """高级优化训练器"""
    
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = GradScaler() if self.device.type == 'cuda' else None
        
        # 初始化模型
        self._init_models()
        
        # 初始化损失函数和调度器
        self._init_loss_and_schedulers()
        
        # 初始化优化器
        self._init_optimizers()
        
        # 训练统计
        self.global_step = 0
        self.best_loss = float('inf')
        self.loss_history = []
        
        logger.info(f"高级训练器初始化完成，使用设备: {self.device}")
    
    def _init_models(self):
        """初始化生成器和判别器模型"""
        # 生成器
        self.net_g = DINet_five_Ref(source_channel=6, ref_channel=15)
        
        # 高级判别器
        self.net_d = Discriminator(
            num_channels=3,
            num_blocks=self.opt.D_num_blocks,
            block_expansion=self.opt.D_block_expansion,
            max_features=self.opt.D_max_features
        )
        
        # 移动到设备
        self.net_g.to(self.device)
        self.net_d.to(self.device)
        
        # 加载预训练模型（如果存在）
        if hasattr(self.opt, 'pretrained_g_path') and os.path.exists(self.opt.pretrained_g_path):
            self.net_g.load_state_dict(torch.load(self.opt.pretrained_g_path, map_location=self.device))
            logger.info(f"加载预训练生成器: {self.opt.pretrained_g_path}")
        
        if hasattr(self.opt, 'pretrained_d_path') and os.path.exists(self.opt.pretrained_d_path):
            self.net_d.load_state_dict(torch.load(self.opt.pretrained_d_path, map_location=self.device))
            logger.info(f"加载预训练判别器: {self.opt.pretrained_d_path}")
        
        logger.info(f"生成器参数数量: {sum(p.numel() for p in self.net_g.parameters()):,}")
        logger.info(f"判别器参数数量: {sum(p.numel() for p in self.net_d.parameters()):,}")
    
    def _init_loss_and_schedulers(self):
        """初始化损失函数和调度器"""
        # 高级优化损失函数
        self.criterion_advanced = AdvancedOptimizedLoss(
            lambda_pixel=self.opt.lamb_pixel,
            lambda_perceptual=self.opt.lamb_perception,
            lambda_mouth_region=self.opt.lamb_mouth_region,
            lambda_edge_preservation=self.opt.lamb_edge_preservation,
            lambda_texture_consistency=self.opt.lamb_texture_consistency
        ).to(self.device)
        
        # 对抗损失
        self.criterion_gan = nn.MSELoss()
        
        # 自适应权重调整器
        self.adaptive_weights = AdaptiveLossWeights(
            initial_weights={
                'lambda_pixel': self.opt.lamb_pixel,
                'lambda_mouth_region': self.opt.lamb_mouth_region,
                'lambda_edge_preservation': self.opt.lamb_edge_preservation,
                'lambda_texture_consistency': self.opt.lamb_texture_consistency
            },
            adaptation_schedule={
                20: {'lambda_mouth_region': 1.2},
                40: {'lambda_edge_preservation': 1.1},
                60: {'lambda_texture_consistency': 1.15}
            }
        )
        
        # 渐进式训练调度器
        self.progressive_scheduler = ProgressiveTrainingScheduler(
            target_resolution=self.opt.mouth_region_size,
            start_resolution=64,
            progression_epochs=[10, 25, 40]
        )
    
    def _init_optimizers(self):
        """初始化优化器和学习率调度器"""
        self.optimizer_g = torch.optim.Adam(
            self.net_g.parameters(),
            lr=self.opt.lr_g,
            betas=(0.5, 0.999)
        )
        
        self.optimizer_d = torch.optim.Adam(
            self.net_d.parameters(),
            lr=self.opt.lr_d,
            betas=(0.5, 0.999)
        )
        
        # 学习率调度器
        self.scheduler_g = torch.optim.lr_scheduler.StepLR(
            self.optimizer_g, step_size=30, gamma=0.5
        )
        
        self.scheduler_d = torch.optim.lr_scheduler.StepLR(
            self.optimizer_d, step_size=30, gamma=0.5
        )
    
    def _get_current_resolution(self, epoch):
        """获取当前训练分辨率"""
        return self.progressive_scheduler.get_current_resolution(epoch)
    
    def _resize_batch(self, batch, target_size):
        """调整批次数据到目标分辨率"""
        resized_batch = {}
        for key, tensor in batch.items():
            if len(tensor.shape) == 4 and tensor.shape[1] in [3, 15]:  # 图像数据
                resized_batch[key] = F.interpolate(
                    tensor, size=(target_size, target_size), 
                    mode='bilinear', align_corners=False
                )
            else:
                resized_batch[key] = tensor
        return resized_batch
    
    def train_epoch(self, dataloader, epoch):
        """训练一个epoch"""
        self.net_g.train()
        self.net_d.train()
        
        # 获取当前分辨率
        current_resolution = self._get_current_resolution(epoch)
        logger.info(f"Epoch {epoch}: 使用分辨率 {current_resolution}px")
        
        # 更新自适应权重
        updated_weights = self.adaptive_weights.update_weights(epoch, self.loss_history)
        if updated_weights:
            self.criterion_advanced.update_weights(updated_weights)
            logger.info(f"权重已更新: {updated_weights}")
        
        epoch_losses = {
            'g_total': 0.0, 'g_advanced': 0.0, 'g_gan': 0.0,
            'd_real': 0.0, 'd_fake': 0.0,
            'pixel': 0.0, 'mouth': 0.0, 'edge': 0.0, 'texture': 0.0
        }
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for i, batch in enumerate(pbar):
            # 调整到当前分辨率
            if current_resolution != self.opt.mouth_region_size:
                batch = self._resize_batch(batch, current_resolution)
            
            # 移动数据到设备
            source_img = batch['source_img'].to(self.device)
            source_prompt = batch['source_prompt'].to(self.device)
            ref_imgs = batch['ref_imgs'].to(self.device)
            target_img = batch['target_img'].to(self.device)
            
            batch_size = source_img.size(0)
            
            # ==================== 训练判别器 ====================
            self.optimizer_d.zero_grad()
            
            with autocast() if self.scaler else torch.no_grad():
                # 生成假图像
                fake_img = self.net_g(source_img, source_prompt, ref_imgs)
                
                # 判别器对真实图像的判断
                _, d_real_out = self.net_d(target_img)
                real_labels = torch.ones_like(d_real_out)
                loss_d_real = self.criterion_gan(d_real_out, real_labels)
                
                # 判别器对假图像的判断
                _, d_fake_out = self.net_d(fake_img.detach())
                fake_labels = torch.zeros_like(d_fake_out)
                loss_d_fake = self.criterion_gan(d_fake_out, fake_labels)
                
                loss_d = (loss_d_real + loss_d_fake) * 0.5
            
            if self.scaler:
                self.scaler.scale(loss_d).backward()
                self.scaler.step(self.optimizer_d)
            else:
                loss_d.backward()
                self.optimizer_d.step()
            
            # ==================== 训练生成器 ====================
            if (i + 1) % self.opt.gradient_accumulation_steps == 0:
                self.optimizer_g.zero_grad()
                
                with autocast() if self.scaler else torch.no_grad():
                    # 生成图像
                    fake_img = self.net_g(source_img, source_prompt, ref_imgs)
                    
                    # 高级损失
                    loss_g_advanced, loss_dict = self.criterion_advanced(fake_img, target_img)
                    
                    # 对抗损失
                    _, d_fake_out = self.net_d(fake_img)
                    real_labels = torch.ones_like(d_fake_out)
                    loss_g_gan = self.criterion_gan(d_fake_out, real_labels) * self.opt.lamb_gan
                    
                    # 总生成器损失
                    loss_g = loss_g_advanced + loss_g_gan
                
                if self.scaler:
                    self.scaler.scale(loss_g).backward()
                    self.scaler.step(self.optimizer_g)
                    self.scaler.update()
                else:
                    loss_g.backward()
                    self.optimizer_g.step()
                
                # 记录损失
                epoch_losses['g_total'] += loss_g.item()
                epoch_losses['g_advanced'] += loss_g_advanced.item()
                epoch_losses['g_gan'] += loss_g_gan.item()
                epoch_losses['pixel'] += loss_dict['pixel_loss'].item()
                epoch_losses['mouth'] += loss_dict['mouth_region_loss'].item()
                epoch_losses['edge'] += loss_dict['edge_preservation_loss'].item()
                epoch_losses['texture'] += loss_dict['texture_consistency_loss'].item()
            
            epoch_losses['d_real'] += loss_d_real.item()
            epoch_losses['d_fake'] += loss_d_fake.item()
            
            # 更新进度条
            if i % 10 == 0:
                pbar.set_postfix({
                    'G': f"{loss_g.item():.4f}",
                    'D': f"{loss_d.item():.4f}",
                    'Res': f"{current_resolution}px"
                })
            
            self.global_step += 1
        
        # 计算平均损失
        num_batches = len(dataloader)
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        # 记录损失历史
        self.loss_history.append({
            'epoch': epoch,
            'pixel_loss': epoch_losses['pixel'],
            'mouth_region_loss': epoch_losses['mouth'],
            'edge_preservation_loss': epoch_losses['edge'],
            'texture_consistency_loss': epoch_losses['texture']
        })
        
        # 更新学习率
        self.scheduler_g.step()
        self.scheduler_d.step()
        
        return epoch_losses
    
    def save_checkpoint(self, epoch, losses, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'net_g_state_dict': self.net_g.state_dict(),
            'net_d_state_dict': self.net_d.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
            'scheduler_g_state_dict': self.scheduler_g.state_dict(),
            'scheduler_d_state_dict': self.scheduler_d.state_dict(),
            'losses': losses,
            'global_step': self.global_step,
            'best_loss': self.best_loss
        }
        
        # 保存最新检查点
        checkpoint_path = os.path.join(self.opt.result_path, f'checkpoint_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.opt.result_path, 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"保存最佳模型: {best_path}")
    
    def train(self, dataloader):
        """主训练循环"""
        logger.info("开始高级优化训练")
        logger.info(f"训练配置: {self.opt.mouth_region_size}px, {self.opt.num_epochs} epochs")
        
        for epoch in range(1, self.opt.num_epochs + 1):
            start_time = time.time()
            
            # 训练一个epoch
            losses = self.train_epoch(dataloader, epoch)
            
            # 计算训练时间
            epoch_time = time.time() - start_time
            
            # 日志记录
            logger.info(
                f"Epoch {epoch}/{self.opt.num_epochs} - "
                f"时间: {epoch_time:.2f}s - "
                f"G_total: {losses['g_total']:.6f} - "
                f"G_advanced: {losses['g_advanced']:.6f} - "
                f"G_gan: {losses['g_gan']:.6f} - "
                f"D_real: {losses['d_real']:.6f} - "
                f"D_fake: {losses['d_fake']:.6f}"
            )
            
            logger.info(
                f"详细损失 - "
                f"Pixel: {losses['pixel']:.6f} - "
                f"Mouth: {losses['mouth']:.6f} - "
                f"Edge: {losses['edge']:.6f} - "
                f"Texture: {losses['texture']:.6f}"
            )
            
            # 检查是否为最佳模型
            current_loss = losses['g_total']
            is_best = current_loss < self.best_loss
            if is_best:
                self.best_loss = current_loss
            
            # 保存检查点
            if epoch % self.opt.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, losses, is_best)
        
        logger.info("训练完成！")

def main():
    """主函数"""
    # 解析配置
    opt = DINetAdvancedTrainingOptions().parse_args()
    
    # 创建结果目录
    os.makedirs(opt.result_path, exist_ok=True)
    
    # 初始化训练器
    trainer = AdvancedTrainer(opt)
    
    # 创建数据加载器
    try:
        dataset = FewShotDataset(
            opt.train_data_path,
            mouth_region_size=opt.mouth_region_size,
            is_train=True
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"数据集加载成功，样本数量: {len(dataset)}")
        
    except Exception as e:
        logger.error(f"数据集加载失败: {e}")
        logger.info("使用模拟数据进行测试训练")
        
        # 创建模拟数据加载器
        class MockDataset:
            def __init__(self, size=100, resolution=160):
                self.size = size
                self.resolution = resolution
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                return {
                    'source_img': torch.randn(3, self.resolution, self.resolution),
                    'source_prompt': torch.randn(3, self.resolution, self.resolution),
                    'ref_imgs': torch.randn(15, self.resolution, self.resolution),
                    'target_img': torch.randn(3, self.resolution, self.resolution)
                }
        
        mock_dataset = MockDataset(size=50, resolution=opt.mouth_region_size)
        dataloader = DataLoader(mock_dataset, batch_size=opt.batch_size, shuffle=True)
    
    # 开始训练
    trainer.train(dataloader)

if __name__ == "__main__":
    main()