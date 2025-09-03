import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms

class AdvancedOptimizedLoss(nn.Module):
    """
    高级优化损失函数
    包含嘴部区域专门损失、边缘保持损失和纹理一致性损失
    """
    def __init__(self, lambda_pixel=20, lambda_perceptual=20, lambda_mouth_region=8.0, 
                 lambda_edge_preservation=5.0, lambda_texture_consistency=3.0):
        super(AdvancedOptimizedLoss, self).__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_mouth_region = lambda_mouth_region
        self.lambda_edge_preservation = lambda_edge_preservation
        self.lambda_texture_consistency = lambda_texture_consistency
        
        # 基础损失函数
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)
        
        # Sobel算子用于边缘检测
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3))
        
    def compute_edge_map(self, image):
        """
        计算图像的边缘图
        """
        # 转换为灰度图
        if image.size(1) == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image
            
        # 应用Sobel算子
        edge_x = F.conv2d(gray, self.sobel_x, padding=1)
        edge_y = F.conv2d(gray, self.sobel_y, padding=1)
        
        # 计算边缘强度
        edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
        return edge_magnitude
    
    def compute_texture_features(self, image, window_size=5):
        """
        计算图像的纹理特征（基于局部方差）
        """
        # 使用平均池化计算局部均值
        kernel = torch.ones(1, 1, window_size, window_size, device=image.device) / (window_size * window_size)
        
        if image.size(1) == 3:
            gray = 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]
        else:
            gray = image
            
        # 计算局部均值
        local_mean = F.conv2d(gray, kernel, padding=window_size//2)
        
        # 计算局部方差（纹理特征）
        local_variance = F.conv2d((gray - local_mean)**2, kernel, padding=window_size//2)
        
        return local_variance
    
    def create_mouth_mask(self, image_size, mouth_center_ratio=(0.5, 0.7), mouth_size_ratio=(0.4, 0.3)):
        """
        创建嘴部区域掩码
        mouth_center_ratio: (x, y) 嘴部中心相对位置
        mouth_size_ratio: (width, height) 嘴部区域相对大小
        """
        batch_size, _, height, width = image_size
        
        # 计算嘴部区域
        center_x = int(width * mouth_center_ratio[0])
        center_y = int(height * mouth_center_ratio[1])
        mouth_width = int(width * mouth_size_ratio[0])
        mouth_height = int(height * mouth_size_ratio[1])
        
        # 创建掩码
        mask = torch.zeros(batch_size, 1, height, width, device=image_size[0] if torch.is_tensor(image_size[0]) else 'cpu')
        
        x1 = max(0, center_x - mouth_width // 2)
        x2 = min(width, center_x + mouth_width // 2)
        y1 = max(0, center_y - mouth_height // 2)
        y2 = min(height, center_y + mouth_height // 2)
        
        mask[:, :, y1:y2, x1:x2] = 1.0
        
        return mask
    
    def forward(self, pred, target, mouth_mask=None):
        """
        前向传播计算高级优化损失
        
        Args:
            pred: 预测图像 [B, C, H, W]
            target: 目标图像 [B, C, H, W]
            mouth_mask: 嘴部掩码 [B, 1, H, W]，可选
        
        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        loss_dict = {}
        
        # 1. 基础像素损失
        pixel_loss = self.l1_loss(pred, target)
        loss_dict['pixel_loss'] = pixel_loss
        
        # 2. 嘴部区域专门损失
        mouth_region_loss = 0
        if mouth_mask is None:
            # 自动创建嘴部掩码
            mouth_mask = self.create_mouth_mask(pred.shape).to(pred.device)
        
        if mouth_mask is not None:
            # 扩展掩码到所有通道
            mouth_mask_expanded = mouth_mask.expand_as(pred)
            
            # 嘴部区域像素损失
            pred_mouth = pred * mouth_mask_expanded
            target_mouth = target * mouth_mask_expanded
            mouth_region_loss = self.mse_loss(pred_mouth, target_mouth)
            
            # 嘴部区域感知损失（基于余弦相似度）
            pred_mouth_flat = pred_mouth.view(pred.size(0), -1)
            target_mouth_flat = target_mouth.view(target.size(0), -1)
            mouth_cosine_loss = 1 - self.cosine_similarity(pred_mouth_flat, target_mouth_flat).mean()
            mouth_region_loss = mouth_region_loss + mouth_cosine_loss * 0.5
            
        loss_dict['mouth_region_loss'] = mouth_region_loss
        
        # 3. 边缘保持损失
        pred_edges = self.compute_edge_map(pred)
        target_edges = self.compute_edge_map(target)
        edge_preservation_loss = self.l1_loss(pred_edges, target_edges)
        
        # 对嘴部区域的边缘给予更高权重
        if mouth_mask is not None:
            mouth_edge_mask = mouth_mask.expand_as(pred_edges)
            pred_mouth_edges = pred_edges * mouth_edge_mask
            target_mouth_edges = target_edges * mouth_edge_mask
            mouth_edge_loss = self.mse_loss(pred_mouth_edges, target_mouth_edges)
            edge_preservation_loss = edge_preservation_loss + mouth_edge_loss * 2.0
            
        loss_dict['edge_preservation_loss'] = edge_preservation_loss
        
        # 4. 纹理一致性损失
        pred_texture = self.compute_texture_features(pred)
        target_texture = self.compute_texture_features(target)
        texture_consistency_loss = self.l1_loss(pred_texture, target_texture)
        
        # 对嘴部区域的纹理给予更高权重
        if mouth_mask is not None:
            mouth_texture_mask = F.interpolate(mouth_mask, size=pred_texture.shape[-2:], mode='bilinear', align_corners=False)
            pred_mouth_texture = pred_texture * mouth_texture_mask
            target_mouth_texture = target_texture * mouth_texture_mask
            mouth_texture_loss = self.mse_loss(pred_mouth_texture, target_mouth_texture)
            texture_consistency_loss = texture_consistency_loss + mouth_texture_loss * 1.5
            
        loss_dict['texture_consistency_loss'] = texture_consistency_loss
        
        # 5. 计算总损失
        total_loss = (self.lambda_pixel * pixel_loss + 
                     self.lambda_mouth_region * mouth_region_loss +
                     self.lambda_edge_preservation * edge_preservation_loss +
                     self.lambda_texture_consistency * texture_consistency_loss)
        
        loss_dict['total_loss'] = total_loss
        
        return total_loss, loss_dict

class AdaptiveLossWeights:
    """
    自适应损失权重调整器
    根据训练进度动态调整各项损失的权重
    """
    def __init__(self, initial_weights, adaptation_schedule=None):
        self.initial_weights = initial_weights
        self.current_weights = initial_weights.copy()
        self.adaptation_schedule = adaptation_schedule or {}
        
    def update_weights(self, epoch, loss_history=None):
        """
        根据训练进度更新权重
        
        Args:
            epoch: 当前训练轮数
            loss_history: 损失历史记录，用于自适应调整
        """
        # 基于epoch的权重调整
        if epoch in self.adaptation_schedule:
            adjustments = self.adaptation_schedule[epoch]
            for key, adjustment in adjustments.items():
                if key in self.current_weights:
                    self.current_weights[key] *= adjustment
        
        # 基于损失历史的自适应调整
        if loss_history and len(loss_history) > 10:
            recent_losses = loss_history[-10:]
            
            # 如果像素损失下降缓慢，增加像素损失权重
            pixel_trend = (recent_losses[-1]['pixel_loss'] - recent_losses[0]['pixel_loss']) / recent_losses[0]['pixel_loss']
            if pixel_trend > -0.01:  # 下降不足1%
                self.current_weights['lambda_pixel'] *= 1.05
                
            # 如果嘴部损失下降缓慢，增加嘴部损失权重
            mouth_trend = (recent_losses[-1]['mouth_region_loss'] - recent_losses[0]['mouth_region_loss']) / recent_losses[0]['mouth_region_loss']
            if mouth_trend > -0.01:
                self.current_weights['lambda_mouth_region'] *= 1.03
        
        return self.current_weights

class ProgressiveTrainingScheduler:
    """
    渐进式训练调度器
    从低分辨率开始，逐步提升到目标分辨率
    """
    def __init__(self, target_resolution=160, start_resolution=64, progression_epochs=[10, 20, 30]):
        self.target_resolution = target_resolution
        self.start_resolution = start_resolution
        self.progression_epochs = progression_epochs
        self.current_resolution = start_resolution
        
    def get_current_resolution(self, epoch):
        """
        根据当前epoch返回应该使用的分辨率
        """
        if epoch < self.progression_epochs[0]:
            self.current_resolution = self.start_resolution
        elif epoch < self.progression_epochs[1]:
            self.current_resolution = int(self.start_resolution * 1.5)  # 96
        elif epoch < self.progression_epochs[2]:
            self.current_resolution = int(self.start_resolution * 2)    # 128
        else:
            self.current_resolution = self.target_resolution            # 160
            
        return self.current_resolution

# 使用示例和配置说明
"""
高级优化损失函数使用说明：

1. 基础使用：
   loss_fn = AdvancedOptimizedLoss(
       lambda_pixel=20,
       lambda_perceptual=20,
       lambda_mouth_region=8.0,
       lambda_edge_preservation=5.0,
       lambda_texture_consistency=3.0
   )
   
   total_loss, loss_dict = loss_fn(pred_image, target_image)

2. 自适应权重使用：
   adaptive_weights = AdaptiveLossWeights(
       initial_weights={
           'lambda_pixel': 20,
           'lambda_mouth_region': 8.0,
           'lambda_edge_preservation': 5.0,
           'lambda_texture_consistency': 3.0
       },
       adaptation_schedule={
           20: {'lambda_mouth_region': 1.2},  # 第20轮增加嘴部损失权重
           40: {'lambda_edge_preservation': 1.1}  # 第40轮增加边缘损失权重
       }
   )

3. 渐进式训练使用：
   progressive_scheduler = ProgressiveTrainingScheduler(
       target_resolution=160,
       start_resolution=64,
       progression_epochs=[10, 20, 30]
   )
   
   current_res = progressive_scheduler.get_current_resolution(epoch)

特性说明：
- 嘴部区域专门损失：专注于嘴部区域的像素和感知损失
- 边缘保持损失：使用Sobel算子检测边缘，保持嘴部轮廓清晰
- 纹理一致性损失：基于局部方差的纹理特征匹配
- 自适应权重：根据训练进度自动调整损失权重
- 渐进式训练：从低分辨率逐步提升到目标分辨率
"""