#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化数据增强策略
减少过度模糊和噪声，提升训练数据质量
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from typing import Tuple, Optional, Union

class OptimizedDataAugmentation:
    """优化的数据增强类"""
    
    def __init__(
        self,
        resolution: int = 160,
        enable_geometric: bool = True,
        enable_photometric: bool = True,
        enable_mouth_specific: bool = True,
        quality_threshold: float = 0.7
    ):
        """
        初始化优化数据增强
        
        Args:
            resolution: 目标分辨率
            enable_geometric: 启用几何变换
            enable_photometric: 启用光度变换
            enable_mouth_specific: 启用嘴部特定增强
            quality_threshold: 质量阈值，低于此值的增强会被跳过
        """
        self.resolution = resolution
        self.enable_geometric = enable_geometric
        self.enable_photometric = enable_photometric
        self.enable_mouth_specific = enable_mouth_specific
        self.quality_threshold = quality_threshold
        
        # 初始化变换参数
        self._init_transform_params()
    
    def _init_transform_params(self):
        """初始化变换参数"""
        # 几何变换参数（保守设置）
        self.geometric_params = {
            'rotation_range': (-5, 5),  # 减小旋转角度
            'scale_range': (0.95, 1.05),  # 减小缩放范围
            'translation_range': (-0.02, 0.02),  # 减小平移范围
            'shear_range': (-2, 2),  # 减小剪切角度
        }
        
        # 光度变换参数（温和设置）
        self.photometric_params = {
            'brightness_range': (0.9, 1.1),  # 温和亮度调整
            'contrast_range': (0.9, 1.1),  # 温和对比度调整
            'saturation_range': (0.95, 1.05),  # 轻微饱和度调整
            'hue_range': (-0.02, 0.02),  # 轻微色调调整
            'gamma_range': (0.95, 1.05),  # 轻微伽马校正
        }
        
        # 噪声参数（控制强度）
        self.noise_params = {
            'gaussian_std_range': (0.001, 0.005),  # 很小的高斯噪声
            'salt_pepper_prob': 0.001,  # 很低的椒盐噪声概率
            'poisson_lambda': 0.1,  # 轻微泊松噪声
        }
        
        # 模糊参数（避免过度模糊）
        self.blur_params = {
            'gaussian_kernel_range': (1, 3),  # 小核高斯模糊
            'motion_blur_kernel_range': (3, 5),  # 小核运动模糊
            'blur_probability': 0.1,  # 降低模糊概率
        }
    
    def _calculate_image_quality(self, image: np.ndarray) -> float:
        """计算图像质量分数"""
        # 计算拉普拉斯方差（清晰度指标）
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 计算对比度
        contrast = gray.std()
        
        # 计算亮度分布均匀性
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist / hist.sum()
        entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
        
        # 综合质量分数（归一化到0-1）
        quality_score = (
            min(laplacian_var / 1000, 1.0) * 0.4 +  # 清晰度权重
            min(contrast / 50, 1.0) * 0.3 +         # 对比度权重
            min(entropy / 8, 1.0) * 0.3             # 信息熵权重
        )
        
        return quality_score
    
    def _apply_geometric_transform(self, image: np.ndarray) -> np.ndarray:
        """应用几何变换"""
        if not self.enable_geometric or random.random() > 0.5:
            return image
        
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # 随机选择变换类型
        transform_type = random.choice(['rotation', 'scale', 'translation', 'shear'])
        
        if transform_type == 'rotation':
            angle = random.uniform(*self.geometric_params['rotation_range'])
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        elif transform_type == 'scale':
            scale = random.uniform(*self.geometric_params['scale_range'])
            M = cv2.getRotationMatrix2D(center, 0, scale)
        
        elif transform_type == 'translation':
            tx = random.uniform(*self.geometric_params['translation_range']) * w
            ty = random.uniform(*self.geometric_params['translation_range']) * h
            M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        elif transform_type == 'shear':
            shear_x = random.uniform(*self.geometric_params['shear_range'])
            M = np.float32([[1, np.tan(np.radians(shear_x)), 0], [0, 1, 0]])
        
        # 应用变换
        transformed = cv2.warpAffine(
            image, M, (w, h), 
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT
        )
        
        return transformed
    
    def _apply_photometric_transform(self, image: np.ndarray) -> np.ndarray:
        """应用光度变换"""
        if not self.enable_photometric or random.random() > 0.6:
            return image
        
        # 转换为PIL图像进行处理
        pil_image = Image.fromarray(image.astype(np.uint8))
        
        # 随机选择变换类型
        transform_types = ['brightness', 'contrast', 'saturation', 'gamma']
        selected_transforms = random.sample(transform_types, k=random.randint(1, 2))
        
        for transform_type in selected_transforms:
            if transform_type == 'brightness':
                factor = random.uniform(*self.photometric_params['brightness_range'])
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(factor)
            
            elif transform_type == 'contrast':
                factor = random.uniform(*self.photometric_params['contrast_range'])
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(factor)
            
            elif transform_type == 'saturation':
                factor = random.uniform(*self.photometric_params['saturation_range'])
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(factor)
            
            elif transform_type == 'gamma':
                gamma = random.uniform(*self.photometric_params['gamma_range'])
                # 伽马校正
                img_array = np.array(pil_image).astype(np.float32) / 255.0
                img_array = np.power(img_array, gamma)
                pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
        
        return np.array(pil_image)
    
    def _apply_controlled_noise(self, image: np.ndarray) -> np.ndarray:
        """应用受控噪声"""
        if random.random() > 0.2:  # 降低噪声应用概率
            return image
        
        noise_type = random.choice(['gaussian', 'poisson'])
        
        if noise_type == 'gaussian':
            std = random.uniform(*self.noise_params['gaussian_std_range'])
            noise = np.random.normal(0, std, image.shape)
            noisy_image = image.astype(np.float32) / 255.0 + noise
            noisy_image = np.clip(noisy_image * 255, 0, 255).astype(np.uint8)
        
        elif noise_type == 'poisson':
            # 泊松噪声
            lambda_val = self.noise_params['poisson_lambda']
            noise = np.random.poisson(lambda_val, image.shape).astype(np.float32)
            noisy_image = image.astype(np.float32) + noise
            noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def _apply_controlled_blur(self, image: np.ndarray) -> np.ndarray:
        """应用受控模糊"""
        if random.random() > self.blur_params['blur_probability']:
            return image
        
        blur_type = random.choice(['gaussian', 'motion'])
        
        if blur_type == 'gaussian':
            kernel_size = random.randint(*self.blur_params['gaussian_kernel_range'])
            if kernel_size % 2 == 0:
                kernel_size += 1
            blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        
        elif blur_type == 'motion':
            kernel_size = random.randint(*self.blur_params['motion_blur_kernel_range'])
            angle = random.uniform(0, 180)
            
            # 创建运动模糊核
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1
            kernel = kernel / kernel_size
            
            # 旋转核
            center = (kernel_size // 2, kernel_size // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            
            blurred = cv2.filter2D(image, -1, kernel)
        
        return blurred
    
    def _apply_mouth_specific_augmentation(self, image: np.ndarray, mouth_region: Optional[Tuple[int, int, int, int]] = None) -> np.ndarray:
        """应用嘴部特定增强"""
        if not self.enable_mouth_specific or random.random() > 0.3:
            return image
        
        # 如果没有提供嘴部区域，假设整个图像都是嘴部区域
        if mouth_region is None:
            mouth_region = (0, 0, image.shape[1], image.shape[0])
        
        x, y, w, h = mouth_region
        mouth_roi = image[y:y+h, x:x+w]
        
        # 嘴部特定增强：轻微锐化
        if random.random() < 0.5:
            kernel = np.array([[-0.1, -0.1, -0.1],
                              [-0.1,  1.8, -0.1],
                              [-0.1, -0.1, -0.1]])
            sharpened = cv2.filter2D(mouth_roi, -1, kernel)
            mouth_roi = cv2.addWeighted(mouth_roi, 0.7, sharpened, 0.3, 0)
        
        # 嘴部对比度增强
        if random.random() < 0.3:
            lab = cv2.cvtColor(mouth_roi, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(2, 2))
            l = clahe.apply(l)
            mouth_roi = cv2.merge([l, a, b])
            mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_LAB2RGB)
        
        # 将处理后的嘴部区域放回原图
        result = image.copy()
        result[y:y+h, x:x+w] = mouth_roi
        
        return result
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor], mouth_region: Optional[Tuple[int, int, int, int]] = None) -> Union[np.ndarray, torch.Tensor]:
        """执行优化数据增强"""
        # 处理输入格式
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            # 转换tensor到numpy
            if image.dim() == 4:  # batch
                image = image[0]  # 取第一个
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
        
        original_image = image.copy()
        
        # 计算原始图像质量
        original_quality = self._calculate_image_quality(original_image)
        
        # 应用增强变换
        augmented = image.copy()
        
        # 1. 几何变换
        augmented = self._apply_geometric_transform(augmented)
        
        # 2. 光度变换
        augmented = self._apply_photometric_transform(augmented)
        
        # 3. 受控噪声
        augmented = self._apply_controlled_noise(augmented)
        
        # 4. 受控模糊
        augmented = self._apply_controlled_blur(augmented)
        
        # 5. 嘴部特定增强
        augmented = self._apply_mouth_specific_augmentation(augmented, mouth_region)
        
        # 质量检查
        augmented_quality = self._calculate_image_quality(augmented)
        
        # 如果增强后质量下降太多，使用原图
        if augmented_quality < self.quality_threshold * original_quality:
            augmented = original_image
        
        # 转换回原始格式
        if is_tensor:
            augmented = torch.from_numpy(augmented.astype(np.float32) / 255.0)
            augmented = augmented.permute(2, 0, 1)
        
        return augmented

class AdvancedAugmentationPipeline:
    """高级增强管道"""
    
    def __init__(self, resolution: int = 160, training_phase: str = 'progressive'):
        """
        初始化高级增强管道
        
        Args:
            resolution: 目标分辨率
            training_phase: 训练阶段 ('early', 'middle', 'late', 'progressive')
        """
        self.resolution = resolution
        self.training_phase = training_phase
        
        # 根据训练阶段调整增强强度
        self._setup_phase_specific_params()
        
        # 初始化增强器
        self.augmenter = OptimizedDataAugmentation(
            resolution=resolution,
            enable_geometric=self.enable_geometric,
            enable_photometric=self.enable_photometric,
            enable_mouth_specific=self.enable_mouth_specific,
            quality_threshold=self.quality_threshold
        )
    
    def _setup_phase_specific_params(self):
        """根据训练阶段设置参数"""
        if self.training_phase == 'early':
            # 早期训练：温和增强
            self.enable_geometric = True
            self.enable_photometric = True
            self.enable_mouth_specific = False
            self.quality_threshold = 0.8
        
        elif self.training_phase == 'middle':
            # 中期训练：平衡增强
            self.enable_geometric = True
            self.enable_photometric = True
            self.enable_mouth_specific = True
            self.quality_threshold = 0.7
        
        elif self.training_phase == 'late':
            # 后期训练：精细增强
            self.enable_geometric = False
            self.enable_photometric = True
            self.enable_mouth_specific = True
            self.quality_threshold = 0.9
        
        else:  # progressive
            # 渐进式训练：自适应增强
            self.enable_geometric = True
            self.enable_photometric = True
            self.enable_mouth_specific = True
            self.quality_threshold = 0.75
    
    def update_phase(self, epoch: int, total_epochs: int):
        """根据训练进度更新阶段"""
        progress = epoch / total_epochs
        
        if progress < 0.3:
            self.training_phase = 'early'
        elif progress < 0.7:
            self.training_phase = 'middle'
        else:
            self.training_phase = 'late'
        
        # 重新设置参数
        self._setup_phase_specific_params()
        
        # 更新增强器
        self.augmenter.enable_geometric = self.enable_geometric
        self.augmenter.enable_photometric = self.enable_photometric
        self.augmenter.enable_mouth_specific = self.enable_mouth_specific
        self.augmenter.quality_threshold = self.quality_threshold
    
    def __call__(self, batch_data: dict) -> dict:
        """处理批次数据"""
        augmented_batch = {}
        
        for key, data in batch_data.items():
            if 'img' in key.lower() and isinstance(data, torch.Tensor):
                # 对图像数据应用增强
                if data.dim() == 4:  # batch of images
                    augmented_images = []
                    for i in range(data.size(0)):
                        aug_img = self.augmenter(data[i])
                        augmented_images.append(aug_img)
                    augmented_batch[key] = torch.stack(augmented_images)
                else:
                    augmented_batch[key] = self.augmenter(data)
            else:
                # 非图像数据直接复制
                augmented_batch[key] = data
        
        return augmented_batch

# 测试函数
def test_augmentation():
    """测试数据增强功能"""
    print("测试优化数据增强...")
    
    # 创建测试图像
    test_image = torch.randn(3, 160, 160)
    
    # 初始化增强器
    augmenter = OptimizedDataAugmentation(resolution=160)
    
    # 应用增强
    augmented = augmenter(test_image)
    
    print(f"原始图像形状: {test_image.shape}")
    print(f"增强后图像形状: {augmented.shape}")
    print("数据增强测试成功！")
    
    # 测试管道
    pipeline = AdvancedAugmentationPipeline(resolution=160)
    
    batch_data = {
        'source_img': torch.randn(2, 3, 160, 160),
        'target_img': torch.randn(2, 3, 160, 160),
        'other_data': torch.randn(2, 10)
    }
    
    augmented_batch = pipeline(batch_data)
    
    print(f"批次增强测试成功！")
    for key, value in augmented_batch.items():
        print(f"{key}: {value.shape}")

if __name__ == "__main__":
    test_augmentation()