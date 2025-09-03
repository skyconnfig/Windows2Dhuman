import numpy as np
import cv2
import torch
import torch.nn as nn
from talkingface.utils import *

def draw_face_feature_maps_optimized(keypoints, mode=["mouth_bias"], im_edges=None, 
                                   mouth_width=None, mouth_height=None, blur_kernel_size=3):
    """
    优化版本的面部特征图生成函数
    - 减少模糊程度，提高唇形清晰度
    - 降低噪声强度
    - 改进边缘保持
    """
    if im_edges is None:
        im_edges = np.zeros([256, 256, 3], dtype=np.uint8)
    
    im_edges = im_edges.copy()
    
    for mode_name in mode:
        if mode_name == "mouth_bias":
            # 优化：减少模糊核大小，从10x10降到3x3或5x5
            blur_kernel = max(3, min(blur_kernel_size, 7))  # 限制模糊核大小
            
            # 嘴部区域处理
            if mouth_width is not None and mouth_height is not None:
                # 计算嘴部中心点
                mouth_center_x = int(keypoints[INDEX_NOSE_EDGE[5], 0])
                mouth_center_y = int(keypoints[INDEX_NOSE_EDGE[5], 1])
                
                # 优化：使用更精确的嘴部区域定义
                mouth_region_x1 = max(0, int(mouth_center_x - mouth_width // 2))
                mouth_region_y1 = max(0, int(mouth_center_y - mouth_height // 2))
                mouth_region_x2 = min(im_edges.shape[1], int(mouth_center_x + mouth_width // 2))
                mouth_region_y2 = min(im_edges.shape[0], int(mouth_center_y + mouth_height // 2))
                
                # 提取嘴部区域
                mouth_region = im_edges[mouth_region_y1:mouth_region_y2, mouth_region_x1:mouth_region_x2].copy()
                
                if mouth_region.size > 0:
                    # 优化：减少亮度调整的随机性，使用更温和的调整
                    brightness_factor = np.random.uniform(0.85, 1.15)  # 从0.7-1.3改为0.85-1.15
                    mouth_region = np.clip(mouth_region.astype(np.float32) * brightness_factor, 0, 255).astype(np.uint8)
                    
                    # 优化：使用更轻微的模糊
                    if blur_kernel > 1:
                        mouth_region = cv2.GaussianBlur(mouth_region, (blur_kernel, blur_kernel), 0)
                    
                    # 优化：减少噪声强度
                    noise_sigma = 3  # 从8降到3
                    if noise_sigma > 0:
                        noise = np.random.normal(0, noise_sigma, mouth_region.shape).astype(np.float32)
                        mouth_region = np.clip(mouth_region.astype(np.float32) + noise, 0, 255).astype(np.uint8)
                    
                    # 优化：改进缩放质量，使用双三次插值
                    if mouth_region.shape[0] != 50 or mouth_region.shape[1] != 100:
                        mouth_region = cv2.resize(mouth_region, (100, 50), interpolation=cv2.INTER_CUBIC)
                    
                    # 恢复到原始尺寸，保持高质量
                    original_height = mouth_region_y2 - mouth_region_y1
                    original_width = mouth_region_x2 - mouth_region_x1
                    if original_height > 0 and original_width > 0:
                        mouth_region = cv2.resize(mouth_region, (original_width, original_height), interpolation=cv2.INTER_CUBIC)
                        im_edges[mouth_region_y1:mouth_region_y2, mouth_region_x1:mouth_region_x2] = mouth_region
        
        elif mode_name == "edge":
            # 优化：改进边缘检测，保持更多细节
            gray = cv2.cvtColor(im_edges, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # 调整阈值以保持更多边缘
            edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)  # 轻微膨胀
            im_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        elif mode_name == "nose":
            # 鼻子区域处理（保持原有逻辑但减少模糊）
            nose_points = keypoints[INDEX_NOSE_EDGE]
            if len(nose_points) > 0:
                nose_center = np.mean(nose_points, axis=0).astype(int)
                cv2.circle(im_edges, tuple(nose_center), 5, (255, 255, 255), -1)
        
        elif mode_name == "eye":
            # 眼部区域处理（保持原有逻辑）
            for eye_index in [INDEX_LEFT_EYE, INDEX_RIGHT_EYE]:
                if len(keypoints[eye_index]) > 0:
                    eye_center = np.mean(keypoints[eye_index], axis=0).astype(int)
                    cv2.circle(im_edges, tuple(eye_center), 3, (255, 255, 255), -1)
    
    return im_edges

def apply_optimized_augmentation(image, keypoints, is_train=True):
    """
    优化的数据增强函数
    - 减少过度的随机变换
    - 保持嘴部区域的清晰度
    - 改进颜色和亮度调整
    """
    if not is_train:
        return image, keypoints
    
    augmented_image = image.copy()
    augmented_keypoints = keypoints.copy()
    
    # 优化：减少随机裁剪的幅度
    if np.random.random() < 0.3:  # 降低随机裁剪概率
        crop_factor = np.random.uniform(0.95, 1.0)  # 减少裁剪幅度
        h, w = augmented_image.shape[:2]
        new_h, new_w = int(h * crop_factor), int(w * crop_factor)
        
        if new_h > 0 and new_w > 0:
            start_h = np.random.randint(0, h - new_h + 1)
            start_w = np.random.randint(0, w - new_w + 1)
            
            augmented_image = augmented_image[start_h:start_h+new_h, start_w:start_w+new_w]
            augmented_image = cv2.resize(augmented_image, (w, h), interpolation=cv2.INTER_CUBIC)
            
            # 调整关键点坐标
            scale_h, scale_w = h / new_h, w / new_w
            augmented_keypoints[:, 0] = (augmented_keypoints[:, 0] - start_w) * scale_w
            augmented_keypoints[:, 1] = (augmented_keypoints[:, 1] - start_h) * scale_h
    
    # 优化：更温和的亮度和对比度调整
    if np.random.random() < 0.5:
        brightness = np.random.uniform(-15, 15)  # 从-30,30改为-15,15
        contrast = np.random.uniform(0.9, 1.1)   # 从0.8,1.2改为0.9,1.1
        
        augmented_image = augmented_image.astype(np.float32)
        augmented_image = augmented_image * contrast + brightness
        augmented_image = np.clip(augmented_image, 0, 255).astype(np.uint8)
    
    # 优化：减少色彩抖动
    if np.random.random() < 0.3:  # 降低色彩抖动概率
        hsv = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= np.random.uniform(0.95, 1.05)  # 饱和度微调
        hsv[:, :, 2] *= np.random.uniform(0.95, 1.05)  # 明度微调
        hsv = np.clip(hsv, 0, 255)
        augmented_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # 优化：轻微的高斯噪声（如果需要）
    if np.random.random() < 0.2:  # 降低噪声添加概率
        noise = np.random.normal(0, 2, augmented_image.shape).astype(np.float32)  # 减少噪声强度
        augmented_image = np.clip(augmented_image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    
    return augmented_image, augmented_keypoints

def create_optimized_mouth_mask(keypoints, image_shape, blur_strength=1):
    """
    创建优化的嘴部遮罩
    - 更精确的嘴部轮廓
    - 减少模糊边缘
    - 保持细节
    """
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    
    # 使用更精确的嘴部关键点
    mouth_points = keypoints[INDEX_LIPS_OUTER].astype(np.int32)
    
    if len(mouth_points) > 0:
        # 创建嘴部轮廓
        cv2.fillPoly(mask, [mouth_points], 255)
        
        # 优化：使用更小的模糊核
        if blur_strength > 0:
            kernel_size = max(3, blur_strength * 2 + 1)
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
    
    return mask

class OptimizedLoss(nn.Module):
    """
    优化的损失函数，专注于嘴部区域的清晰度
    """
    def __init__(self, lambda_pixel=10, lambda_perceptual=15, lambda_mouth=5):
        super(OptimizedLoss, self).__init__()
        self.lambda_pixel = lambda_pixel
        self.lambda_perceptual = lambda_perceptual
        self.lambda_mouth = lambda_mouth
        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, pred, target, mouth_mask=None):
        # 基础像素损失
        pixel_loss = self.l1_loss(pred, target)
        
        # 嘴部区域专门损失
        mouth_loss = 0
        if mouth_mask is not None:
            mouth_mask = mouth_mask.unsqueeze(1).expand_as(pred)
            pred_mouth = pred * mouth_mask
            target_mouth = target * mouth_mask
            mouth_loss = self.mse_loss(pred_mouth, target_mouth)
        
        total_loss = (self.lambda_pixel * pixel_loss + 
                     self.lambda_mouth * mouth_loss)
        
        return total_loss, pixel_loss, mouth_loss

def save_training_samples_optimized(fake_out, target_tensor, source_prompt_tensor, 
                                  ref_tensor, epoch, iteration, save_path):
    """
    保存优化训练样本，便于监控训练质量
    """
    os.makedirs(save_path, exist_ok=True)
    
    with torch.no_grad():
        # 转换为numpy格式
        fake_np = (fake_out[0] * 255).cpu().permute(1, 2, 0).float().numpy().astype(np.uint8)
        target_np = (target_tensor[0] * 255).cpu().permute(1, 2, 0).float().numpy().astype(np.uint8)
        source_np = (source_prompt_tensor[0, :3] * 255).cpu().permute(1, 2, 0).float().numpy().astype(np.uint8)
        
        # 参考帧
        ref1_np = (ref_tensor[0, :3] * 255).cpu().permute(1, 2, 0).float().numpy().astype(np.uint8)
        ref2_np = (ref_tensor[0, 3:6] * 255).cpu().permute(1, 2, 0).float().numpy().astype(np.uint8)
        
        # 拼接图像
        combined = np.concatenate([source_np, target_np, fake_np, ref1_np, ref2_np], axis=1)
        combined = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        
        # 保存
        filename = f"epoch_{epoch:03d}_iter_{iteration:04d}.jpg"
        cv2.imwrite(os.path.join(save_path, filename), combined)

print("Optimized utilities loaded successfully!")