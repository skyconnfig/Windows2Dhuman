#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的128px分辨率训练测试脚本
用于验证优化配置是否正常工作
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from talkingface.config.config_optimized import DINetTrainingOptions
from talkingface.models.DINet import DINet_five_Ref
from talkingface.models.common.Discriminator import Discriminator
from talkingface.util.utils_optimized import OptimizedLoss

def test_model_initialization():
    """测试模型初始化和优化配置"""
    print("=== 测试优化配置初始化 ===")
    
    # 初始化配置
    opt = DINetTrainingOptions().parse_args()
    print(f"配置加载成功:")
    print(f"- 嘴部区域大小: {opt.mouth_region_size}px")
    print(f"- 批次大小: {opt.batch_size}")
    print(f"- 生成器学习率: {opt.lr_g}")
    print(f"- 判别器学习率: {opt.lr_d}")
    print(f"- 感知损失权重: {opt.lamb_perception}")
    print(f"- 像素损失权重: {opt.lamb_pixel}")
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cpu':
        print("警告: 未检测到CUDA，将使用CPU训练（速度较慢）")
    
    # 初始化模型
    print("\n=== 测试模型初始化 ===")
    try:
        # 生成器 (source_channel=6 因为source_img+source_prompt会拼接)
        net_g = DINet_five_Ref(source_channel=6, ref_channel=15)
        print(f"生成器初始化成功，参数数量: {sum(p.numel() for p in net_g.parameters()):,}")
        
        # 判别器
        net_d = Discriminator(num_channels=3)
        print(f"判别器初始化成功，参数数量: {sum(p.numel() for p in net_d.parameters()):,}")
        
        # 损失函数
        loss_fn = OptimizedLoss(
            lambda_pixel=opt.lamb_pixel,
            lambda_perceptual=opt.lamb_perception,
            lambda_mouth=5
        )
        print("优化损失函数初始化成功")
        
        # 优化器
        optimizer_g = torch.optim.Adam(net_g.parameters(), lr=opt.lr_g, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(net_d.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))
        print("优化器初始化成功")
        
    except Exception as e:
        print(f"模型初始化失败: {e}")
        return False
    
    # 测试前向传播
    print("\n=== 测试前向传播 ===")
    try:
        batch_size = 2
        # 创建测试数据
        source_img = torch.randn(batch_size, 3, opt.mouth_region_size, opt.mouth_region_size)
        source_prompt = torch.randn(batch_size, 3, opt.mouth_region_size, opt.mouth_region_size)  # 源提示图像
        ref_imgs = torch.randn(batch_size, 15, opt.mouth_region_size, opt.mouth_region_size)
        
        print(f"测试数据形状:")
        print(f"- 源图像: {source_img.shape}")
        print(f"- 源提示: {source_prompt.shape}")
        print(f"- 参考图像: {ref_imgs.shape}")
        
        # 前向传播
        with torch.no_grad():
            fake_out = net_g(source_img, source_prompt, ref_imgs)
            print(f"生成器输出形状: {fake_out.shape}")
            
            d_fake_features, d_fake_out = net_d(fake_out)
            d_real_features, d_real_out = net_d(source_img)
            print(f"判别器输出形状: fake_out={d_fake_out.shape}, real_out={d_real_out.shape}")
            print(f"判别器特征图数量: fake={len(d_fake_features)}, real={len(d_real_features)}")
        
        print("前向传播测试成功！")
        
    except Exception as e:
        print(f"前向传播测试失败: {e}")
        return False
    
    print("\n=== 所有测试通过！===")
    print("优化配置已成功验证，可以开始正式训练")
    return True

def main():
    """主函数"""
    print("数字人唇形优化 - 128px分辨率训练配置测试")
    print("=" * 50)
    
    success = test_model_initialization()
    
    if success:
        print("\n✅ 配置测试成功！")
        print("\n建议下一步:")
        print("1. 准备完整的训练数据集")
        print("2. 运行完整的训练脚本")
        print("3. 监控训练过程中的损失变化")
    else:
        print("\n❌ 配置测试失败，请检查环境配置")

if __name__ == "__main__":
    main()