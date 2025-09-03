#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级优化训练配置测试脚本
测试160px分辨率和高级损失函数配置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from talkingface.config.config_advanced_optimized import DINetAdvancedTrainingOptions
from talkingface.models.DINet import DINet_five_Ref
from talkingface.models.common.Discriminator import Discriminator
from talkingface.util.utils_advanced_optimized import AdvancedOptimizedLoss, AdaptiveLossWeights, ProgressiveTrainingScheduler

def test_advanced_model_initialization():
    """测试高级优化模型初始化和配置"""
    print("=== 测试高级优化配置初始化 ===")
    
    # 初始化高级配置
    opt = DINetAdvancedTrainingOptions().parse_args()
    print(f"高级配置加载成功:")
    print(f"- 嘴部区域大小: {opt.mouth_region_size}px")
    print(f"- 批次大小: {opt.batch_size}")
    print(f"- 生成器学习率: {opt.lr_g}")
    print(f"- 判别器学习率: {opt.lr_d}")
    print(f"- 感知损失权重: {opt.lamb_perception}")
    print(f"- 像素损失权重: {opt.lamb_pixel}")
    print(f"- 嘴部区域损失权重: {opt.lamb_mouth_region}")
    print(f"- 边缘保持损失权重: {opt.lamb_edge_preservation}")
    print(f"- 纹理一致性损失权重: {opt.lamb_texture_consistency}")
    print(f"- 梯度累积步数: {opt.gradient_accumulation_steps}")
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    if device.type == 'cpu':
        print("警告: 未检测到CUDA，将使用CPU训练（速度较慢）")
    
    # 初始化模型
    print("\n=== 测试高级模型初始化 ===")
    try:
        # 生成器 (source_channel=6 因为source_img+source_prompt会拼接)
        net_g = DINet_five_Ref(source_channel=6, ref_channel=15)
        print(f"生成器初始化成功，参数数量: {sum(p.numel() for p in net_g.parameters()):,}")
        
        # 高级判别器（增加层数和特征数）
        net_d = Discriminator(
            num_channels=3, 
            num_blocks=opt.D_num_blocks,
            block_expansion=opt.D_block_expansion,
            max_features=opt.D_max_features
        )
        print(f"高级判别器初始化成功，参数数量: {sum(p.numel() for p in net_d.parameters()):,}")
        
        # 高级优化损失函数
        advanced_loss_fn = AdvancedOptimizedLoss(
            lambda_pixel=opt.lamb_pixel,
            lambda_perceptual=opt.lamb_perception,
            lambda_mouth_region=opt.lamb_mouth_region,
            lambda_edge_preservation=opt.lamb_edge_preservation,
            lambda_texture_consistency=opt.lamb_texture_consistency
        )
        print("高级优化损失函数初始化成功")
        
        # 自适应权重调整器
        adaptive_weights = AdaptiveLossWeights(
            initial_weights={
                'lambda_pixel': opt.lamb_pixel,
                'lambda_mouth_region': opt.lamb_mouth_region,
                'lambda_edge_preservation': opt.lamb_edge_preservation,
                'lambda_texture_consistency': opt.lamb_texture_consistency
            },
            adaptation_schedule={
                20: {'lambda_mouth_region': 1.2},
                40: {'lambda_edge_preservation': 1.1}
            }
        )
        print("自适应权重调整器初始化成功")
        
        # 渐进式训练调度器
        progressive_scheduler = ProgressiveTrainingScheduler(
            target_resolution=opt.mouth_region_size,
            start_resolution=64,
            progression_epochs=[10, 20, 30]
        )
        print("渐进式训练调度器初始化成功")
        
        # 优化器
        optimizer_g = torch.optim.Adam(net_g.parameters(), lr=opt.lr_g, betas=(0.5, 0.999))
        optimizer_d = torch.optim.Adam(net_d.parameters(), lr=opt.lr_d, betas=(0.5, 0.999))
        print("优化器初始化成功")
        
    except Exception as e:
        print(f"高级模型初始化失败: {e}")
        return False
    
    # 测试前向传播
    print("\n=== 测试高级前向传播 ===")
    try:
        batch_size = 2
        # 创建测试数据
        source_img = torch.randn(batch_size, 3, opt.mouth_region_size, opt.mouth_region_size)
        source_prompt = torch.randn(batch_size, 3, opt.mouth_region_size, opt.mouth_region_size)
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
            
            # 测试高级损失函数
            total_loss, loss_dict = advanced_loss_fn(fake_out, source_img)
            print(f"\n高级损失函数测试:")
            print(f"- 总损失: {total_loss.item():.6f}")
            print(f"- 像素损失: {loss_dict['pixel_loss'].item():.6f}")
            print(f"- 嘴部区域损失: {loss_dict['mouth_region_loss'].item():.6f}")
            print(f"- 边缘保持损失: {loss_dict['edge_preservation_loss'].item():.6f}")
            print(f"- 纹理一致性损失: {loss_dict['texture_consistency_loss'].item():.6f}")
            
        print("高级前向传播测试成功！")
        
    except Exception as e:
        print(f"高级前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 测试渐进式训练调度
    print("\n=== 测试渐进式训练调度 ===")
    try:
        test_epochs = [5, 15, 25, 35]
        for epoch in test_epochs:
            current_res = progressive_scheduler.get_current_resolution(epoch)
            print(f"Epoch {epoch}: 分辨率 = {current_res}px")
        
        print("渐进式训练调度测试成功！")
        
    except Exception as e:
        print(f"渐进式训练调度测试失败: {e}")
        return False
    
    # 测试自适应权重调整
    print("\n=== 测试自适应权重调整 ===")
    try:
        # 模拟损失历史
        mock_loss_history = [
            {'pixel_loss': 0.1, 'mouth_region_loss': 0.05} for _ in range(15)
        ]
        
        # 测试权重更新
        original_weights = adaptive_weights.current_weights.copy()
        updated_weights = adaptive_weights.update_weights(epoch=25, loss_history=mock_loss_history)
        
        print(f"权重调整测试:")
        for key in original_weights:
            if key in updated_weights:
                print(f"- {key}: {original_weights[key]} → {updated_weights[key]}")
        
        print("自适应权重调整测试成功！")
        
    except Exception as e:
        print(f"自适应权重调整测试失败: {e}")
        return False
    
    print("\n=== 所有高级测试通过！===")
    print("高级优化配置已成功验证，可以开始高级训练")
    return True

def main():
    """主函数"""
    print("数字人唇形高级优化 - 160px分辨率训练配置测试")
    print("=" * 60)
    
    success = test_advanced_model_initialization()
    
    if success:
        print("\n✅ 高级配置测试成功！")
        print("\n高级优化特性:")
        print("1. 160px高分辨率训练")
        print("2. 嘴部区域专门损失函数")
        print("3. 边缘保持和纹理一致性损失")
        print("4. 自适应损失权重调整")
        print("5. 渐进式训练策略")
        print("6. 增强的判别器架构")
        print("\n建议下一步:")
        print("1. 准备高质量的训练数据集")
        print("2. 运行高级优化训练脚本")
        print("3. 使用渐进式训练策略")
        print("4. 监控各项损失的变化趋势")
        print("5. 根据训练进度调整损失权重")
    else:
        print("\n❌ 高级配置测试失败，请检查环境配置")

if __name__ == "__main__":
    main()