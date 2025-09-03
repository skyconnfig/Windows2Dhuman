import argparse

class DINetAdvancedTrainingOptions():
    """
    进一步优化的DINet训练配置
    针对嘴部清晰度进行高级优化
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialize()

    def initialize(self):
        # ========================= 基础配置 =========================
        self.parser.add_argument('--source_channel', type=int, default=6, help='input source image channels')
        self.parser.add_argument('--ref_channel', type=int, default=15, help='input reference image channels')
        self.parser.add_argument('--audio_channel', type=int, default=29, help='input audio channels')
        self.parser.add_argument('--augment_num', type=int, default=32, help='augment training data')
        
        # 高级优化：进一步提升分辨率到160px，在128px基础上继续优化
        self.parser.add_argument('--mouth_region_size', type=int, default=160, 
                                help='mouth region size - advanced optimization to 160px')
        
        self.parser.add_argument('--train_data', type=str, default=r"./asserts/training_data/training_json.json",
                            help='path of training json')
        
        # 高级优化：进一步减小batch_size以适应更高分辨率
        self.parser.add_argument('--batch_size', type=int, default=12, 
                                help='training batch size - further reduced for 160px resolution')
        
        # ========================= 损失函数高级优化 =========================
        # 高级优化：进一步增加感知损失权重，强化视觉质量
        self.parser.add_argument('--lamb_perception', type=int, default=20, 
                                help='weight of perception loss - advanced increase for superior visual quality')
        
        self.parser.add_argument('--lamb_syncnet_perception', type=float, default=0.15, 
                                help='weight of syncnet perception loss - slightly increased')
        
        # 高级优化：进一步增加像素损失权重，最大化细节保真度
        self.parser.add_argument('--lamb_pixel', type=int, default=20, 
                                help='weight of pixel loss - advanced increase for maximum detail preservation')
        
        # 新增：嘴部区域专门损失权重
        self.parser.add_argument('--lamb_mouth_region', type=float, default=8.0, 
                                help='weight of mouth region specific loss - new advanced feature')
        
        # 新增：边缘保持损失权重
        self.parser.add_argument('--lamb_edge_preservation', type=float, default=5.0, 
                                help='weight of edge preservation loss - new advanced feature')
        
        # 新增：纹理一致性损失权重
        self.parser.add_argument('--lamb_texture_consistency', type=float, default=3.0, 
                                help='weight of texture consistency loss - new advanced feature')
        
        # ========================= 学习率高级优化 =========================
        # 高级优化：进一步降低学习率以确保高分辨率训练稳定性
        self.parser.add_argument('--lr_g', type=float, default=0.00004, 
                                help='initial learning rate for generator - further reduced for 160px stability')
        self.parser.add_argument('--lr_d', type=float, default=0.00004, 
                                help='initial learning rate for discriminator - further reduced for 160px stability')
        
        # 高级优化：调整学习率衰减策略
        self.parser.add_argument('--start_epoch', default=1, type=int, help='start epoch in training stage')
        self.parser.add_argument('--non_decay', default=6, type=int, 
                                help='num of epoches with fixed learning rate - increased for stability')
        self.parser.add_argument('--decay', default=44, type=int, 
                                help='num of linearly decay epochs - increased for gradual learning')
        
        self.parser.add_argument('--checkpoint', type=int, default=2, help='num of checkpoints in training stage')
        
        # 高级优化：更新结果路径以反映160分辨率训练
        self.parser.add_argument('--result_path', type=str, 
                                default=r"./asserts/training_model_weight/frame_training_160_advanced",
                                help='result path to save model - updated for 160px advanced training')
        
        self.parser.add_argument('--coarse2fine', action='store_true', help='If true, load pretrained model path.')
        self.parser.add_argument('--coarse_model_path', default='', type=str,
                                help='Save data (.pth) of previous training')
        self.parser.add_argument('--pretrained_syncnet_path', default='', type=str,
                                help='Save data (.pth) of pretrained syncnet')
        self.parser.add_argument('--pretrained_frame_DINet_path', default='', type=str,
                                help='Save data (.pth) of frame trained DINet')
        
        # ========================= 判别器高级配置 =========================
        self.parser.add_argument('--D_num_blocks', type=int, default=5, 
                                help='num of down blocks in discriminator - increased for higher resolution')
        self.parser.add_argument('--D_block_expansion', type=int, default=64, 
                                help='block expansion in discriminator')
        self.parser.add_argument('--D_max_features', type=int, default=512, 
                                help='max channels in discriminator - increased for better feature extraction')
        
        # ========================= 高级训练策略 =========================
        # 新增：梯度累积步数
        self.parser.add_argument('--gradient_accumulation_steps', type=int, default=2, 
                                help='gradient accumulation steps for effective larger batch size')
        
        # 新增：混合精度训练
        self.parser.add_argument('--use_mixed_precision', action='store_true', 
                                help='use mixed precision training for memory efficiency')
        
        # 新增：渐进式训练
        self.parser.add_argument('--progressive_training', action='store_true', 
                                help='use progressive training starting from lower resolution')
        
        # 新增：自适应损失权重
        self.parser.add_argument('--adaptive_loss_weights', action='store_true', 
                                help='use adaptive loss weights based on training progress')

    def parse_args(self):
        return self.parser.parse_args()

# 高级优化配置说明
"""
高级优化配置特点：

1. 分辨率优化：
   - 从128px进一步提升到160px
   - 更接近推理时的256px分辨率

2. 损失函数高级优化：
   - 感知损失权重：15 → 20
   - 像素损失权重：15 → 20
   - 新增嘴部区域专门损失：8.0
   - 新增边缘保持损失：5.0
   - 新增纹理一致性损失：3.0

3. 学习率精细调整：
   - 生成器学习率：6e-5 → 4e-5
   - 判别器学习率：6e-5 → 4e-5
   - 延长固定学习率阶段：4 → 6 epochs
   - 延长衰减阶段：36 → 44 epochs

4. 新增高级特性：
   - 梯度累积以模拟更大batch size
   - 混合精度训练节省显存
   - 渐进式训练策略
   - 自适应损失权重调整

5. 判别器增强：
   - 增加判别器层数：4 → 5
   - 增加最大特征数：256 → 512

预期效果：
- 进一步提升嘴部清晰度
- 更好的细节保真度
- 更稳定的训练过程
- 更接近推理分辨率的训练效果
"""