#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
音素到口型精确映射系统 - 数字人嘴部动画优化
实现不同音素对应的精确口型变化，支持嘴唇开合、牙齿显露、舌头位置控制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from enum import Enum

class PhonemeType(Enum):
    """音素类型枚举"""
    VOWEL = "vowel"           # 元音
    CONSONANT = "consonant"   # 辅音
    NASAL = "nasal"           # 鼻音
    FRICATIVE = "fricative"   # 摩擦音
    PLOSIVE = "plosive"       # 爆破音
    LIQUID = "liquid"         # 流音
    GLIDE = "glide"           # 滑音
    SILENCE = "silence"       # 静音

@dataclass
class MouthShape:
    """嘴型形状参数"""
    lip_opening: float        # 嘴唇开合度 (0-1)
    lip_width: float          # 嘴唇宽度 (0-1)
    lip_protrusion: float     # 嘴唇突出度 (0-1)
    teeth_visibility: float   # 牙齿显露度 (0-1)
    tongue_position: float    # 舌头位置 (0-1, 0=后, 1=前)
    tongue_height: float      # 舌头高度 (0-1, 0=低, 1=高)
    jaw_opening: float        # 下颌开合度 (0-1)
    corner_pull: float        # 嘴角拉伸 (-1到1, 负值向下，正值向上)
    
    def to_tensor(self) -> torch.Tensor:
        """转换为张量格式"""
        return torch.tensor([
            self.lip_opening, self.lip_width, self.lip_protrusion,
            self.teeth_visibility, self.tongue_position, self.tongue_height,
            self.jaw_opening, self.corner_pull
        ], dtype=torch.float32)
    
    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> 'MouthShape':
        """从张量创建MouthShape"""
        return cls(
            lip_opening=tensor[0].item(),
            lip_width=tensor[1].item(),
            lip_protrusion=tensor[2].item(),
            teeth_visibility=tensor[3].item(),
            tongue_position=tensor[4].item(),
            tongue_height=tensor[5].item(),
            jaw_opening=tensor[6].item(),
            corner_pull=tensor[7].item()
        )

class PhonemeToMouthMapper:
    """音素到口型映射器"""
    
    def __init__(self):
        self.phoneme_shapes = self._initialize_phoneme_shapes()
        self.transition_weights = self._initialize_transition_weights()
    
    def _initialize_phoneme_shapes(self) -> Dict[str, MouthShape]:
        """初始化音素对应的标准口型"""
        shapes = {}
        
        # 元音 (Vowels)
        shapes['a'] = MouthShape(0.8, 0.7, 0.2, 0.3, 0.3, 0.2, 0.8, 0.0)  # /a/ 如"啊"
        shapes['e'] = MouthShape(0.5, 0.8, 0.1, 0.4, 0.5, 0.6, 0.5, 0.1)  # /e/ 如"诶"
        shapes['i'] = MouthShape(0.3, 0.9, 0.0, 0.5, 0.8, 0.8, 0.3, 0.2)  # /i/ 如"衣"
        shapes['o'] = MouthShape(0.7, 0.4, 0.8, 0.2, 0.2, 0.3, 0.6, 0.0)  # /o/ 如"哦"
        shapes['u'] = MouthShape(0.4, 0.2, 0.9, 0.1, 0.1, 0.2, 0.4, 0.0)  # /u/ 如"乌"
        
        # 双元音
        shapes['ai'] = MouthShape(0.6, 0.8, 0.3, 0.4, 0.6, 0.5, 0.6, 0.1) # /ai/ 如"爱"
        shapes['ei'] = MouthShape(0.4, 0.9, 0.2, 0.5, 0.7, 0.7, 0.4, 0.2) # /ei/ 如"诶"
        shapes['ao'] = MouthShape(0.8, 0.5, 0.6, 0.3, 0.3, 0.3, 0.7, 0.0) # /ao/ 如"熬"
        shapes['ou'] = MouthShape(0.6, 0.3, 0.8, 0.2, 0.2, 0.3, 0.5, 0.0) # /ou/ 如"欧"
        
        # 辅音 - 爆破音 (Plosives)
        shapes['p'] = MouthShape(0.0, 0.5, 0.1, 0.0, 0.3, 0.3, 0.0, 0.0)  # /p/ 双唇闭合
        shapes['b'] = MouthShape(0.0, 0.5, 0.1, 0.0, 0.3, 0.3, 0.0, 0.0)  # /b/ 双唇闭合
        shapes['t'] = MouthShape(0.2, 0.6, 0.0, 0.6, 0.9, 0.8, 0.2, 0.0)  # /t/ 舌尖齿音
        shapes['d'] = MouthShape(0.2, 0.6, 0.0, 0.6, 0.9, 0.8, 0.2, 0.0)  # /d/ 舌尖齿音
        shapes['k'] = MouthShape(0.3, 0.5, 0.0, 0.3, 0.1, 0.7, 0.3, 0.0)  # /k/ 舌根音
        shapes['g'] = MouthShape(0.3, 0.5, 0.0, 0.3, 0.1, 0.7, 0.3, 0.0)  # /g/ 舌根音
        
        # 辅音 - 摩擦音 (Fricatives)
        shapes['f'] = MouthShape(0.1, 0.6, 0.2, 0.7, 0.4, 0.4, 0.1, 0.0)  # /f/ 唇齿音
        shapes['v'] = MouthShape(0.1, 0.6, 0.2, 0.7, 0.4, 0.4, 0.1, 0.0)  # /v/ 唇齿音
        shapes['s'] = MouthShape(0.2, 0.7, 0.0, 0.8, 0.8, 0.7, 0.2, 0.0)  # /s/ 齿音
        shapes['z'] = MouthShape(0.2, 0.7, 0.0, 0.8, 0.8, 0.7, 0.2, 0.0)  # /z/ 齿音
        shapes['sh'] = MouthShape(0.3, 0.4, 0.6, 0.4, 0.6, 0.6, 0.3, 0.0) # /ʃ/ 如"师"
        shapes['zh'] = MouthShape(0.3, 0.4, 0.6, 0.4, 0.6, 0.6, 0.3, 0.0) # /ʒ/ 如"日"
        shapes['h'] = MouthShape(0.4, 0.6, 0.1, 0.2, 0.3, 0.3, 0.4, 0.0)  # /h/ 喉音
        
        # 辅音 - 鼻音 (Nasals)
        shapes['m'] = MouthShape(0.0, 0.5, 0.1, 0.0, 0.3, 0.3, 0.0, 0.0)  # /m/ 双唇鼻音
        shapes['n'] = MouthShape(0.2, 0.6, 0.0, 0.4, 0.8, 0.7, 0.2, 0.0)  # /n/ 舌尖鼻音
        shapes['ng'] = MouthShape(0.3, 0.5, 0.0, 0.2, 0.1, 0.6, 0.3, 0.0) # /ŋ/ 舌根鼻音
        
        # 辅音 - 流音 (Liquids)
        shapes['l'] = MouthShape(0.3, 0.6, 0.0, 0.5, 0.7, 0.6, 0.3, 0.0)  # /l/ 舌侧音
        shapes['r'] = MouthShape(0.4, 0.5, 0.3, 0.3, 0.5, 0.5, 0.4, 0.0)  # /r/ 卷舌音
        
        # 辅音 - 滑音 (Glides)
        shapes['w'] = MouthShape(0.3, 0.2, 0.9, 0.1, 0.2, 0.3, 0.3, 0.0)  # /w/ 圆唇滑音
        shapes['j'] = MouthShape(0.2, 0.8, 0.0, 0.6, 0.9, 0.9, 0.2, 0.3)  # /j/ 如"衣"
        
        # 特殊音素
        shapes['sil'] = MouthShape(0.1, 0.5, 0.0, 0.0, 0.3, 0.3, 0.1, 0.0) # 静音
        shapes['sp'] = MouthShape(0.1, 0.5, 0.0, 0.0, 0.3, 0.3, 0.1, 0.0)  # 短暂停顿
        
        return shapes
    
    def _initialize_transition_weights(self) -> Dict[Tuple[str, str], float]:
        """初始化音素间转换权重"""
        weights = {}
        
        # 元音间转换较平滑
        vowels = ['a', 'e', 'i', 'o', 'u', 'ai', 'ei', 'ao', 'ou']
        for v1 in vowels:
            for v2 in vowels:
                weights[(v1, v2)] = 0.8
        
        # 辅音到元音转换
        consonants = ['p', 'b', 't', 'd', 'k', 'g', 'f', 'v', 's', 'z', 
                     'sh', 'zh', 'h', 'm', 'n', 'ng', 'l', 'r', 'w', 'j']
        for c in consonants:
            for v in vowels:
                weights[(c, v)] = 0.6
                weights[(v, c)] = 0.6
        
        # 辅音间转换较快
        for c1 in consonants:
            for c2 in consonants:
                weights[(c1, c2)] = 0.4
        
        # 静音转换
        for phoneme in list(vowels) + consonants:
            weights[('sil', phoneme)] = 0.3
            weights[(phoneme, 'sil')] = 0.3
            weights[('sp', phoneme)] = 0.2
            weights[(phoneme, 'sp')] = 0.2
        
        return weights
    
    def get_mouth_shape(self, phoneme: str) -> MouthShape:
        """获取音素对应的口型"""
        if phoneme in self.phoneme_shapes:
            return self.phoneme_shapes[phoneme]
        else:
            # 未知音素返回中性口型
            return MouthShape(0.3, 0.5, 0.2, 0.2, 0.4, 0.4, 0.3, 0.0)
    
    def interpolate_shapes(self, shape1: MouthShape, shape2: MouthShape, 
                          alpha: float) -> MouthShape:
        """在两个口型间插值"""
        tensor1 = shape1.to_tensor()
        tensor2 = shape2.to_tensor()
        interpolated = tensor1 * (1 - alpha) + tensor2 * alpha
        return MouthShape.from_tensor(interpolated)
    
    def get_transition_weight(self, phoneme1: str, phoneme2: str) -> float:
        """获取音素间转换权重"""
        key = (phoneme1, phoneme2)
        return self.transition_weights.get(key, 0.5)  # 默认权重

class PhonemeSequenceProcessor:
    """音素序列处理器"""
    
    def __init__(self, mapper: PhonemeToMouthMapper):
        self.mapper = mapper
        self.smoothing_window = 3  # 平滑窗口大小
    
    def process_phoneme_sequence(self, phonemes: List[str], 
                               durations: List[float]) -> List[MouthShape]:
        """处理音素序列，生成平滑的口型变化"""
        if len(phonemes) != len(durations):
            raise ValueError("音素数量与持续时间数量不匹配")
        
        mouth_shapes = []
        
        for i, (phoneme, duration) in enumerate(zip(phonemes, durations)):
            base_shape = self.mapper.get_mouth_shape(phoneme)
            
            # 考虑前后音素的影响
            if i > 0:
                prev_phoneme = phonemes[i-1]
                prev_shape = self.mapper.get_mouth_shape(prev_phoneme)
                transition_weight = self.mapper.get_transition_weight(prev_phoneme, phoneme)
                
                # 在音素开始时进行插值
                start_shape = self.mapper.interpolate_shapes(
                    prev_shape, base_shape, transition_weight
                )
            else:
                start_shape = base_shape
            
            if i < len(phonemes) - 1:
                next_phoneme = phonemes[i+1]
                next_shape = self.mapper.get_mouth_shape(next_phoneme)
                transition_weight = self.mapper.get_transition_weight(phoneme, next_phoneme)
                
                # 在音素结束时进行插值
                end_shape = self.mapper.interpolate_shapes(
                    base_shape, next_shape, transition_weight
                )
            else:
                end_shape = base_shape
            
            # 生成该音素持续时间内的口型序列
            num_frames = max(1, int(duration * 25))  # 假设25fps
            for frame in range(num_frames):
                alpha = frame / max(1, num_frames - 1)
                frame_shape = self.mapper.interpolate_shapes(start_shape, end_shape, alpha)
                mouth_shapes.append(frame_shape)
        
        # 应用时序平滑
        return self._apply_temporal_smoothing(mouth_shapes)
    
    def _apply_temporal_smoothing(self, shapes: List[MouthShape]) -> List[MouthShape]:
        """应用时序平滑"""
        if len(shapes) <= self.smoothing_window:
            return shapes
        
        smoothed_shapes = []
        half_window = self.smoothing_window // 2
        
        for i in range(len(shapes)):
            start_idx = max(0, i - half_window)
            end_idx = min(len(shapes), i + half_window + 1)
            
            # 计算窗口内的平均口型
            window_tensors = [shapes[j].to_tensor() for j in range(start_idx, end_idx)]
            avg_tensor = torch.stack(window_tensors).mean(dim=0)
            
            smoothed_shapes.append(MouthShape.from_tensor(avg_tensor))
        
        return smoothed_shapes

class DetailedMouthAnimationController:
    """精细嘴部动画控制器"""
    
    def __init__(self):
        self.mapper = PhonemeToMouthMapper()
        self.processor = PhonemeSequenceProcessor(self.mapper)
        
        # 微表情参数
        self.micro_expression_intensity = 0.1
        self.breathing_amplitude = 0.05
        self.natural_variation = 0.02
    
    def generate_mouth_animation(self, phonemes: List[str], 
                               durations: List[float],
                               add_micro_expressions: bool = True) -> torch.Tensor:
        """生成完整的嘴部动画序列"""
        # 处理音素序列
        mouth_shapes = self.processor.process_phoneme_sequence(phonemes, durations)
        
        # 转换为张量格式
        animation_tensor = torch.stack([shape.to_tensor() for shape in mouth_shapes])
        
        if add_micro_expressions:
            animation_tensor = self._add_micro_expressions(animation_tensor)
        
        return animation_tensor
    
    def _add_micro_expressions(self, animation: torch.Tensor) -> torch.Tensor:
        """添加微表情和自然变化"""
        num_frames = animation.shape[0]
        
        # 添加呼吸效果
        breathing_phase = torch.linspace(0, 4 * np.pi, num_frames)
        breathing_effect = self.breathing_amplitude * torch.sin(breathing_phase)
        
        # 嘴唇轻微开合变化
        animation[:, 0] += breathing_effect  # lip_opening
        
        # 添加自然随机变化
        natural_noise = torch.randn_like(animation) * self.natural_variation
        animation += natural_noise
        
        # 添加微表情（嘴角轻微变化）
        micro_expression_phase = torch.linspace(0, 2 * np.pi, num_frames)
        micro_expression = self.micro_expression_intensity * torch.sin(micro_expression_phase * 0.3)
        animation[:, 7] += micro_expression  # corner_pull
        
        # 确保参数在合理范围内
        animation = torch.clamp(animation, 0.0, 1.0)
        animation[:, 7] = torch.clamp(animation[:, 7], -1.0, 1.0)  # corner_pull可以为负
        
        return animation
    
    def convert_to_blendshape(self, mouth_animation: torch.Tensor) -> torch.Tensor:
        """将口型参数转换为BlendShape系数"""
        # 这里需要根据具体的BlendShape定义进行映射
        # 假设输出6个BlendShape参数（与原Audio2Feature模型兼容）
        
        batch_size, seq_len, _ = mouth_animation.shape
        blendshape_coeffs = torch.zeros(batch_size, seq_len, 6)
        
        # 映射规则（可根据实际BlendShape定义调整）
        blendshape_coeffs[:, :, 0] = mouth_animation[:, :, 0]  # 嘴唇开合 -> jaw_open
        blendshape_coeffs[:, :, 1] = mouth_animation[:, :, 1]  # 嘴唇宽度 -> mouth_stretch
        blendshape_coeffs[:, :, 2] = mouth_animation[:, :, 2]  # 嘴唇突出 -> mouth_pucker
        blendshape_coeffs[:, :, 3] = mouth_animation[:, :, 7]  # 嘴角拉伸 -> mouth_smile
        blendshape_coeffs[:, :, 4] = mouth_animation[:, :, 3]  # 牙齿显露 -> mouth_upper_lip
        blendshape_coeffs[:, :, 5] = mouth_animation[:, :, 6]  # 下颌开合 -> mouth_lower_lip
        
        return blendshape_coeffs

# 使用示例和测试函数
def test_phoneme_mapping():
    """测试音素映射功能"""
    controller = DetailedMouthAnimationController()
    
    # 测试音素序列："你好" (ni hao)
    phonemes = ['n', 'i', 'h', 'ao']
    durations = [0.1, 0.2, 0.1, 0.3]  # 秒
    
    # 生成动画
    animation = controller.generate_mouth_animation(phonemes, durations)
    print(f"生成动画序列形状: {animation.shape}")
    
    # 转换为BlendShape
    blendshapes = controller.convert_to_blendshape(animation.unsqueeze(0))
    print(f"BlendShape系数形状: {blendshapes.shape}")
    
    return animation, blendshapes

if __name__ == "__main__":
    # 运行测试
    animation, blendshapes = test_phoneme_mapping()
    print("音素到口型映射系统测试完成！")