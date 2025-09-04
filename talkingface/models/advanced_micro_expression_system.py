#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级微表情系统
实现更自然的面部肌肉细微活动和情感表达的同步控制
包含呼吸效应、眨眼模拟、微笑渐变等高级功能
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import cv2
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import butter, filtfilt
import random
import time
from enum import Enum
from dataclasses import dataclass
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionType(Enum):
    """扩展的情感类型枚举"""
    NEUTRAL = "neutral"          # 中性
    HAPPY = "happy"              # 快乐
    SAD = "sad"                  # 悲伤
    ANGRY = "angry"              # 愤怒
    SURPRISED = "surprised"      # 惊讶
    DISGUSTED = "disgusted"      # 厌恶
    FEARFUL = "fearful"          # 恐惧
    CONTEMPT = "contempt"        # 轻蔑
    EXCITED = "excited"          # 兴奋
    CONFUSED = "confused"        # 困惑
    THOUGHTFUL = "thoughtful"    # 沉思
    RELAXED = "relaxed"          # 放松

@dataclass
class MicroExpressionConfig:
    """微表情配置参数"""
    breathing_amplitude: float = 0.015      # 呼吸幅度
    breathing_frequency: float = 0.25       # 呼吸频率 (Hz)
    blink_frequency: float = 0.33           # 眨眼频率 (Hz)
    blink_duration: float = 0.15            # 眨眼持续时间 (s)
    micro_movement_amplitude: float = 0.008 # 微动幅度
    emotion_transition_speed: float = 0.5   # 情感过渡速度
    asymmetry_factor: float = 0.05          # 不对称因子
    natural_variation_strength: float = 0.03 # 自然变化强度

class AdvancedFacialMuscleGroup:
    """
    高级面部肌肉群模拟
    模拟真实的肌肉收缩和放松过程
    """
    
    def __init__(self, name: str, landmarks: List[int], 
                 activation_pattern: str = 'linear'):
        self.name = name
        self.landmarks = landmarks
        self.activation_pattern = activation_pattern
        self.current_activation = 0.0
        self.target_activation = 0.0
        self.activation_history = []
        
        # 肌肉特性参数
        self.contraction_speed = 0.8    # 收缩速度
        self.relaxation_speed = 0.6     # 放松速度
        self.fatigue_factor = 0.95      # 疲劳因子
        self.recovery_rate = 0.02       # 恢复速率
        
    def update_activation(self, target: float, dt: float = 1/25) -> float:
        """
        更新肌肉激活状态
        
        Args:
            target: 目标激活水平
            dt: 时间步长
            
        Returns:
            当前激活水平
        """
        self.target_activation = target
        
        # 计算激活变化速度
        if target > self.current_activation:
            # 收缩
            speed = self.contraction_speed
        else:
            # 放松
            speed = self.relaxation_speed
        
        # 应用疲劳效应
        if len(self.activation_history) > 10:
            avg_activation = np.mean(self.activation_history[-10:])
            if avg_activation > 0.7:
                speed *= self.fatigue_factor
        
        # 更新激活水平
        diff = target - self.current_activation
        self.current_activation += diff * speed * dt
        
        # 限制范围
        self.current_activation = np.clip(self.current_activation, 0.0, 1.0)
        
        # 记录历史
        self.activation_history.append(self.current_activation)
        if len(self.activation_history) > 50:
            self.activation_history.pop(0)
        
        return self.current_activation
    
    def get_displacement_vector(self, base_landmarks: np.ndarray) -> np.ndarray:
        """
        计算肌肉收缩产生的位移向量
        
        Args:
            base_landmarks: 基础关键点位置
            
        Returns:
            位移向量
        """
        displacement = np.zeros_like(base_landmarks)
        
        if self.activation_pattern == 'radial':
            # 径向收缩（如眼部肌肉）
            center = np.mean(base_landmarks[self.landmarks], axis=0)
            for idx in self.landmarks:
                direction = base_landmarks[idx] - center
                displacement[idx] = direction * self.current_activation * 0.1
        
        elif self.activation_pattern == 'linear':
            # 线性收缩（如嘴角肌肉）
            for i, idx in enumerate(self.landmarks):
                factor = (i / max(1, len(self.landmarks) - 1)) * 2 - 1  # -1 to 1
                displacement[idx, 1] = factor * self.current_activation * 0.05
        
        elif self.activation_pattern == 'lift':
            # 提升效应（如眉毛肌肉）
            for idx in self.landmarks:
                displacement[idx, 1] = -self.current_activation * 0.08
        
        return displacement

class BreathingSimulator:
    """
    呼吸效应模拟器
    模拟呼吸对面部的细微影响
    """
    
    def __init__(self, config: MicroExpressionConfig):
        self.config = config
        self.phase = 0.0
        self.breathing_pattern = 'normal'  # normal, deep, shallow
        
    def update_breathing(self, dt: float = 1/25) -> Dict[str, float]:
        """
        更新呼吸状态
        
        Args:
            dt: 时间步长
            
        Returns:
            呼吸参数
        """
        # 更新相位
        self.phase += 2 * np.pi * self.config.breathing_frequency * dt
        if self.phase > 2 * np.pi:
            self.phase -= 2 * np.pi
        
        # 计算呼吸强度
        base_intensity = np.sin(self.phase)
        
        # 根据呼吸模式调整
        if self.breathing_pattern == 'deep':
            intensity = base_intensity * 1.5
        elif self.breathing_pattern == 'shallow':
            intensity = base_intensity * 0.5
        else:
            intensity = base_intensity
        
        # 添加自然变化
        noise = np.random.normal(0, 0.1)
        intensity += noise
        
        return {
            'chest_expansion': intensity * self.config.breathing_amplitude,
            'nostril_flare': abs(intensity) * 0.3 * self.config.breathing_amplitude,
            'jaw_slight_drop': max(0, intensity) * 0.2 * self.config.breathing_amplitude
        }
    
    def set_breathing_pattern(self, pattern: str):
        """设置呼吸模式"""
        if pattern in ['normal', 'deep', 'shallow']:
            self.breathing_pattern = pattern

class BlinkSimulator:
    """
    眨眼模拟器
    实现自然的眨眼动作
    """
    
    def __init__(self, config: MicroExpressionConfig):
        self.config = config
        self.last_blink_time = 0.0
        self.current_blink_phase = 0.0
        self.is_blinking = False
        self.blink_type = 'normal'  # normal, slow, quick
        
    def update_blink(self, current_time: float) -> Dict[str, float]:
        """
        更新眨眼状态
        
        Args:
            current_time: 当前时间
            
        Returns:
            眨眼参数
        """
        # 检查是否需要开始新的眨眼
        time_since_last_blink = current_time - self.last_blink_time
        blink_interval = 1.0 / self.config.blink_frequency
        
        # 添加随机性
        random_factor = np.random.exponential(1.0)
        adjusted_interval = blink_interval * random_factor
        
        if not self.is_blinking and time_since_last_blink > adjusted_interval:
            self.is_blinking = True
            self.current_blink_phase = 0.0
            self.last_blink_time = current_time
        
        # 更新眨眼动画
        if self.is_blinking:
            # 眨眼动画曲线（快速闭合，慢速张开）
            progress = self.current_blink_phase / self.config.blink_duration
            
            if progress < 0.3:  # 闭合阶段
                closure = np.sin(progress / 0.3 * np.pi / 2)
            elif progress < 0.7:  # 完全闭合
                closure = 1.0
            else:  # 张开阶段
                closure = np.cos((progress - 0.7) / 0.3 * np.pi / 2)
            
            self.current_blink_phase += 1/25  # 假设25fps
            
            if self.current_blink_phase >= self.config.blink_duration:
                self.is_blinking = False
                closure = 0.0
        else:
            closure = 0.0
        
        return {
            'left_eye_closure': closure,
            'right_eye_closure': closure * (0.95 + np.random.normal(0, 0.02)),  # 轻微不对称
            'upper_lid_tension': closure * 0.3,
            'lower_lid_lift': closure * 0.1
        }

class EmotionTransitionManager:
    """
    情感过渡管理器
    处理情感之间的平滑过渡
    """
    
    def __init__(self, config: MicroExpressionConfig):
        self.config = config
        self.current_emotion = EmotionType.NEUTRAL
        self.target_emotion = EmotionType.NEUTRAL
        self.transition_progress = 1.0
        self.transition_duration = 2.0  # 默认过渡时间
        
    def set_target_emotion(self, emotion: EmotionType, duration: float = None):
        """
        设置目标情感
        
        Args:
            emotion: 目标情感
            duration: 过渡持续时间
        """
        if emotion != self.target_emotion:
            self.current_emotion = self.target_emotion
            self.target_emotion = emotion
            self.transition_progress = 0.0
            self.transition_duration = duration or (2.0 / self.config.emotion_transition_speed)
    
    def update_transition(self, dt: float = 1/25) -> Tuple[EmotionType, EmotionType, float]:
        """
        更新情感过渡
        
        Args:
            dt: 时间步长
            
        Returns:
            (当前情感, 目标情感, 过渡进度)
        """
        if self.transition_progress < 1.0:
            self.transition_progress += dt / self.transition_duration
            self.transition_progress = min(1.0, self.transition_progress)
        
        return self.current_emotion, self.target_emotion, self.transition_progress
    
    def get_blended_intensity(self, current_intensity: float, target_intensity: float) -> float:
        """
        获取混合后的情感强度
        
        Args:
            current_intensity: 当前情感强度
            target_intensity: 目标情感强度
            
        Returns:
            混合强度
        """
        # 使用平滑的过渡曲线
        smooth_progress = 3 * self.transition_progress**2 - 2 * self.transition_progress**3
        return current_intensity * (1 - smooth_progress) + target_intensity * smooth_progress

class AdvancedMicroExpressionSystem:
    """
    高级微表情系统主类
    整合所有微表情功能
    """
    
    def __init__(self, config: MicroExpressionConfig = None):
        self.config = config or MicroExpressionConfig()
        
        # 初始化子系统
        self.breathing_simulator = BreathingSimulator(self.config)
        self.blink_simulator = BlinkSimulator(self.config)
        self.emotion_manager = EmotionTransitionManager(self.config)
        
        # 面部区域定义（68点标记）
        self.facial_regions = {
            'left_eyebrow': [17, 18, 19, 20, 21],
            'right_eyebrow': [22, 23, 24, 25, 26],
            'left_eye': [36, 37, 38, 39, 40, 41],
            'right_eye': [42, 43, 44, 45, 46, 47],
            'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35],
            'mouth': list(range(48, 68)),
            'left_cheek': [1, 2, 3, 31, 49],
            'right_cheek': [13, 14, 15, 35, 53],
            'jaw': list(range(0, 17)),
            'forehead': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
        }
        
        # 初始化肌肉群
        self.muscle_groups = self._initialize_muscle_groups()
        
        # 情感到微表情的映射
        self.emotion_micro_mapping = self._create_advanced_emotion_mapping()
        
        # 时间跟踪
        self.current_time = 0.0
        
        logger.info("高级微表情系统初始化完成")
    
    def _initialize_muscle_groups(self) -> Dict[str, AdvancedFacialMuscleGroup]:
        """
        初始化面部肌肉群
        
        Returns:
            肌肉群字典
        """
        muscle_groups = {}
        
        # 眼部肌肉
        muscle_groups['orbicularis_oculi_left'] = AdvancedFacialMuscleGroup(
            'orbicularis_oculi_left', self.facial_regions['left_eye'], 'radial'
        )
        muscle_groups['orbicularis_oculi_right'] = AdvancedFacialMuscleGroup(
            'orbicularis_oculi_right', self.facial_regions['right_eye'], 'radial'
        )
        
        # 眉毛肌肉
        muscle_groups['frontalis_left'] = AdvancedFacialMuscleGroup(
            'frontalis_left', self.facial_regions['left_eyebrow'], 'lift'
        )
        muscle_groups['frontalis_right'] = AdvancedFacialMuscleGroup(
            'frontalis_right', self.facial_regions['right_eyebrow'], 'lift'
        )
        
        # 嘴部肌肉
        muscle_groups['zygomaticus_major'] = AdvancedFacialMuscleGroup(
            'zygomaticus_major', [48, 49, 50, 51, 52, 53, 54], 'linear'
        )
        muscle_groups['depressor_anguli_oris'] = AdvancedFacialMuscleGroup(
            'depressor_anguli_oris', [54, 55, 56, 57, 58, 59, 60], 'linear'
        )
        
        # 鼻部肌肉
        muscle_groups['nasalis'] = AdvancedFacialMuscleGroup(
            'nasalis', [31, 32, 33, 34, 35], 'radial'
        )
        
        return muscle_groups
    
    def _create_advanced_emotion_mapping(self) -> Dict[EmotionType, Dict[str, float]]:
        """
        创建高级情感到微表情的映射
        
        Returns:
            情感映射字典
        """
        return {
            EmotionType.NEUTRAL: {
                'eyebrow_raise': 0.0,
                'eye_squint': 0.0,
                'nose_wrinkle': 0.0,
                'mouth_corner_pull': 0.0,
                'jaw_drop': 0.0,
                'cheek_raise': 0.0,
                'forehead_wrinkle': 0.0,
                'asymmetry': 0.02
            },
            EmotionType.HAPPY: {
                'eyebrow_raise': 0.15,
                'eye_squint': 0.25,
                'nose_wrinkle': 0.1,
                'mouth_corner_pull': 0.8,
                'jaw_drop': 0.1,
                'cheek_raise': 0.6,
                'forehead_wrinkle': 0.0,
                'asymmetry': 0.03
            },
            EmotionType.SAD: {
                'eyebrow_raise': 0.4,
                'eye_squint': 0.1,
                'nose_wrinkle': 0.0,
                'mouth_corner_pull': -0.5,
                'jaw_drop': 0.2,
                'cheek_raise': 0.0,
                'forehead_wrinkle': 0.3,
                'asymmetry': 0.04
            },
            EmotionType.ANGRY: {
                'eyebrow_raise': -0.3,
                'eye_squint': 0.6,
                'nose_wrinkle': 0.4,
                'mouth_corner_pull': -0.2,
                'jaw_drop': 0.0,
                'cheek_raise': 0.0,
                'forehead_wrinkle': 0.5,
                'asymmetry': 0.02
            },
            EmotionType.SURPRISED: {
                'eyebrow_raise': 0.8,
                'eye_squint': -0.3,
                'nose_wrinkle': 0.0,
                'mouth_corner_pull': 0.0,
                'jaw_drop': 0.6,
                'cheek_raise': 0.0,
                'forehead_wrinkle': 0.2,
                'asymmetry': 0.01
            },
            EmotionType.DISGUSTED: {
                'eyebrow_raise': -0.2,
                'eye_squint': 0.3,
                'nose_wrinkle': 0.7,
                'mouth_corner_pull': -0.3,
                'jaw_drop': 0.0,
                'cheek_raise': 0.2,
                'forehead_wrinkle': 0.1,
                'asymmetry': 0.03
            },
            EmotionType.FEARFUL: {
                'eyebrow_raise': 0.6,
                'eye_squint': -0.2,
                'nose_wrinkle': 0.1,
                'mouth_corner_pull': -0.1,
                'jaw_drop': 0.3,
                'cheek_raise': 0.0,
                'forehead_wrinkle': 0.4,
                'asymmetry': 0.05
            },
            EmotionType.EXCITED: {
                'eyebrow_raise': 0.3,
                'eye_squint': 0.2,
                'nose_wrinkle': 0.0,
                'mouth_corner_pull': 0.9,
                'jaw_drop': 0.2,
                'cheek_raise': 0.7,
                'forehead_wrinkle': 0.0,
                'asymmetry': 0.04
            },
            EmotionType.THOUGHTFUL: {
                'eyebrow_raise': 0.2,
                'eye_squint': 0.1,
                'nose_wrinkle': 0.0,
                'mouth_corner_pull': 0.0,
                'jaw_drop': 0.0,
                'cheek_raise': 0.0,
                'forehead_wrinkle': 0.2,
                'asymmetry': 0.03
            },
            EmotionType.RELAXED: {
                'eyebrow_raise': 0.0,
                'eye_squint': 0.05,
                'nose_wrinkle': 0.0,
                'mouth_corner_pull': 0.1,
                'jaw_drop': 0.05,
                'cheek_raise': 0.1,
                'forehead_wrinkle': 0.0,
                'asymmetry': 0.01
            }
        }
    
    def update_frame(self, dt: float = 1/25) -> Dict[str, float]:
        """
        更新一帧的微表情参数
        
        Args:
            dt: 时间步长
            
        Returns:
            微表情参数字典
        """
        self.current_time += dt
        
        # 更新呼吸效应
        breathing_params = self.breathing_simulator.update_breathing(dt)
        
        # 更新眨眼
        blink_params = self.blink_simulator.update_blink(self.current_time)
        
        # 更新情感过渡
        current_emotion, target_emotion, transition_progress = self.emotion_manager.update_transition(dt)
        
        # 获取基础情感参数
        current_emotion_params = self.emotion_micro_mapping[current_emotion]
        target_emotion_params = self.emotion_micro_mapping[target_emotion]
        
        # 混合情感参数
        blended_params = {}
        for key in current_emotion_params:
            current_val = current_emotion_params[key]
            target_val = target_emotion_params[key]
            blended_params[key] = self.emotion_manager.get_blended_intensity(current_val, target_val)
        
        # 更新肌肉群激活
        muscle_activations = {}
        for name, muscle in self.muscle_groups.items():
            if 'eye' in name:
                target = blended_params['eye_squint'] + blink_params.get('left_eye_closure', 0)
            elif 'eyebrow' in name or 'frontalis' in name:
                target = blended_params['eyebrow_raise']
            elif 'zygomaticus' in name:
                target = max(0, blended_params['mouth_corner_pull'])
            elif 'depressor' in name:
                target = max(0, -blended_params['mouth_corner_pull'])
            elif 'nasalis' in name:
                target = blended_params['nose_wrinkle'] + breathing_params['nostril_flare']
            else:
                target = 0.0
            
            muscle_activations[name] = muscle.update_activation(target, dt)
        
        # 添加自然变化
        natural_variations = self._generate_natural_variations()
        
        # 合并所有参数
        final_params = {
            **blended_params,
            **breathing_params,
            **blink_params,
            **muscle_activations,
            **natural_variations,
            'current_emotion': current_emotion.value,
            'target_emotion': target_emotion.value,
            'transition_progress': transition_progress,
            'timestamp': self.current_time
        }
        
        return final_params
    
    def _generate_natural_variations(self) -> Dict[str, float]:
        """
        生成自然的微小变化
        
        Returns:
            自然变化参数
        """
        variations = {}
        
        # 微小的随机移动
        for region in ['left_eye', 'right_eye', 'mouth', 'nose']:
            variations[f'{region}_micro_x'] = np.random.normal(0, self.config.micro_movement_amplitude)
            variations[f'{region}_micro_y'] = np.random.normal(0, self.config.micro_movement_amplitude)
        
        # 不对称变化
        variations['left_right_asymmetry'] = np.random.normal(0, self.config.asymmetry_factor)
        
        # 整体面部张力
        variations['facial_tension'] = np.random.uniform(0, self.config.natural_variation_strength)
        
        return variations
    
    def apply_to_landmarks(self, landmarks: np.ndarray, micro_params: Dict[str, float]) -> np.ndarray:
        """
        将微表情参数应用到关键点
        
        Args:
            landmarks: 原始关键点 [68, 2]
            micro_params: 微表情参数
            
        Returns:
            调整后的关键点
        """
        adjusted_landmarks = landmarks.copy()
        
        # 应用肌肉群效应
        for name, muscle in self.muscle_groups.items():
            if name in micro_params:
                displacement = muscle.get_displacement_vector(adjusted_landmarks)
                adjusted_landmarks += displacement * micro_params[name]
        
        # 应用呼吸效应
        if 'chest_expansion' in micro_params:
            # 轻微的整体缩放
            center = np.mean(adjusted_landmarks, axis=0)
            scale_factor = 1.0 + micro_params['chest_expansion'] * 0.01
            adjusted_landmarks = (adjusted_landmarks - center) * scale_factor + center
        
        # 应用眨眼效应
        if 'left_eye_closure' in micro_params:
            self._apply_eye_closure(adjusted_landmarks, 'left', micro_params['left_eye_closure'])
        if 'right_eye_closure' in micro_params:
            self._apply_eye_closure(adjusted_landmarks, 'right', micro_params['right_eye_closure'])
        
        # 应用微小移动
        for region in ['left_eye', 'right_eye', 'mouth', 'nose']:
            if f'{region}_micro_x' in micro_params and f'{region}_micro_y' in micro_params:
                region_indices = self.facial_regions.get(region, [])
                for idx in region_indices:
                    adjusted_landmarks[idx, 0] += micro_params[f'{region}_micro_x']
                    adjusted_landmarks[idx, 1] += micro_params[f'{region}_micro_y']
        
        # 应用不对称效应
        if 'left_right_asymmetry' in micro_params:
            asymmetry = micro_params['left_right_asymmetry']
            center_x = np.mean(adjusted_landmarks[:, 0])
            
            # 左侧点向左偏移，右侧点向右偏移
            for i, landmark in enumerate(adjusted_landmarks):
                if landmark[0] < center_x:  # 左侧
                    adjusted_landmarks[i, 0] -= asymmetry
                else:  # 右侧
                    adjusted_landmarks[i, 0] += asymmetry
        
        return adjusted_landmarks
    
    def _apply_eye_closure(self, landmarks: np.ndarray, side: str, closure_amount: float):
        """
        应用眼部闭合效果
        
        Args:
            landmarks: 关键点数组
            side: 'left' 或 'right'
            closure_amount: 闭合程度 [0, 1]
        """
        if side == 'left':
            eye_indices = self.facial_regions['left_eye']
        else:
            eye_indices = self.facial_regions['right_eye']
        
        # 计算眼部中心
        eye_center = np.mean(landmarks[eye_indices], axis=0)
        
        # 上下眼睑向中心靠拢
        for idx in eye_indices:
            direction = eye_center - landmarks[idx]
            landmarks[idx] += direction * closure_amount * 0.3
    
    def set_emotion(self, emotion: EmotionType, duration: float = None):
        """
        设置目标情感
        
        Args:
            emotion: 目标情感
            duration: 过渡持续时间
        """
        self.emotion_manager.set_target_emotion(emotion, duration)
        logger.info(f"设置目标情感: {emotion.value}")
    
    def set_breathing_pattern(self, pattern: str):
        """
        设置呼吸模式
        
        Args:
            pattern: 呼吸模式 ('normal', 'deep', 'shallow')
        """
        self.breathing_simulator.set_breathing_pattern(pattern)
        logger.info(f"设置呼吸模式: {pattern}")
    
    def get_system_state(self) -> Dict[str, any]:
        """
        获取系统状态信息
        
        Returns:
            系统状态字典
        """
        return {
            'current_time': self.current_time,
            'current_emotion': self.emotion_manager.current_emotion.value,
            'target_emotion': self.emotion_manager.target_emotion.value,
            'transition_progress': self.emotion_manager.transition_progress,
            'breathing_pattern': self.breathing_simulator.breathing_pattern,
            'is_blinking': self.blink_simulator.is_blinking,
            'muscle_activations': {name: muscle.current_activation 
                                 for name, muscle in self.muscle_groups.items()}
        }
    
    def reset_system(self):
        """
        重置系统状态
        """
        self.current_time = 0.0
        self.emotion_manager.current_emotion = EmotionType.NEUTRAL
        self.emotion_manager.target_emotion = EmotionType.NEUTRAL
        self.emotion_manager.transition_progress = 1.0
        
        for muscle in self.muscle_groups.values():
            muscle.current_activation = 0.0
            muscle.target_activation = 0.0
            muscle.activation_history.clear()
        
        logger.info("微表情系统已重置")

# 使用示例
if __name__ == "__main__":
    # 创建配置
    config = MicroExpressionConfig(
        breathing_amplitude=0.02,
        blink_frequency=0.4,
        natural_variation_strength=0.05
    )
    
    # 创建系统
    micro_system = AdvancedMicroExpressionSystem(config)
    
    # 模拟一段时间的运行
    print("开始微表情模拟...")
    
    # 设置情感序列
    emotions = [
        (EmotionType.NEUTRAL, 2.0),
        (EmotionType.HAPPY, 3.0),
        (EmotionType.SURPRISED, 1.5),
        (EmotionType.THOUGHTFUL, 2.5),
        (EmotionType.RELAXED, 2.0)
    ]
    
    current_emotion_idx = 0
    emotion_start_time = 0.0
    
    # 模拟25fps，总共10秒
    for frame in range(250):
        current_time = frame / 25.0
        
        # 检查是否需要切换情感
        if current_emotion_idx < len(emotions):
            emotion, duration = emotions[current_emotion_idx]
            if current_time - emotion_start_time >= duration:
                if current_emotion_idx + 1 < len(emotions):
                    next_emotion, _ = emotions[current_emotion_idx + 1]
                    micro_system.set_emotion(next_emotion, 1.5)
                    current_emotion_idx += 1
                    emotion_start_time = current_time
        
        # 更新微表情
        micro_params = micro_system.update_frame(1/25)
        
        # 每秒输出一次状态
        if frame % 25 == 0:
            state = micro_system.get_system_state()
            print(f"时间: {current_time:.1f}s, 情感: {state['current_emotion']} -> {state['target_emotion']} ({state['transition_progress']:.2f})")
            print(f"  呼吸: {state['breathing_pattern']}, 眨眼: {state['is_blinking']}")
            print(f"  主要肌肉激活: {list(state['muscle_activations'].values())[:3]}")
    
    print("微表情模拟完成")
    
    # 测试关键点应用
    print("\n测试关键点应用...")
    dummy_landmarks = np.random.randn(68, 2) * 10 + 100  # 模拟关键点
    
    micro_system.set_emotion(EmotionType.HAPPY)
    for _ in range(5):
        micro_params = micro_system.update_frame()
        adjusted_landmarks = micro_system.apply_to_landmarks(dummy_landmarks, micro_params)
        displacement = np.mean(np.abs(adjusted_landmarks - dummy_landmarks))
        print(f"平均位移: {displacement:.4f}")
    
    print("测试完成")