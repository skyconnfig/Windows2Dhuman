# -*- coding: utf-8 -*-
"""
面部微表情系统
实现面部肌肉细微活动和情感表达的同步控制
包括眼部、眉毛、脸颊等区域的微妙变化
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.interpolate import interp1d
import random
from enum import Enum

class EmotionType(Enum):
    """情感类型枚举"""
    NEUTRAL = "neutral"      # 中性
    HAPPY = "happy"          # 快乐
    SAD = "sad"              # 悲伤
    ANGRY = "angry"          # 愤怒
    SURPRISED = "surprised"  # 惊讶
    DISGUSTED = "disgusted"  # 厌恶
    FEARFUL = "fearful"      # 恐惧
    CONTEMPT = "contempt"    # 轻蔑

class MicroExpressionSystem:
    """
    面部微表情系统
    控制面部各区域的细微表情变化
    """
    
    def __init__(self):
        # 面部区域关键点映射（基于68点标记）
        self.facial_regions = {
            'left_eyebrow': [17, 18, 19, 20, 21],      # 左眉毛
            'right_eyebrow': [22, 23, 24, 25, 26],     # 右眉毛
            'left_eye': [36, 37, 38, 39, 40, 41],      # 左眼
            'right_eye': [42, 43, 44, 45, 46, 47],     # 右眼
            'nose': [27, 28, 29, 30, 31, 32, 33, 34, 35],  # 鼻子
            'mouth': list(range(48, 68)),               # 嘴部
            'left_cheek': [1, 2, 3, 31, 49],          # 左脸颊
            'right_cheek': [13, 14, 15, 35, 53],       # 右脸颊
            'jaw': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 下颌
            'forehead': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]  # 前额
        }
        
        # 微表情参数定义
        self.micro_expression_params = {
            # 眼部参数
            'left_eye_openness': 1.0,      # 左眼睁开程度 [0-1]
            'right_eye_openness': 1.0,     # 右眼睁开程度 [0-1]
            'left_eye_squint': 0.0,        # 左眼眯眼程度 [0-1]
            'right_eye_squint': 0.0,       # 右眼眯眼程度 [0-1]
            'eye_gaze_x': 0.0,             # 眼球水平注视 [-1,1]
            'eye_gaze_y': 0.0,             # 眼球垂直注视 [-1,1]
            
            # 眉毛参数
            'left_eyebrow_raise': 0.0,     # 左眉抬起 [0-1]
            'right_eyebrow_raise': 0.0,    # 右眉抬起 [0-1]
            'left_eyebrow_furrow': 0.0,    # 左眉皱起 [0-1]
            'right_eyebrow_furrow': 0.0,   # 右眉皱起 [0-1]
            'eyebrow_asymmetry': 0.0,      # 眉毛不对称 [-1,1]
            
            # 鼻子参数
            'nostril_flare': 0.0,          # 鼻翼张开 [0-1]
            'nose_wrinkle': 0.0,           # 鼻子皱纹 [0-1]
            
            # 脸颊参数
            'left_cheek_raise': 0.0,       # 左脸颊抬起 [0-1]
            'right_cheek_raise': 0.0,      # 右脸颊抬起 [0-1]
            'cheek_puff': 0.0,             # 脸颊鼓起 [0-1]
            
            # 下颌参数
            'jaw_clench': 0.0,             # 下颌紧咬 [0-1]
            'jaw_shift_x': 0.0,            # 下颌水平移动 [-1,1]
            
            # 前额参数
            'forehead_wrinkle': 0.0,       # 前额皱纹 [0-1]
            
            # 整体参数
            'facial_tension': 0.0,         # 面部紧张度 [0-1]
            'asymmetry_factor': 0.0        # 整体不对称因子 [-1,1]
        }
        
        # 情感到微表情的映射
        self.emotion_micro_mapping = self._create_emotion_micro_mapping()
        
        # 自然变化参数
        self.natural_variation = {
            'blink_frequency': 0.2,        # 眨眼频率（每秒）
            'micro_movement_amplitude': 0.02,  # 微动幅度
            'breathing_influence': 0.01,   # 呼吸对表情的影响
            'random_seed': 42
        }
        
        # 历史状态记录
        self.expression_history = []
        self.last_blink_frame = 0
        
        # 设置随机种子
        np.random.seed(self.natural_variation['random_seed'])
        
    def _create_emotion_micro_mapping(self) -> Dict[EmotionType, Dict[str, float]]:
        """
        创建情感到微表情参数的映射
        """
        mapping = {
            EmotionType.NEUTRAL: {
                'left_eye_openness': 1.0, 'right_eye_openness': 1.0,
                'left_eye_squint': 0.0, 'right_eye_squint': 0.0,
                'left_eyebrow_raise': 0.0, 'right_eyebrow_raise': 0.0,
                'left_eyebrow_furrow': 0.0, 'right_eyebrow_furrow': 0.0,
                'nostril_flare': 0.0, 'nose_wrinkle': 0.0,
                'left_cheek_raise': 0.0, 'right_cheek_raise': 0.0,
                'jaw_clench': 0.0, 'forehead_wrinkle': 0.0,
                'facial_tension': 0.0
            },
            
            EmotionType.HAPPY: {
                'left_eye_openness': 0.8, 'right_eye_openness': 0.8,
                'left_eye_squint': 0.3, 'right_eye_squint': 0.3,
                'left_eyebrow_raise': 0.2, 'right_eyebrow_raise': 0.2,
                'left_cheek_raise': 0.7, 'right_cheek_raise': 0.7,
                'nostril_flare': 0.1, 'facial_tension': 0.2
            },
            
            EmotionType.SAD: {
                'left_eye_openness': 0.6, 'right_eye_openness': 0.6,
                'left_eyebrow_raise': 0.0, 'right_eyebrow_raise': 0.0,
                'left_eyebrow_furrow': 0.4, 'right_eyebrow_furrow': 0.4,
                'left_cheek_raise': 0.0, 'right_cheek_raise': 0.0,
                'jaw_clench': 0.1, 'facial_tension': 0.3
            },
            
            EmotionType.ANGRY: {
                'left_eye_openness': 0.9, 'right_eye_openness': 0.9,
                'left_eye_squint': 0.2, 'right_eye_squint': 0.2,
                'left_eyebrow_furrow': 0.8, 'right_eyebrow_furrow': 0.8,
                'nostril_flare': 0.6, 'nose_wrinkle': 0.3,
                'jaw_clench': 0.7, 'forehead_wrinkle': 0.4,
                'facial_tension': 0.8
            },
            
            EmotionType.SURPRISED: {
                'left_eye_openness': 1.0, 'right_eye_openness': 1.0,
                'left_eyebrow_raise': 0.8, 'right_eyebrow_raise': 0.8,
                'forehead_wrinkle': 0.6, 'nostril_flare': 0.3,
                'facial_tension': 0.4
            },
            
            EmotionType.DISGUSTED: {
                'left_eye_squint': 0.5, 'right_eye_squint': 0.5,
                'nose_wrinkle': 0.7, 'nostril_flare': 0.2,
                'left_cheek_raise': 0.3, 'right_cheek_raise': 0.3,
                'facial_tension': 0.5
            },
            
            EmotionType.FEARFUL: {
                'left_eye_openness': 1.0, 'right_eye_openness': 1.0,
                'left_eyebrow_raise': 0.6, 'right_eyebrow_raise': 0.6,
                'left_eyebrow_furrow': 0.3, 'right_eyebrow_furrow': 0.3,
                'jaw_clench': 0.4, 'facial_tension': 0.7
            },
            
            EmotionType.CONTEMPT: {
                'left_eye_squint': 0.3, 'right_eye_squint': 0.1,
                'left_cheek_raise': 0.2, 'right_cheek_raise': 0.0,
                'asymmetry_factor': 0.4, 'facial_tension': 0.3
            }
        }
        
        return mapping
    
    def get_emotion_micro_params(self, 
                               emotion: EmotionType, 
                               intensity: float = 1.0) -> Dict[str, float]:
        """
        获取指定情感的微表情参数
        
        Args:
            emotion: 情感类型
            intensity: 情感强度 [0-1]
            
        Returns:
            微表情参数字典
        """
        base_params = self.micro_expression_params.copy()
        
        if emotion in self.emotion_micro_mapping:
            emotion_params = self.emotion_micro_mapping[emotion]
            
            # 应用情感参数和强度
            for key, value in emotion_params.items():
                if key in base_params:
                    base_params[key] = value * intensity
        
        return base_params
    
    def add_natural_variations(self, 
                             micro_params: Dict[str, float], 
                             frame_index: int) -> Dict[str, float]:
        """
        添加自然变化到微表情参数
        
        Args:
            micro_params: 基础微表情参数
            frame_index: 当前帧索引
            
        Returns:
            添加自然变化后的参数
        """
        modified_params = micro_params.copy()
        
        # 添加眨眼效果
        blink_interval = int(25 / self.natural_variation['blink_frequency'])  # 25fps
        if frame_index - self.last_blink_frame > blink_interval:
            # 随机决定是否眨眼
            if np.random.random() < 0.3:
                self.last_blink_frame = frame_index
                # 眨眼持续3-5帧
                blink_duration = np.random.randint(3, 6)
                if frame_index - self.last_blink_frame < blink_duration:
                    blink_factor = 1.0 - abs(frame_index - self.last_blink_frame - blink_duration/2) / (blink_duration/2)
                    modified_params['left_eye_openness'] *= (1.0 - blink_factor * 0.9)
                    modified_params['right_eye_openness'] *= (1.0 - blink_factor * 0.9)
        
        # 添加微小随机运动
        amplitude = self.natural_variation['micro_movement_amplitude']
        for key in ['left_eyebrow_raise', 'right_eyebrow_raise', 'eye_gaze_x', 'eye_gaze_y']:
            if key in modified_params:
                noise = (np.random.random() - 0.5) * 2 * amplitude
                modified_params[key] += noise
        
        # 添加呼吸影响
        breathing_phase = np.sin(frame_index * 0.1)  # 慢呼吸
        breathing_amplitude = self.natural_variation['breathing_influence']
        
        modified_params['nostril_flare'] += breathing_phase * breathing_amplitude
        modified_params['facial_tension'] += abs(breathing_phase) * breathing_amplitude * 0.5
        
        # 确保参数在有效范围内
        for key, value in modified_params.items():
            if 'openness' in key or 'raise' in key or 'squint' in key or 'furrow' in key or \
               'flare' in key or 'wrinkle' in key or 'clench' in key or 'puff' in key or 'tension' in key:
                modified_params[key] = np.clip(value, 0.0, 1.0)
            else:
                modified_params[key] = np.clip(value, -1.0, 1.0)
        
        return modified_params
    
    def apply_micro_expressions_to_landmarks(self, 
                                           landmarks: np.ndarray, 
                                           micro_params: Dict[str, float]) -> np.ndarray:
        """
        将微表情参数应用到人脸关键点
        
        Args:
            landmarks: 原始68点人脸关键点 [68, 2]
            micro_params: 微表情参数
            
        Returns:
            变换后的关键点
        """
        modified_landmarks = landmarks.copy()
        
        # 应用眼部变化
        self._apply_eye_changes(modified_landmarks, micro_params)
        
        # 应用眉毛变化
        self._apply_eyebrow_changes(modified_landmarks, micro_params)
        
        # 应用鼻子变化
        self._apply_nose_changes(modified_landmarks, micro_params)
        
        # 应用脸颊变化
        self._apply_cheek_changes(modified_landmarks, micro_params)
        
        # 应用下颌变化
        self._apply_jaw_changes(modified_landmarks, micro_params)
        
        # 应用前额变化
        self._apply_forehead_changes(modified_landmarks, micro_params)
        
        # 应用整体不对称
        self._apply_asymmetry(modified_landmarks, micro_params)
        
        return modified_landmarks
    
    def _apply_eye_changes(self, landmarks: np.ndarray, params: Dict[str, float]):
        """
        应用眼部变化
        """
        # 左眼睁开程度
        left_eye_openness = params.get('left_eye_openness', 1.0)
        left_eye_indices = self.facial_regions['left_eye']
        left_eye_center = np.mean(landmarks[left_eye_indices], axis=0)
        
        for idx in left_eye_indices:
            # 垂直方向调整
            offset_y = landmarks[idx, 1] - left_eye_center[1]
            landmarks[idx, 1] = left_eye_center[1] + offset_y * left_eye_openness
        
        # 右眼睁开程度
        right_eye_openness = params.get('right_eye_openness', 1.0)
        right_eye_indices = self.facial_regions['right_eye']
        right_eye_center = np.mean(landmarks[right_eye_indices], axis=0)
        
        for idx in right_eye_indices:
            offset_y = landmarks[idx, 1] - right_eye_center[1]
            landmarks[idx, 1] = right_eye_center[1] + offset_y * right_eye_openness
        
        # 眯眼效果
        left_squint = params.get('left_eye_squint', 0.0)
        if left_squint > 0:
            # 下眼睑上移
            landmarks[41, 1] -= left_squint * 2
            landmarks[40, 1] -= left_squint * 1.5
        
        right_squint = params.get('right_eye_squint', 0.0)
        if right_squint > 0:
            landmarks[46, 1] -= right_squint * 2
            landmarks[47, 1] -= right_squint * 1.5
    
    def _apply_eyebrow_changes(self, landmarks: np.ndarray, params: Dict[str, float]):
        """
        应用眉毛变化
        """
        # 左眉抬起
        left_raise = params.get('left_eyebrow_raise', 0.0)
        for idx in self.facial_regions['left_eyebrow']:
            landmarks[idx, 1] -= left_raise * 5
        
        # 右眉抬起
        right_raise = params.get('right_eyebrow_raise', 0.0)
        for idx in self.facial_regions['right_eyebrow']:
            landmarks[idx, 1] -= right_raise * 5
        
        # 左眉皱起
        left_furrow = params.get('left_eyebrow_furrow', 0.0)
        if left_furrow > 0:
            # 眉毛内侧向下向内
            landmarks[21, 1] += left_furrow * 3
            landmarks[21, 0] += left_furrow * 2
        
        # 右眉皱起
        right_furrow = params.get('right_eyebrow_furrow', 0.0)
        if right_furrow > 0:
            landmarks[22, 1] += right_furrow * 3
            landmarks[22, 0] -= right_furrow * 2
    
    def _apply_nose_changes(self, landmarks: np.ndarray, params: Dict[str, float]):
        """
        应用鼻子变化
        """
        # 鼻翼张开
        nostril_flare = params.get('nostril_flare', 0.0)
        if nostril_flare > 0:
            landmarks[31, 0] -= nostril_flare * 2  # 左鼻翼
            landmarks[35, 0] += nostril_flare * 2  # 右鼻翼
        
        # 鼻子皱纹（通过鼻梁点的微调表现）
        nose_wrinkle = params.get('nose_wrinkle', 0.0)
        if nose_wrinkle > 0:
            landmarks[27, 1] -= nose_wrinkle * 1
            landmarks[28, 1] -= nose_wrinkle * 0.5
    
    def _apply_cheek_changes(self, landmarks: np.ndarray, params: Dict[str, float]):
        """
        应用脸颊变化
        """
        # 左脸颊抬起
        left_cheek_raise = params.get('left_cheek_raise', 0.0)
        for idx in self.facial_regions['left_cheek']:
            landmarks[idx, 1] -= left_cheek_raise * 3
        
        # 右脸颊抬起
        right_cheek_raise = params.get('right_cheek_raise', 0.0)
        for idx in self.facial_regions['right_cheek']:
            landmarks[idx, 1] -= right_cheek_raise * 3
        
        # 脸颊鼓起
        cheek_puff = params.get('cheek_puff', 0.0)
        if cheek_puff > 0:
            # 脸颊向外扩张
            for idx in self.facial_regions['left_cheek']:
                landmarks[idx, 0] -= cheek_puff * 2
            for idx in self.facial_regions['right_cheek']:
                landmarks[idx, 0] += cheek_puff * 2
    
    def _apply_jaw_changes(self, landmarks: np.ndarray, params: Dict[str, float]):
        """
        应用下颌变化
        """
        # 下颌紧咬
        jaw_clench = params.get('jaw_clench', 0.0)
        if jaw_clench > 0:
            # 下颌线条更加紧绷
            jaw_indices = self.facial_regions['jaw']
            for idx in jaw_indices[3:14]:  # 下颌底部
                landmarks[idx, 1] -= jaw_clench * 2
        
        # 下颌水平移动
        jaw_shift_x = params.get('jaw_shift_x', 0.0)
        if jaw_shift_x != 0:
            jaw_indices = self.facial_regions['jaw']
            for idx in jaw_indices[5:12]:  # 下颌中部
                landmarks[idx, 0] += jaw_shift_x * 3
    
    def _apply_forehead_changes(self, landmarks: np.ndarray, params: Dict[str, float]):
        """
        应用前额变化
        """
        # 前额皱纹（通过眉毛上方的调整表现）
        forehead_wrinkle = params.get('forehead_wrinkle', 0.0)
        if forehead_wrinkle > 0:
            forehead_indices = self.facial_regions['forehead']
            for idx in forehead_indices:
                landmarks[idx, 1] -= forehead_wrinkle * 2
    
    def _apply_asymmetry(self, landmarks: np.ndarray, params: Dict[str, float]):
        """
        应用整体不对称效果
        """
        asymmetry_factor = params.get('asymmetry_factor', 0.0)
        if asymmetry_factor != 0:
            face_center_x = np.mean(landmarks[:, 0])
            
            for i, (x, y) in enumerate(landmarks):
                if x < face_center_x:  # 左侧
                    landmarks[i, 1] += asymmetry_factor * 1
                else:  # 右侧
                    landmarks[i, 1] -= asymmetry_factor * 1
    
    def interpolate_micro_expressions(self, 
                                    emotion_sequence: List[EmotionType],
                                    intensities: List[float],
                                    durations: List[float],
                                    target_fps: int = 25) -> List[Dict[str, float]]:
        """
        为情感序列生成平滑的微表情参数序列
        
        Args:
            emotion_sequence: 情感序列
            intensities: 情感强度序列
            durations: 持续时间序列
            target_fps: 目标帧率
            
        Returns:
            每帧的微表情参数列表
        """
        if len(emotion_sequence) != len(intensities) or len(emotion_sequence) != len(durations):
            raise ValueError("情感序列、强度和持续时间长度不匹配")
        
        # 计算关键帧时间点
        key_times = [0]
        for duration in durations:
            key_times.append(key_times[-1] + duration)
        
        # 获取关键帧微表情参数
        key_params = []
        for emotion, intensity in zip(emotion_sequence, intensities):
            key_params.append(self.get_emotion_micro_params(emotion, intensity))
        
        # 添加结束帧
        key_params.append(key_params[-1])
        
        # 生成目标时间序列
        total_duration = key_times[-1]
        frame_times = np.arange(0, total_duration, 1.0 / target_fps)
        
        # 插值生成每帧参数
        interpolated_params = []
        param_names = list(self.micro_expression_params.keys())
        
        for frame_time in frame_times:
            frame_params = {}
            
            for param_name in param_names:
                param_values = [params.get(param_name, 0.0) for params in key_params]
                
                if len(set(param_values)) == 1:
                    frame_params[param_name] = param_values[0]
                else:
                    interp_func = interp1d(key_times, param_values, 
                                         kind='cubic', bounds_error=False, 
                                         fill_value='extrapolate')
                    frame_params[param_name] = float(interp_func(frame_time))
            
            interpolated_params.append(frame_params)
        
        return interpolated_params
    
    def process_emotion_sequence(self, 
                               emotion_sequence: List[EmotionType],
                               intensities: List[float],
                               durations: List[float],
                               target_fps: int = 25,
                               add_natural_variations: bool = True) -> List[Dict[str, float]]:
        """
        处理完整的情感序列，生成微表情参数序列
        
        Args:
            emotion_sequence: 情感序列
            intensities: 强度序列
            durations: 持续时间序列
            target_fps: 目标帧率
            add_natural_variations: 是否添加自然变化
            
        Returns:
            完整的微表情参数序列
        """
        # 生成插值参数序列
        params_sequence = self.interpolate_micro_expressions(
            emotion_sequence, intensities, durations, target_fps
        )
        
        # 添加自然变化
        if add_natural_variations:
            for i, params in enumerate(params_sequence):
                params_sequence[i] = self.add_natural_variations(params, i)
        
        # 记录历史
        self.expression_history.extend(params_sequence)
        
        return params_sequence
    
    def get_current_expression_state(self) -> Dict[str, float]:
        """
        获取当前表情状态
        
        Returns:
            当前微表情参数
        """
        if self.expression_history:
            return self.expression_history[-1]
        else:
            return self.micro_expression_params.copy()
    
    def reset_expression_state(self):
        """
        重置表情状态
        """
        self.expression_history = []
        self.last_blink_frame = 0


class FacialMuscleSimulator:
    """
    面部肌肉模拟器
    模拟面部肌肉的细微活动
    """
    
    def __init__(self):
        # 面部肌肉群定义
        self.muscle_groups = {
            'frontalis': {  # 额肌
                'landmarks': [17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
                'activation_pattern': 'vertical_lift',
                'max_displacement': 3.0
            },
            'corrugator': {  # 皱眉肌
                'landmarks': [19, 20, 21, 22, 23, 24],
                'activation_pattern': 'inward_pull',
                'max_displacement': 2.0
            },
            'orbicularis_oculi': {  # 眼轮匝肌
                'landmarks': [36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47],
                'activation_pattern': 'circular_squeeze',
                'max_displacement': 1.5
            },
            'levator_labii': {  # 提上唇肌
                'landmarks': [48, 49, 50, 51, 52, 53, 54],
                'activation_pattern': 'upward_lift',
                'max_displacement': 4.0
            },
            'depressor_labii': {  # 降下唇肌
                'landmarks': [54, 55, 56, 57, 58, 59, 48],
                'activation_pattern': 'downward_pull',
                'max_displacement': 4.0
            },
            'zygomaticus': {  # 颧肌
                'landmarks': [48, 54, 12, 4],
                'activation_pattern': 'diagonal_lift',
                'max_displacement': 3.0
            },
            'masseter': {  # 咬肌
                'landmarks': [5, 6, 7, 8, 9, 10, 11],
                'activation_pattern': 'jaw_clench',
                'max_displacement': 2.0
            }
        }
        
        # 肌肉激活历史
        self.muscle_activation_history = {}
        
    def simulate_muscle_activation(self, 
                                 landmarks: np.ndarray,
                                 micro_params: Dict[str, float]) -> np.ndarray:
        """
        模拟肌肉激活对关键点的影响
        
        Args:
            landmarks: 人脸关键点
            micro_params: 微表情参数
            
        Returns:
            肌肉激活后的关键点
        """
        modified_landmarks = landmarks.copy()
        
        # 根据微表情参数计算肌肉激活强度
        muscle_activations = self._calculate_muscle_activations(micro_params)
        
        # 应用肌肉激活效果
        for muscle_name, activation in muscle_activations.items():
            if muscle_name in self.muscle_groups and activation > 0.01:
                muscle_info = self.muscle_groups[muscle_name]
                self._apply_muscle_effect(modified_landmarks, muscle_info, activation)
        
        return modified_landmarks
    
    def _calculate_muscle_activations(self, micro_params: Dict[str, float]) -> Dict[str, float]:
        """
        根据微表情参数计算肌肉激活强度
        """
        activations = {}
        
        # 额肌激活（前额皱纹、眉毛抬起）
        activations['frontalis'] = max(
            micro_params.get('forehead_wrinkle', 0.0),
            micro_params.get('left_eyebrow_raise', 0.0),
            micro_params.get('right_eyebrow_raise', 0.0)
        )
        
        # 皱眉肌激活
        activations['corrugator'] = max(
            micro_params.get('left_eyebrow_furrow', 0.0),
            micro_params.get('right_eyebrow_furrow', 0.0)
        )
        
        # 眼轮匝肌激活
        activations['orbicularis_oculi'] = max(
            micro_params.get('left_eye_squint', 0.0),
            micro_params.get('right_eye_squint', 0.0)
        )
        
        # 提上唇肌激活
        activations['levator_labii'] = max(
            micro_params.get('left_cheek_raise', 0.0),
            micro_params.get('right_cheek_raise', 0.0)
        )
        
        # 降下唇肌激活
        activations['depressor_labii'] = micro_params.get('facial_tension', 0.0) * 0.5
        
        # 颧肌激活（微笑）
        activations['zygomaticus'] = max(
            micro_params.get('left_cheek_raise', 0.0),
            micro_params.get('right_cheek_raise', 0.0)
        ) * 0.8
        
        # 咬肌激活
        activations['masseter'] = micro_params.get('jaw_clench', 0.0)
        
        return activations
    
    def _apply_muscle_effect(self, 
                           landmarks: np.ndarray, 
                           muscle_info: Dict, 
                           activation: float):
        """
        应用特定肌肉的激活效果
        """
        muscle_landmarks = muscle_info['landmarks']
        pattern = muscle_info['activation_pattern']
        max_displacement = muscle_info['max_displacement']
        
        displacement = activation * max_displacement
        
        if pattern == 'vertical_lift':
            for idx in muscle_landmarks:
                landmarks[idx, 1] -= displacement
        
        elif pattern == 'inward_pull':
            center_x = np.mean(landmarks[muscle_landmarks, 0])
            for idx in muscle_landmarks:
                if landmarks[idx, 0] < center_x:
                    landmarks[idx, 0] += displacement
                else:
                    landmarks[idx, 0] -= displacement
        
        elif pattern == 'circular_squeeze':
            center = np.mean(landmarks[muscle_landmarks], axis=0)
            for idx in muscle_landmarks:
                direction = landmarks[idx] - center
                landmarks[idx] -= direction * displacement * 0.1
        
        elif pattern == 'upward_lift':
            for idx in muscle_landmarks:
                landmarks[idx, 1] -= displacement
        
        elif pattern == 'downward_pull':
            for idx in muscle_landmarks:
                landmarks[idx, 1] += displacement
        
        elif pattern == 'diagonal_lift':
            for idx in muscle_landmarks:
                landmarks[idx, 1] -= displacement * 0.7
                if landmarks[idx, 0] < np.mean(landmarks[:, 0]):
                    landmarks[idx, 0] -= displacement * 0.3
                else:
                    landmarks[idx, 0] += displacement * 0.3
        
        elif pattern == 'jaw_clench':
            for idx in muscle_landmarks:
                landmarks[idx, 1] -= displacement * 0.5


# 使用示例
if __name__ == "__main__":
    # 创建微表情系统
    micro_system = MicroExpressionSystem()
    
    # 测试情感序列
    emotions = [EmotionType.NEUTRAL, EmotionType.HAPPY, EmotionType.SURPRISED]
    intensities = [1.0, 0.8, 0.9]
    durations = [1.0, 2.0, 1.5]
    
    # 生成微表情序列
    micro_sequence = micro_system.process_emotion_sequence(
        emotions, intensities, durations, target_fps=25, add_natural_variations=True
    )
    
    print(f"生成了 {len(micro_sequence)} 帧的微表情参数")
    print(f"第一帧参数示例: {list(micro_sequence[0].keys())[:5]}")
    
    # 创建肌肉模拟器
    muscle_sim = FacialMuscleSimulator()
    print(f"肌肉模拟器已初始化，包含 {len(muscle_sim.muscle_groups)} 个肌肉群")