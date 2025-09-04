# -*- coding: utf-8 -*-
"""
增强数字人说话系统
集成精细嘴部动画控制和微表情系统
实现更自然、更细腻的数字人表情和说话动画
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import cv2
from scipy.interpolate import interp1d
import librosa
import sys
import os

# 添加train_audio路径以导入melspectrogram函数
current_dir = os.path.dirname(os.path.abspath(__file__))
train_audio_path = os.path.join(current_dir, '..', '..', 'train_audio')
sys.path.append(train_audio_path)
from audio import melspectrogram

# 导入自定义模块
from .detailed_mouth_animation import DetailedMouthAnimationController, TeethAndTongueRenderer
from .micro_expression_system import MicroExpressionSystem, FacialMuscleSimulator, EmotionType
from .advanced_micro_expression_system import AdvancedMicroExpressionSystem, MicroExpressionConfig
from .lip_sync_optimizer import LipSyncOptimizer, RealTimeLipSyncOptimizer
from .phoneme_mouth_mapping import PhonemeToMouthMapper
from .audio2bs_lstm import Audio2Feature

class EnhancedTalkingFaceSystem:
    """
    增强数字人说话系统
    整合音频处理、音素识别、嘴部动画、微表情等功能
    """
    
    def __init__(self, 
                 model_path: str = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # 初始化各个子系统
        self.mouth_controller = DetailedMouthAnimationController()
        
        # 创建高级微表情系统配置
        micro_config = MicroExpressionConfig(
            breathing_amplitude=0.012,
            breathing_frequency=0.28,
            blink_frequency=0.35,
            blink_duration=0.12,
            micro_movement_amplitude=0.006,
            emotion_transition_speed=0.6,
            asymmetry_factor=0.03,
            natural_variation_strength=0.025
        )
        
        # 初始化高级微表情系统
        self.advanced_micro_system = AdvancedMicroExpressionSystem(micro_config)
        self.micro_expression_system = MicroExpressionSystem()  # 保持兼容性
        self.muscle_simulator = FacialMuscleSimulator()
        self.phoneme_mapper = PhonemeToMouthMapper()
        self.teeth_tongue_renderer = TeethAndTongueRenderer()
        
        # 初始化唇形同步优化器
        self.lip_sync_optimizer = LipSyncOptimizer()
        self.realtime_lip_sync = RealTimeLipSyncOptimizer(self.lip_sync_optimizer)
        
        # 加载音频到特征模型
        if model_path:
            self.audio_model = self._load_audio_model(model_path)
        else:
            self.audio_model = None
        
        # 系统配置参数
        self.config = {
            'sample_rate': 16000,
            'hop_length': 160,
            'frame_rate': 25,
            'audio_buffer_size': 1024,
            'lip_sync_lookahead': 3,  # 帧数
            'emotion_smoothing': 0.8,
            'micro_expression_enabled': True,
            'advanced_breathing': True,
            'natural_asymmetry': True,
            'win_length': 400,
            'n_mels': 80,
            'target_fps': 25,
            'mouth_region_size': 128,
            'enable_micro_expressions': True,
            'enable_muscle_simulation': True,
            'enable_teeth_tongue': True,
            'emotion_intensity_factor': 0.7,
            'natural_variation_factor': 1.0
        }
        
        # 状态管理
        self.current_emotion = EmotionType.NEUTRAL
        self.emotion_intensity = 0.5
        self.frame_index = 0
        
        # 缓存管理
        self.audio_features_cache = None
        self.phoneme_sequence_cache = None
        self.animation_cache = None
        
    def _load_audio_model(self, model_path: str) -> nn.Module:
        """
        加载音频到特征转换模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的模型
        """
        try:
            model = Audio2Feature(ndim=80, output_size=6)
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self.device)
            model.eval()
            return model
        except Exception as e:
            print(f"警告：无法加载音频模型 {model_path}: {e}")
            return None
    
    def extract_audio_features(self, audio_path: str) -> np.ndarray:
        """
        从音频文件提取特征
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音频特征数组 [T, 80]
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
            
            # 提取梅尔频谱特征
            mel_spec = melspectrogram(audio)
            
            # 转置以匹配模型输入格式 [T, 80]
            features = mel_spec.T
            
            # 缓存特征
            self.audio_features_cache = features
            
            return features
            
        except Exception as e:
            print(f"音频特征提取失败: {e}")
            return np.zeros((100, 80))  # 返回默认特征
    
    def extract_phonemes_from_audio(self, audio_path: str) -> List[Tuple[str, float, float]]:
        """
        从音频中提取音素序列（简化实现）
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            音素序列 [(phoneme, start_time, duration), ...]
        """
        try:
            # 这里应该使用专业的语音识别和音素对齐工具
            # 为了演示，我们使用简化的实现
            audio, sr = librosa.load(audio_path, sr=self.config['sample_rate'])
            duration = len(audio) / sr
            
            # 简化的音素序列生成（实际应用中需要使用ASR+强制对齐）
            # 这里假设是中文"你好"的发音
            phoneme_sequence = [
                ('n', 0.0, 0.2),
                ('i', 0.2, 0.3),
                ('h', 0.5, 0.2),
                ('ao', 0.7, 0.4)
            ]
            
            # 调整时长以匹配实际音频
            total_phoneme_duration = sum([dur for _, _, dur in phoneme_sequence])
            scale_factor = duration / total_phoneme_duration
            
            scaled_sequence = []
            current_time = 0.0
            for phoneme, _, dur in phoneme_sequence:
                scaled_duration = dur * scale_factor
                scaled_sequence.append((phoneme, current_time, scaled_duration))
                current_time += scaled_duration
            
            self.phoneme_sequence_cache = scaled_sequence
            return scaled_sequence
            
        except Exception as e:
            print(f"音素提取失败: {e}")
            return [('sil', 0.0, 1.0)]  # 返回静音
    
    def generate_mouth_animation(self, 
                               phoneme_sequence: List[Tuple[str, float, float]],
                               emotion: EmotionType = EmotionType.NEUTRAL,
                               emotion_intensity: float = 0.5) -> List[Dict[str, float]]:
        """
        生成嘴部动画参数序列
        
        Args:
            phoneme_sequence: 音素序列
            emotion: 情感类型
            emotion_intensity: 情感强度
            
        Returns:
            嘴部动画参数序列
        """
        # 提取音素和持续时间
        phonemes = [p[0] for p in phoneme_sequence]
        durations = [p[2] for p in phoneme_sequence]
        
        # 生成基础嘴部动画
        mouth_animation = self.mouth_controller.process_phoneme_sequence(
            phonemes, durations, 
            target_fps=self.config['target_fps'],
            add_breathing=True
        )
        
        # 根据情感调整动画参数
        if emotion != EmotionType.NEUTRAL:
            mouth_animation = self._apply_emotion_to_mouth_animation(
                mouth_animation, emotion, emotion_intensity
            )
        
        return mouth_animation
    
    def generate_micro_expressions(self, 
                                 phoneme_sequence: List[Tuple[str, float, float]],
                                 emotion: EmotionType = EmotionType.NEUTRAL,
                                 emotion_intensity: float = 0.5) -> List[Dict[str, float]]:
        """
        生成微表情参数序列
        
        Args:
            phoneme_sequence: 音素序列
            emotion: 情感类型
            emotion_intensity: 情感强度
            
        Returns:
            微表情参数序列
        """
        # 计算总时长
        total_duration = sum([p[2] for p in phoneme_sequence])
        
        # 为整个序列应用相同的情感
        emotions = [emotion]
        intensities = [emotion_intensity * self.config['emotion_intensity_factor']]
        durations = [total_duration]
        
        # 生成微表情序列
        micro_expressions = self.micro_expression_system.process_emotion_sequence(
            emotions, intensities, durations,
            target_fps=self.config['target_fps'],
            add_natural_variations=True
        )
        
        return micro_expressions
    
    def _apply_emotion_to_mouth_animation(self, 
                                        mouth_animation: List[Dict[str, float]],
                                        emotion: EmotionType,
                                        intensity: float) -> List[Dict[str, float]]:
        """
        将情感影响应用到嘴部动画
        
        Args:
            mouth_animation: 原始嘴部动画
            emotion: 情感类型
            intensity: 情感强度
            
        Returns:
            调整后的嘴部动画
        """
        emotion_adjustments = {
            EmotionType.HAPPY: {
                'mouth_corner_lift': 0.15,  # 降低幅度，更自然
                'lip_width_increase': 0.08,
                'teeth_visibility_boost': 0.12
            },
            EmotionType.SAD: {
                'mouth_corner_drop': 0.08,  # 更细微的变化
                'lip_opening_reduce': 0.05,
                'overall_tension_reduce': 0.08
            },
            EmotionType.ANGRY: {
                'lip_tension_increase': 0.18,  # 减少过度紧张
                'jaw_clench_boost': 0.12,
                'mouth_asymmetry': 0.04
            },
            EmotionType.SURPRISED: {
                'lip_opening_boost': 0.12,  # 更自然的惊讶表情
                'jaw_opening_increase': 0.08
            },
            EmotionType.NEUTRAL: {  # 添加中性情感的微调
                'natural_variation': 0.02  # 轻微的自然变化
            }
        }
        
        if emotion not in emotion_adjustments:
            return mouth_animation
        
        adjustments = emotion_adjustments[emotion]
        modified_animation = []
        
        for frame_params in mouth_animation:
            modified_params = frame_params.copy()
            
            # 应用情感调整
            for adjustment, value in adjustments.items():
                if 'corner_lift' in adjustment:
                    # 嘴角上扬
                    modified_params['mouth_corner_stretch'] = modified_params.get('mouth_corner_stretch', 0.0) + value * intensity
                elif 'corner_drop' in adjustment:
                    # 嘴角下垂
                    modified_params['mouth_corner_stretch'] = modified_params.get('mouth_corner_stretch', 0.0) - value * intensity
                elif 'width_increase' in adjustment:
                    # 嘴巴变宽
                    modified_params['lip_width'] = modified_params.get('lip_width', 0.0) + value * intensity
                elif 'opening_boost' in adjustment:
                    # 增加开口度
                    modified_params['lip_opening'] = min(1.0, modified_params.get('lip_opening', 0.0) + value * intensity)
                elif 'opening_reduce' in adjustment:
                    # 减少开口度
                    modified_params['lip_opening'] = max(0.0, modified_params.get('lip_opening', 0.0) - value * intensity)
                elif 'tension_increase' in adjustment:
                    # 增加紧张度
                    for param in ['lip_protrusion', 'jaw_opening']:
                        if param in modified_params:
                            modified_params[param] = min(1.0, modified_params[param] + value * intensity * 0.5)
                elif 'visibility_boost' in adjustment:
                    # 增加牙齿可见度
                    modified_params['teeth_visibility'] = min(1.0, modified_params.get('teeth_visibility', 0.0) + value * intensity)
            
            # 确保参数在有效范围内
            for key, val in modified_params.items():
                if 'opening' in key or 'visibility' in key or 'raise' in key or 'drop' in key:
                    modified_params[key] = np.clip(val, 0.0, 1.0)
                else:
                    modified_params[key] = np.clip(val, -1.0, 1.0)
            
            modified_animation.append(modified_params)
        
        return modified_animation
    
    def process_audio_to_animation(self, 
                                 audio_path: str,
                                 emotion: EmotionType = EmotionType.NEUTRAL,
                                 emotion_intensity: float = 0.5) -> Dict[str, List[Dict[str, float]]]:
        """
        处理音频文件，生成完整的动画数据
        使用高级微表情系统和唇形同步优化
        
        Args:
            audio_path: 音频文件路径
            emotion: 情感类型
            emotion_intensity: 情感强度
            
        Returns:
            包含嘴部动画和微表情的完整动画数据
        """
        try:
            # 设置高级微表情系统的目标情感
            self.advanced_micro_system.set_emotion(emotion, duration=2.0)
            
            # 提取高精度音频特征
            audio_features = self.lip_sync_optimizer.extract_audio_features(audio_path)
            
            # 提取优化的音素序列
            phoneme_sequence = self.lip_sync_optimizer.extract_phonemes(audio_path)
            
            # 进行时序对齐优化
            aligned_phonemes = self.lip_sync_optimizer.align_phonemes_with_audio(
                phoneme_sequence, audio_features
            )
            
            # 生成基础嘴部动画
            mouth_animation = self.generate_mouth_animation(
                aligned_phonemes, emotion, emotion_intensity
            )
            
            # 应用唇形同步优化
            optimized_mouth_animation = self.lip_sync_optimizer.optimize_lip_sync(
                mouth_animation, audio_features, aligned_phonemes
            )
            
            # 生成高级微表情序列
            advanced_micro_expressions = []
            frame_count = len(optimized_mouth_animation)
            
            for frame_idx in range(frame_count):
                # 更新高级微表情系统
                micro_params = self.advanced_micro_system.update_frame(1/self.config['frame_rate'])
                
                # 添加音频驱动的微表情调整
                if frame_idx < len(audio_features):
                    audio_intensity = np.mean(np.abs(audio_features[frame_idx]))
                    micro_params['audio_driven_intensity'] = audio_intensity
                    
                    # 根据音频强度调整微表情
                    micro_params['eyebrow_raise'] += audio_intensity * 0.1
                    micro_params['eye_squint'] += audio_intensity * 0.05
                
                advanced_micro_expressions.append(micro_params)
            
            # 生成传统微表情（保持兼容性）
            traditional_micro_expressions = None
            if self.config['enable_micro_expressions']:
                traditional_micro_expressions = self.generate_micro_expressions(
                    aligned_phonemes, emotion, emotion_intensity
                )
            
            # 应用情感调整到嘴部动画
            final_mouth_animation = self._apply_emotion_to_mouth_animation(
                optimized_mouth_animation, emotion, emotion_intensity
            )
            
            # 计算同步质量指标
            sync_quality = self.lip_sync_optimizer.calculate_sync_quality(
                final_mouth_animation, audio_features, aligned_phonemes
            )
            
            # 缓存结果
            self.animation_cache = {
                'mouth_animation': final_mouth_animation,
                'micro_expressions': traditional_micro_expressions,
                'advanced_micro_expressions': advanced_micro_expressions,
                'audio_features': audio_features.tolist() if isinstance(audio_features, np.ndarray) else audio_features,
                'phoneme_sequence': aligned_phonemes,
                'original_phonemes': phoneme_sequence,
                'emotion': emotion.value,
                'emotion_intensity': emotion_intensity,
                'sync_quality': sync_quality,
                'optimization_applied': True
            }
            
            return self.animation_cache
            
        except Exception as e:
            print(f"处理音频时出错: {e}")
            # 降级到传统处理方式
            try:
                audio_features = self.extract_audio_features(audio_path)
                phoneme_sequence = self.extract_phonemes_from_audio(audio_path)
                mouth_animation = self.generate_mouth_animation(
                    phoneme_sequence, emotion, emotion_intensity
                )
                micro_expressions = None
                if self.config['enable_micro_expressions']:
                    micro_expressions = self.generate_micro_expressions(
                        phoneme_sequence, emotion, emotion_intensity
                    )
                
                self.animation_cache = {
                    'mouth_animation': mouth_animation,
                    'micro_expressions': micro_expressions,
                    'advanced_micro_expressions': [],
                    'audio_features': audio_features.tolist() if isinstance(audio_features, np.ndarray) else audio_features,
                    'phoneme_sequence': phoneme_sequence,
                    'emotion': emotion.value,
                    'emotion_intensity': emotion_intensity,
                    'sync_quality': {'error': 'optimization_failed'},
                    'optimization_applied': False,
                    'fallback_used': True
                }
                
                return self.animation_cache
                
            except Exception as fallback_error:
                self.animation_cache = {
                    'mouth_animation': [],
                    'micro_expressions': [],
                    'advanced_micro_expressions': [],
                    'audio_features': [],
                    'phoneme_sequence': [],
                    'error': str(e),
                    'fallback_error': str(fallback_error)
                }
                
                return self.animation_cache
    
    def apply_animation_to_landmarks(self, 
                                   landmarks: np.ndarray,
                                   frame_index: int,
                                   animation_data: Dict = None) -> np.ndarray:
        """
        将动画参数应用到人脸关键点
        集成高级微表情系统和优化的唇形同步
        
        Args:
            landmarks: 原始68点人脸关键点 [68, 2]
            frame_index: 当前帧索引
            animation_data: 动画数据（如果为None则使用缓存）
            
        Returns:
            变换后的关键点
        """
        if animation_data is None:
            animation_data = self.animation_cache
        
        if animation_data is None:
            return landmarks
        
        modified_landmarks = landmarks.copy()
        
        # 应用优化的嘴部动画
        mouth_animation = animation_data.get('mouth_animation', [])
        if frame_index < len(mouth_animation):
            mouth_params = mouth_animation[frame_index]
            modified_landmarks = self.mouth_controller.apply_animation_to_landmarks(
                modified_landmarks, mouth_params
            )
        
        # 应用高级微表情系统（优先）
        advanced_micro_expressions = animation_data.get('advanced_micro_expressions', [])
        if (self.config.get('micro_expression_enabled', True) and 
            advanced_micro_expressions and 
            frame_index < len(advanced_micro_expressions)):
            
            micro_params = advanced_micro_expressions[frame_index]
            modified_landmarks = self.advanced_micro_system.apply_to_landmarks(
                modified_landmarks, micro_params
            )
        
        # 应用传统微表情（作为补充或后备）
        elif self.config['enable_micro_expressions']:
            micro_expressions = animation_data.get('micro_expressions', [])
            if frame_index < len(micro_expressions):
                micro_params = micro_expressions[frame_index]
                modified_landmarks = self.micro_expression_system.apply_micro_expressions_to_landmarks(
                    modified_landmarks, micro_params
                )
        
        # 应用肌肉模拟（如果启用）
        if (self.config['enable_muscle_simulation'] and 
            hasattr(self, 'muscle_simulator')):
            micro_expressions = animation_data.get('micro_expressions', [])
            if frame_index < len(micro_expressions):
                micro_params = micro_expressions[frame_index]
                modified_landmarks = self.muscle_simulator.simulate_muscle_activation(
                    modified_landmarks, micro_params
                )
        
        # 应用自然不对称效果
        if self.config.get('natural_asymmetry', True):
            modified_landmarks = self._apply_natural_asymmetry(
                modified_landmarks, frame_index
            )
        
        # 应用时序平滑
        if hasattr(self, '_previous_landmarks') and self.config.get('temporal_smoothing', True):
            smoothing_factor = self.config.get('smoothing_factor', 0.3)
            modified_landmarks = (
                modified_landmarks * (1 - smoothing_factor) + 
                self._previous_landmarks * smoothing_factor
            )
        
        # 缓存当前关键点用于下一帧平滑
        self._previous_landmarks = modified_landmarks.copy()
        
        return modified_landmarks
    
    def _apply_natural_asymmetry(self, landmarks: np.ndarray, frame_index: int) -> np.ndarray:
        """
        应用自然的面部不对称效果
        
        Args:
            landmarks: 关键点数组
            frame_index: 帧索引
            
        Returns:
            调整后的关键点
        """
        adjusted_landmarks = landmarks.copy()
        
        # 计算面部中心线
        center_x = np.mean(landmarks[:, 0])
        
        # 生成基于时间的微小不对称变化
        asymmetry_strength = self.config.get('asymmetry_factor', 0.03)
        time_factor = np.sin(frame_index * 0.1) * asymmetry_strength
        
        # 应用到左右两侧
        for i, landmark in enumerate(landmarks):
            if landmark[0] < center_x:  # 左侧
                adjusted_landmarks[i, 0] += time_factor * 0.5
                adjusted_landmarks[i, 1] += time_factor * 0.3
            else:  # 右侧
                adjusted_landmarks[i, 0] -= time_factor * 0.5
                adjusted_landmarks[i, 1] -= time_factor * 0.3
        
        return adjusted_landmarks
    
    def render_enhanced_frame(self, 
                            image: np.ndarray,
                            landmarks: np.ndarray,
                            frame_index: int,
                            animation_data: Dict = None) -> np.ndarray:
        """
        渲染增强的帧图像
        
        Args:
            image: 输入图像
            landmarks: 人脸关键点
            frame_index: 当前帧索引
            animation_data: 动画数据
            
        Returns:
            渲染后的图像
        """
        if animation_data is None:
            animation_data = self.animation_cache
        
        if animation_data is None:
            return image
        
        result_image = image.copy()
        
        # 渲染牙齿和舌头
        if self.config['enable_teeth_tongue']:
            mouth_animation = animation_data.get('mouth_animation', [])
            if frame_index < len(mouth_animation):
                mouth_params = mouth_animation[frame_index]
                
                # 渲染牙齿
                teeth_visibility = mouth_params.get('teeth_visibility', 0.0)
                if teeth_visibility > 0.1:
                    mouth_landmarks = landmarks[48:68]  # 嘴部关键点
                    result_image = self.teeth_tongue_renderer.render_teeth(
                        result_image, mouth_landmarks, teeth_visibility
                    )
                
                # 渲染舌头
                tongue_position = mouth_params.get('tongue_position', 0.0)
                tongue_height = mouth_params.get('tongue_height', 0.0)
                if tongue_height > 0.1:
                    mouth_landmarks = landmarks[48:68]
                    result_image = self.teeth_tongue_renderer.render_tongue(
                        result_image, mouth_landmarks, tongue_position, tongue_height
                    )
        
        return result_image
    
    def set_emotion(self, emotion: EmotionType, intensity: float = 0.5):
        """
        设置当前情感状态
        
        Args:
            emotion: 情感类型
            intensity: 情感强度
        """
        self.current_emotion = emotion
        self.emotion_intensity = intensity
    
    def get_animation_info(self) -> Dict:
        """
        获取当前动画信息
        
        Returns:
            动画信息字典
        """
        if self.animation_cache is None:
            return {}
        
        mouth_frames = len(self.animation_cache.get('mouth_animation', []))
        micro_frames = len(self.animation_cache.get('micro_expressions', []))
        phonemes = len(self.animation_cache.get('phoneme_sequence', []))
        
        return {
            'mouth_animation_frames': mouth_frames,
            'micro_expression_frames': micro_frames,
            'phoneme_count': phonemes,
            'total_duration': mouth_frames / self.config['target_fps'],
            'current_emotion': self.current_emotion.value,
            'emotion_intensity': self.emotion_intensity
        }
    
    def reset_system(self):
        """
        重置系统状态
        """
        self.frame_index = 0
        self.current_emotion = EmotionType.NEUTRAL
        self.emotion_intensity = 0.5
        self.audio_features_cache = None
        self.phoneme_sequence_cache = None
        self.animation_cache = None
        self.micro_expression_system.reset_expression_state()
    
    def update_config(self, **kwargs):
        """
        更新系统配置
        
        Args:
            **kwargs: 配置参数
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                print(f"配置已更新: {key} = {value}")
            else:
                print(f"警告: 未知配置参数 {key}")


class RealTimeEnhancedTalkingFace:
    """
    实时增强数字人说话系统
    集成实时唇形同步优化和高级微表情
    """
    
    def __init__(self, enhanced_system: EnhancedTalkingFaceSystem):
        self.enhanced_system = enhanced_system
        self.audio_buffer = []
        self.animation_buffer = []
        self.buffer_size = enhanced_system.config.get('audio_buffer_size', 1024)
        
        # 初始化实时处理器
        self.realtime_lip_sync = enhanced_system.realtime_lip_sync
        self.frame_counter = 0
        self.last_emotion = EmotionType.NEUTRAL
        
        # 实时状态缓存
        self.realtime_state = {
            'current_phoneme': None,
            'phoneme_progress': 0.0,
            'audio_intensity': 0.0,
            'micro_expression_state': {},
            'breathing_phase': 0.0
        }
        
    def process_audio_chunk(self, audio_chunk: np.ndarray, 
                          emotion: EmotionType = None) -> Optional[Dict[str, float]]:
        """
        处理实时音频块
        使用高级唇形同步和微表情系统
        
        Args:
            audio_chunk: 音频数据块
            emotion: 当前情感状态
            
        Returns:
            当前帧的完整动画参数
        """
        # 更新情感状态
        if emotion and emotion != self.last_emotion:
            self.enhanced_system.advanced_micro_system.set_emotion(emotion, duration=1.0)
            self.last_emotion = emotion
        
        # 添加到缓冲区
        self.audio_buffer.extend(audio_chunk)
        
        # 当缓冲区足够大时处理
        if len(self.audio_buffer) >= self.buffer_size:
            try:
                # 使用实时唇形同步优化器处理
                chunk_array = np.array(self.audio_buffer[:self.buffer_size])
                
                # 实时特征提取
                realtime_features = self.realtime_lip_sync.process_audio_chunk(chunk_array)
                
                # 更新高级微表情系统
                micro_params = self.enhanced_system.advanced_micro_system.update_frame(
                    1/self.enhanced_system.config['frame_rate']
                )
                
                # 计算音频强度
                audio_intensity = np.mean(np.abs(chunk_array))
                self.realtime_state['audio_intensity'] = audio_intensity
                
                # 生成优化的嘴部动画参数
                mouth_params = self._generate_realtime_mouth_params(
                    realtime_features, audio_intensity
                )
                
                # 合并所有动画参数
                animation_params = {
                    **mouth_params,
                    **micro_params,
                    'frame_index': self.frame_counter,
                    'audio_intensity': audio_intensity,
                    'realtime_processing': True
                }
                
                # 应用实时平滑
                animation_params = self._apply_realtime_smoothing(animation_params)
                
                # 移除已处理的音频（保留重叠部分）
                overlap_size = self.buffer_size // 4
                self.audio_buffer = self.audio_buffer[self.buffer_size - overlap_size:]
                
                self.frame_counter += 1
                return animation_params
                
            except Exception as e:
                print(f"实时处理出错: {e}")
                # 降级到简化处理
                return self._fallback_processing(chunk_array)
        
        return None
    
    def _generate_realtime_mouth_params(self, features: Dict, audio_intensity: float) -> Dict[str, float]:
        """
        生成实时嘴部动画参数
        
        Args:
            features: 音频特征
            audio_intensity: 音频强度
            
        Returns:
            嘴部动画参数
        """
        # 基于音频特征生成嘴部参数
        mouth_open = min(1.0, audio_intensity * 3.0)
        
        # 根据频谱特征调整嘴型
        if 'spectral_centroid' in features:
            spectral_factor = features['spectral_centroid'] / 4000.0  # 归一化
            mouth_width = 0.3 + spectral_factor * 0.4
        else:
            mouth_width = 0.5
        
        # 牙齿和舌头可见度
        teeth_visible = min(0.8, mouth_open * 1.2)
        tongue_position = np.sin(self.frame_counter * 0.3) * 0.1  # 自然摆动
        
        # 嘴角调整
        corner_pull = audio_intensity * 0.2
        if self.last_emotion == EmotionType.HAPPY:
            corner_pull += 0.3
        elif self.last_emotion == EmotionType.SAD:
            corner_pull -= 0.2
        
        return {
            'mouth_open': mouth_open,
            'mouth_width': mouth_width,
            'teeth_visible': teeth_visible,
            'tongue_position': tongue_position,
            'corner_pull': corner_pull,
            'lip_tension': audio_intensity * 0.5
        }
    
    def _apply_realtime_smoothing(self, current_params: Dict[str, float]) -> Dict[str, float]:
        """
        应用实时平滑处理
        
        Args:
            current_params: 当前参数
            
        Returns:
            平滑后的参数
        """
        if not hasattr(self, '_previous_params'):
            self._previous_params = current_params.copy()
            return current_params
        
        smoothing_factor = 0.4  # 实时处理使用较强的平滑
        smoothed_params = {}
        
        for key, value in current_params.items():
            if key in self._previous_params and isinstance(value, (int, float)):
                smoothed_params[key] = (
                    value * (1 - smoothing_factor) + 
                    self._previous_params[key] * smoothing_factor
                )
            else:
                smoothed_params[key] = value
        
        self._previous_params = smoothed_params.copy()
        return smoothed_params
    
    def _fallback_processing(self, audio_chunk: np.ndarray) -> Dict[str, float]:
        """
        降级处理方式
        
        Args:
            audio_chunk: 音频块
            
        Returns:
            基础动画参数
        """
        audio_intensity = np.mean(np.abs(audio_chunk))
        
        return {
            'mouth_open': min(1.0, audio_intensity * 2.0),
            'teeth_visible': min(0.8, audio_intensity * 1.5),
            'tongue_position': np.random.uniform(-0.1, 0.1),
            'corner_pull': 0.0,
            'mouth_width': 0.5,
            'lip_tension': audio_intensity * 0.3,
            'fallback_mode': True
        }
    
    def get_current_animation_params(self) -> Dict[str, float]:
        """
        获取当前帧的动画参数
        
        Returns:
            动画参数字典
        """
        if self.animation_buffer:
            return self.animation_buffer.pop(0)
        else:
            # 返回默认参数
            return {
                'lip_opening': 0.0,
                'teeth_visibility': 0.0,
                'tongue_position': 0.0
            }


# 使用示例
if __name__ == "__main__":
    # 创建增强数字人系统
    enhanced_system = EnhancedTalkingFaceSystem()
    
    # 设置情感状态
    enhanced_system.set_emotion(EmotionType.HAPPY, intensity=0.7)
    
    # 模拟音频处理（实际使用时传入真实音频文件路径）
    print("增强数字人说话系统已初始化")
    print(f"当前配置: {enhanced_system.config}")
    
    # 获取系统信息
    info = enhanced_system.get_animation_info()
    print(f"系统信息: {info}")
    
    # 创建实时系统
    realtime_system = RealTimeEnhancedTalkingFace(enhanced_system)
    print("实时处理系统已就绪")