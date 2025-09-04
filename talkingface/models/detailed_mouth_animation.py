# -*- coding: utf-8 -*-
"""
精细嘴部动画控制系统
实现嘴唇开合度、牙齿显露、舌头位置的动态调整
与音素映射系统协同工作，提供更自然的说话动画
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import cv2
from scipy.interpolate import interp1d

class DetailedMouthAnimationController:
    """
    精细嘴部动画控制器
    负责控制嘴唇、牙齿、舌头的精细动作
    """
    
    def __init__(self):
        # 嘴部关键点索引（基于68点人脸标记）
        self.mouth_landmarks = {
            'outer_lip': list(range(48, 60)),  # 外唇轮廓
            'inner_lip': list(range(60, 68)),  # 内唇轮廓
            'upper_lip': [48, 49, 50, 51, 52, 53, 54],  # 上唇
            'lower_lip': [54, 55, 56, 57, 58, 59, 48],  # 下唇
            'lip_corners': [48, 54],  # 嘴角
            'lip_center': [51, 57]  # 唇中心点
        }
        
        # 嘴部动画参数定义
        self.animation_params = {
            'lip_opening': 0.0,      # 嘴唇开合度 [0-1]
            'lip_width': 0.0,        # 嘴唇宽度 [-1,1]
            'lip_protrusion': 0.0,   # 嘴唇突出度 [-1,1]
            'upper_lip_raise': 0.0,  # 上唇抬起 [0-1]
            'lower_lip_drop': 0.0,   # 下唇下降 [0-1]
            'teeth_visibility': 0.0, # 牙齿显露度 [0-1]
            'tongue_position': 0.0,  # 舌头位置 [-1,1]
            'tongue_height': 0.0,    # 舌头高度 [0-1]
            'jaw_opening': 0.0,      # 下颌开合 [0-1]
            'mouth_asymmetry': 0.0   # 嘴部不对称度 [-1,1]
        }
        
        # 音素到动画参数的映射
        self.phoneme_animation_mapping = self._create_phoneme_animation_mapping()
        
        # 时序平滑参数
        self.smoothing_window = 3  # 减小平滑窗口，保持更多细节
        self.animation_history = []
        
    def _create_phoneme_animation_mapping(self) -> Dict[str, Dict[str, float]]:
        """
        创建音素到动画参数的详细映射
        """
        mapping = {
            # 元音 - 优化为更自然的人类发音参数
            'a': {
                'lip_opening': 0.45, 'lip_width': 0.15, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.18, 'lower_lip_drop': 0.35, 'teeth_visibility': 0.25,
                'tongue_position': 0.0, 'tongue_height': 0.15, 'jaw_opening': 0.4
            },
            'e': {
                'lip_opening': 0.25, 'lip_width': 0.35, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.12, 'lower_lip_drop': 0.18, 'teeth_visibility': 0.4,
                'tongue_position': 0.0, 'tongue_height': 0.45, 'jaw_opening': 0.25
            },
            'i': {
                'lip_opening': 0.15, 'lip_width': 0.45, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.08, 'lower_lip_drop': 0.08, 'teeth_visibility': 0.5,
                'tongue_position': 0.0, 'tongue_height': 0.65, 'jaw_opening': 0.15
            },
            'o': {
                'lip_opening': 0.35, 'lip_width': -0.25, 'lip_protrusion': 0.4,
                'upper_lip_raise': 0.22, 'lower_lip_drop': 0.22, 'teeth_visibility': 0.05,
                'tongue_position': 0.0, 'tongue_height': 0.2, 'jaw_opening': 0.35
            },
            'u': {
                'lip_opening': 0.2, 'lip_width': -0.5, 'lip_protrusion': 0.55,
                'upper_lip_raise': 0.12, 'lower_lip_drop': 0.12, 'teeth_visibility': 0.0,
                'tongue_position': 0.0, 'tongue_height': 0.25, 'jaw_opening': 0.2
            },
            
            # 辅音 - 爆破音 (优化为自然发音参数)
            'p': {
                'lip_opening': 0.02, 'lip_width': 0.0, 'lip_protrusion': 0.05,
                'upper_lip_raise': 0.0, 'lower_lip_drop': 0.0, 'teeth_visibility': 0.0,
                'tongue_position': 0.0, 'tongue_height': 0.0, 'jaw_opening': 0.02
            },
            'b': {
                'lip_opening': 0.02, 'lip_width': 0.0, 'lip_protrusion': 0.05,
                'upper_lip_raise': 0.0, 'lower_lip_drop': 0.0, 'teeth_visibility': 0.0,
                'tongue_position': 0.0, 'tongue_height': 0.0, 'jaw_opening': 0.02
            },
            't': {
                'lip_opening': 0.08, 'lip_width': 0.12, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.06, 'teeth_visibility': 0.35,
                'tongue_position': 0.0, 'tongue_height': 0.5, 'jaw_opening': 0.08
            },
            'd': {
                'lip_opening': 0.08, 'lip_width': 0.12, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.06, 'teeth_visibility': 0.35,
                'tongue_position': 0.0, 'tongue_height': 0.5, 'jaw_opening': 0.08
            },
            'k': {
                'lip_opening': 0.12, 'lip_width': 0.08, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.12, 'teeth_visibility': 0.15,
                'tongue_position': -0.3, 'tongue_height': 0.45, 'jaw_opening': 0.12
            },
            'g': {
                'lip_opening': 0.12, 'lip_width': 0.08, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.12, 'teeth_visibility': 0.15,
                'tongue_position': -0.3, 'tongue_height': 0.45, 'jaw_opening': 0.12
            },
            
            # 摩擦音 (优化为自然发音参数)
            'f': {
                'lip_opening': 0.06, 'lip_width': 0.0, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.12, 'lower_lip_drop': 0.0, 'teeth_visibility': 0.4,
                'tongue_position': 0.0, 'tongue_height': 0.0, 'jaw_opening': 0.06
            },
            'v': {
                'lip_opening': 0.06, 'lip_width': 0.0, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.12, 'lower_lip_drop': 0.0, 'teeth_visibility': 0.4,
                'tongue_position': 0.0, 'tongue_height': 0.0, 'jaw_opening': 0.06
            },
            's': {
                'lip_opening': 0.08, 'lip_width': 0.18, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.06, 'teeth_visibility': 0.45,
                'tongue_position': 0.0, 'tongue_height': 0.42, 'jaw_opening': 0.08
            },
            'z': {
                'lip_opening': 0.08, 'lip_width': 0.18, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.06, 'teeth_visibility': 0.45,
                'tongue_position': 0.0, 'tongue_height': 0.42, 'jaw_opening': 0.08
            },
            'sh': {
                'lip_opening': 0.12, 'lip_width': -0.12, 'lip_protrusion': 0.25,
                'upper_lip_raise': 0.12, 'lower_lip_drop': 0.12, 'teeth_visibility': 0.15,
                'tongue_position': 0.0, 'tongue_height': 0.6, 'jaw_opening': 0.2
            },
            'zh': {
                'lip_opening': 0.12, 'lip_width': -0.12, 'lip_protrusion': 0.25,
                'upper_lip_raise': 0.12, 'lower_lip_drop': 0.12, 'teeth_visibility': 0.15,
                'tongue_position': 0.0, 'tongue_height': 0.38, 'jaw_opening': 0.12
            },
            
            # 鼻音 (优化为自然发音参数)
            'm': {
                'lip_opening': 0.02, 'lip_width': 0.0, 'lip_protrusion': 0.03,
                'upper_lip_raise': 0.0, 'lower_lip_drop': 0.0, 'teeth_visibility': 0.0,
                'tongue_position': 0.0, 'tongue_height': 0.0, 'jaw_opening': 0.02
            },
            'n': {
                'lip_opening': 0.06, 'lip_width': 0.08, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.06, 'teeth_visibility': 0.25,
                'tongue_position': 0.0, 'tongue_height': 0.5, 'jaw_opening': 0.06
            },
            'ng': {
                'lip_opening': 0.1, 'lip_width': 0.0, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.1, 'teeth_visibility': 0.1,
                'tongue_position': -0.4, 'tongue_height': 0.5, 'jaw_opening': 0.1
            },
            
            # 流音 (优化为自然发音参数)
            'l': {
                'lip_opening': 0.12, 'lip_width': 0.12, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.12, 'teeth_visibility': 0.3,
                'tongue_position': 0.0, 'tongue_height': 0.55, 'jaw_opening': 0.12
            },
            'r': {
                'lip_opening': 0.18, 'lip_width': -0.06, 'lip_protrusion': 0.12,
                'upper_lip_raise': 0.12, 'lower_lip_drop': 0.18, 'teeth_visibility': 0.2,
                'tongue_position': 0.0, 'tongue_height': 0.3, 'jaw_opening': 0.18
            },
            
            # 半元音 (优化为自然发音参数)
            'w': {
                'lip_opening': 0.18, 'lip_width': -0.35, 'lip_protrusion': 0.45,
                'upper_lip_raise': 0.12, 'lower_lip_drop': 0.12, 'teeth_visibility': 0.05,
                'tongue_position': 0.0, 'tongue_height': 0.25, 'jaw_opening': 0.18
            },
            'y': {
                'lip_opening': 0.12, 'lip_width': 0.35, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.06, 'lower_lip_drop': 0.06, 'teeth_visibility': 0.4,
                'tongue_position': 0.0, 'tongue_height': 0.5, 'jaw_opening': 0.12
            },
            
            # 静音 (保持自然闭合状态)
            'sil': {
                'lip_opening': 0.01, 'lip_width': 0.0, 'lip_protrusion': 0.0,
                'upper_lip_raise': 0.0, 'lower_lip_drop': 0.0, 'teeth_visibility': 0.0,
                'tongue_position': 0.0, 'tongue_height': 0.0, 'jaw_opening': 0.01
            }
        }
        
        return mapping
    
    def get_animation_params_for_phoneme(self, phoneme: str, intensity: float = 1.0) -> Dict[str, float]:
        """
        获取指定音素的动画参数
        
        Args:
            phoneme: 音素标识
            intensity: 动画强度 [0-1]
            
        Returns:
            动画参数字典
        """
        if phoneme not in self.phoneme_animation_mapping:
            phoneme = 'sil'  # 默认静音
            
        params = self.phoneme_animation_mapping[phoneme].copy()
        
        # 应用强度调节
        for key in params:
            params[key] *= intensity
            
        return params
    
    def interpolate_animation_params(self, 
                                   phoneme_sequence: List[str], 
                                   durations: List[float],
                                   target_fps: int = 25) -> List[Dict[str, float]]:
        """
        为音素序列生成平滑的动画参数序列
        
        Args:
            phoneme_sequence: 音素序列
            durations: 每个音素的持续时间（秒）
            target_fps: 目标帧率
            
        Returns:
            每帧的动画参数列表
        """
        if len(phoneme_sequence) != len(durations):
            raise ValueError("音素序列和持续时间长度不匹配")
        
        # 计算关键帧时间点
        key_times = [0]
        for duration in durations:
            key_times.append(key_times[-1] + duration)
        
        # 获取关键帧动画参数
        key_params = []
        for phoneme in phoneme_sequence:
            key_params.append(self.get_animation_params_for_phoneme(phoneme))
        
        # 添加结束帧（与最后一个音素相同）
        key_params.append(key_params[-1])
        
        # 生成目标时间序列
        total_duration = key_times[-1]
        frame_times = np.arange(0, total_duration, 1.0 / target_fps)
        
        # 为每个参数创建插值函数
        interpolated_params = []
        param_names = list(self.animation_params.keys())
        
        for frame_time in frame_times:
            frame_params = {}
            
            for param_name in param_names:
                # 提取该参数在所有关键帧的值
                param_values = [params.get(param_name, 0.0) for params in key_params]
                
                # 创建插值函数
                if len(set(param_values)) == 1:  # 所有值相同
                    frame_params[param_name] = param_values[0]
                else:
                    interp_func = interp1d(key_times, param_values, 
                                         kind='cubic', bounds_error=False, 
                                         fill_value='extrapolate')
                    frame_params[param_name] = float(interp_func(frame_time))
            
            interpolated_params.append(frame_params)
        
        return interpolated_params
    
    def apply_temporal_smoothing(self, params_sequence: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """
        应用时序平滑处理
        
        Args:
            params_sequence: 原始参数序列
            
        Returns:
            平滑后的参数序列
        """
        if len(params_sequence) < self.smoothing_window:
            return params_sequence
        
        smoothed_sequence = []
        param_names = list(self.animation_params.keys())
        
        for i in range(len(params_sequence)):
            # 确定平滑窗口范围
            start_idx = max(0, i - self.smoothing_window // 2)
            end_idx = min(len(params_sequence), i + self.smoothing_window // 2 + 1)
            
            # 计算窗口内参数的平均值
            smoothed_params = {}
            for param_name in param_names:
                values = [params_sequence[j][param_name] for j in range(start_idx, end_idx)]
                smoothed_params[param_name] = np.mean(values)
            
            smoothed_sequence.append(smoothed_params)
        
        return smoothed_sequence
    
    def apply_animation_to_landmarks(self, 
                                   landmarks: np.ndarray, 
                                   animation_params: Dict[str, float]) -> np.ndarray:
        """
        将动画参数应用到人脸关键点
        
        Args:
            landmarks: 原始68点人脸关键点 [68, 2]
            animation_params: 动画参数
            
        Returns:
            变换后的关键点
        """
        modified_landmarks = landmarks.copy()
        
        # 获取嘴部关键点
        mouth_points = modified_landmarks[self.mouth_landmarks['outer_lip']]
        mouth_center = np.mean(mouth_points, axis=0)
        
        # 应用嘴唇开合度
        lip_opening = animation_params.get('lip_opening', 0.0)
        upper_lip_indices = self.mouth_landmarks['upper_lip']
        lower_lip_indices = self.mouth_landmarks['lower_lip']
        
        # 上唇向上移动
        for idx in upper_lip_indices:
            modified_landmarks[idx, 1] -= lip_opening * 5
        
        # 下唇向下移动
        for idx in lower_lip_indices:
            modified_landmarks[idx, 1] += lip_opening * 5
        
        # 应用嘴唇宽度变化
        lip_width = animation_params.get('lip_width', 0.0)
        for idx in self.mouth_landmarks['outer_lip']:
            # 相对于嘴部中心的水平偏移
            offset_x = modified_landmarks[idx, 0] - mouth_center[0]
            modified_landmarks[idx, 0] += offset_x * lip_width * 0.3
        
        # 应用嘴唇突出度
        lip_protrusion = animation_params.get('lip_protrusion', 0.0)
        if lip_protrusion != 0:
            # 这里需要3D信息，暂时用Y轴偏移模拟
            for idx in self.mouth_landmarks['outer_lip']:
                modified_landmarks[idx, 1] += lip_protrusion * 2
        
        # 应用上唇抬起
        upper_lip_raise = animation_params.get('upper_lip_raise', 0.0)
        for idx in upper_lip_indices:
            modified_landmarks[idx, 1] -= upper_lip_raise * 3
        
        # 应用下唇下降
        lower_lip_drop = animation_params.get('lower_lip_drop', 0.0)
        for idx in lower_lip_indices:
            modified_landmarks[idx, 1] += lower_lip_drop * 3
        
        # 应用嘴部不对称
        mouth_asymmetry = animation_params.get('mouth_asymmetry', 0.0)
        if mouth_asymmetry != 0:
            for idx in self.mouth_landmarks['outer_lip']:
                # 左右不对称变形
                if modified_landmarks[idx, 0] < mouth_center[0]:  # 左侧
                    modified_landmarks[idx, 1] += mouth_asymmetry * 2
                else:  # 右侧
                    modified_landmarks[idx, 1] -= mouth_asymmetry * 2
        
        return modified_landmarks
    
    def generate_mouth_mask(self, 
                          image_shape: Tuple[int, int], 
                          landmarks: np.ndarray,
                          animation_params: Dict[str, float]) -> np.ndarray:
        """
        生成嘴部区域遮罩，用于牙齿和舌头渲染
        
        Args:
            image_shape: 图像尺寸 (H, W)
            landmarks: 人脸关键点
            animation_params: 动画参数
            
        Returns:
            嘴部遮罩 [H, W]
        """
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # 获取嘴部轮廓点
        mouth_points = landmarks[self.mouth_landmarks['outer_lip']].astype(np.int32)
        
        # 创建嘴部轮廓遮罩
        cv2.fillPoly(mask, [mouth_points], 255)
        
        # 根据嘴唇开合度调整遮罩
        lip_opening = animation_params.get('lip_opening', 0.0)
        if lip_opening > 0.3:  # 嘴巴张开时显示内部
            inner_mouth_points = landmarks[self.mouth_landmarks['inner_lip']].astype(np.int32)
            cv2.fillPoly(mask, [inner_mouth_points], 128)  # 内部区域用不同值标记
        
        return mask
    
    def add_breathing_effect(self, 
                           animation_params: Dict[str, float], 
                           frame_index: int,
                           breathing_rate: float = 0.2) -> Dict[str, float]:
        """
        添加呼吸效果到动画参数
        
        Args:
            animation_params: 原始动画参数
            frame_index: 当前帧索引
            breathing_rate: 呼吸频率
            
        Returns:
            添加呼吸效果后的参数
        """
        modified_params = animation_params.copy()
        
        # 计算呼吸周期 - 更自然的呼吸频率
        breathing_phase = np.sin(frame_index * breathing_rate * 2 * np.pi / 25)  # 25fps
        breathing_amplitude = 0.02  # 降低呼吸幅度，更加细腻
        
        # 应用呼吸效果到相关参数 - 更细微的变化
        modified_params['lip_opening'] += breathing_phase * breathing_amplitude * 0.8
        modified_params['jaw_opening'] += breathing_phase * breathing_amplitude * 0.3
        
        # 确保参数在有效范围内
        for key, value in modified_params.items():
            if 'opening' in key or 'raise' in key or 'drop' in key or 'visibility' in key or 'height' in key:
                modified_params[key] = np.clip(value, 0.0, 1.0)
            else:
                modified_params[key] = np.clip(value, -1.0, 1.0)
        
        return modified_params
    
    def process_phoneme_sequence(self, 
                               phoneme_sequence: List[str],
                               durations: List[float],
                               target_fps: int = 25,
                               add_breathing: bool = True) -> List[Dict[str, float]]:
        """
        处理完整的音素序列，生成动画参数序列
        
        Args:
            phoneme_sequence: 音素序列
            durations: 持续时间序列
            target_fps: 目标帧率
            add_breathing: 是否添加呼吸效果
            
        Returns:
            完整的动画参数序列
        """
        # 生成插值参数序列
        params_sequence = self.interpolate_animation_params(
            phoneme_sequence, durations, target_fps
        )
        
        # 应用时序平滑
        params_sequence = self.apply_temporal_smoothing(params_sequence)
        
        # 添加呼吸效果
        if add_breathing:
            for i, params in enumerate(params_sequence):
                params_sequence[i] = self.add_breathing_effect(params, i)
        
        return params_sequence


class TeethAndTongueRenderer:
    """
    牙齿和舌头渲染器
    根据动画参数渲染牙齿和舌头
    """
    
    def __init__(self):
        # 牙齿模板（简化的白色矩形）
        self.teeth_template = self._create_teeth_template()
        # 舌头模板（简化的粉色椭圆）
        self.tongue_template = self._create_tongue_template()
    
    def _create_teeth_template(self) -> np.ndarray:
        """
        创建牙齿模板
        """
        template = np.ones((20, 60, 3), dtype=np.uint8) * 240  # 接近白色
        return template
    
    def _create_tongue_template(self) -> np.ndarray:
        """
        创建舌头模板
        """
        template = np.ones((30, 40, 3), dtype=np.uint8)
        template[:, :, 0] = 200  # R
        template[:, :, 1] = 150  # G
        template[:, :, 2] = 150  # B
        return template
    
    def render_teeth(self, 
                    image: np.ndarray,
                    mouth_landmarks: np.ndarray,
                    teeth_visibility: float) -> np.ndarray:
        """
        渲染牙齿
        
        Args:
            image: 输入图像
            mouth_landmarks: 嘴部关键点
            teeth_visibility: 牙齿可见度 [0-1]
            
        Returns:
            渲染后的图像
        """
        if teeth_visibility <= 0.1:
            return image
        
        result_image = image.copy()
        
        # 计算牙齿位置（上唇下方）
        upper_lip_points = mouth_landmarks[[49, 50, 51, 52, 53]]
        teeth_y = int(np.mean(upper_lip_points[:, 1]) + 5)
        teeth_x = int(np.mean(upper_lip_points[:, 0]) - 30)
        
        # 调整牙齿大小和透明度
        teeth_height = int(20 * teeth_visibility)
        teeth_width = int(60 * min(1.0, teeth_visibility * 1.5))
        
        if teeth_height > 0 and teeth_width > 0:
            teeth_resized = cv2.resize(self.teeth_template, (teeth_width, teeth_height))
            
            # 确保不超出图像边界
            y1 = max(0, teeth_y)
            y2 = min(image.shape[0], teeth_y + teeth_height)
            x1 = max(0, teeth_x)
            x2 = min(image.shape[1], teeth_x + teeth_width)
            
            if y2 > y1 and x2 > x1:
                # 调整模板大小以匹配实际区域
                actual_height = y2 - y1
                actual_width = x2 - x1
                teeth_final = cv2.resize(teeth_resized, (actual_width, actual_height))
                
                # 应用透明度混合
                alpha = teeth_visibility * 0.8
                result_image[y1:y2, x1:x2] = (
                    alpha * teeth_final + (1 - alpha) * result_image[y1:y2, x1:x2]
                ).astype(np.uint8)
        
        return result_image
    
    def render_tongue(self, 
                     image: np.ndarray,
                     mouth_landmarks: np.ndarray,
                     tongue_position: float,
                     tongue_height: float) -> np.ndarray:
        """
        渲染舌头
        
        Args:
            image: 输入图像
            mouth_landmarks: 嘴部关键点
            tongue_position: 舌头位置 [-1,1]
            tongue_height: 舌头高度 [0,1]
            
        Returns:
            渲染后的图像
        """
        if tongue_height <= 0.1:
            return image
        
        result_image = image.copy()
        
        # 计算舌头位置
        mouth_center = np.mean(mouth_landmarks, axis=0)
        tongue_x = int(mouth_center[0] - 20 + tongue_position * 10)
        tongue_y = int(mouth_center[1] - tongue_height * 15)
        
        # 调整舌头大小
        tongue_height_px = int(30 * tongue_height)
        tongue_width_px = int(40 * (1.0 - abs(tongue_position) * 0.3))
        
        if tongue_height_px > 0 and tongue_width_px > 0:
            tongue_resized = cv2.resize(self.tongue_template, (tongue_width_px, tongue_height_px))
            
            # 确保不超出图像边界
            y1 = max(0, tongue_y)
            y2 = min(image.shape[0], tongue_y + tongue_height_px)
            x1 = max(0, tongue_x)
            x2 = min(image.shape[1], tongue_x + tongue_width_px)
            
            if y2 > y1 and x2 > x1:
                # 调整模板大小以匹配实际区域
                actual_height = y2 - y1
                actual_width = x2 - x1
                tongue_final = cv2.resize(tongue_resized, (actual_width, actual_height))
                
                # 应用透明度混合
                alpha = 0.6
                result_image[y1:y2, x1:x2] = (
                    alpha * tongue_final + (1 - alpha) * result_image[y1:y2, x1:x2]
                ).astype(np.uint8)
        
        return result_image


# 使用示例
if __name__ == "__main__":
    # 创建动画控制器
    controller = DetailedMouthAnimationController()
    
    # 测试音素序列
    phonemes = ['n', 'i', 'h', 'ao']  # "你好"
    durations = [0.2, 0.3, 0.2, 0.4]
    
    # 生成动画参数序列
    animation_sequence = controller.process_phoneme_sequence(
        phonemes, durations, target_fps=25, add_breathing=True
    )
    
    print(f"生成了 {len(animation_sequence)} 帧的动画参数")
    print(f"第一帧参数: {animation_sequence[0]}")
    
    # 创建渲染器
    renderer = TeethAndTongueRenderer()
    print("牙齿和舌头渲染器已初始化")