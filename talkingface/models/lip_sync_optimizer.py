#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
唇形同步精度优化模块
实现音频与视觉的精确匹配，减少延迟和不自然感
包含实时音频处理、音素对齐、时序同步等功能
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import cv2
from typing import Dict, List, Tuple, Optional, Union
from collections import deque
import threading
import time
from scipy.signal import butter, filtfilt
from scipy.interpolate import interp1d
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioFeatureExtractor:
    """
    高精度音频特征提取器
    支持实时音频处理和特征提取
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 hop_length: int = 160,  # 10ms帧移
                 win_length: int = 400,  # 25ms窗长
                 n_mels: int = 80,
                 n_fft: int = 1024):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.n_fft = n_fft
        
        # 预计算梅尔滤波器组
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate, 
            n_fft=n_fft, 
            n_mels=n_mels,
            fmin=0,
            fmax=sample_rate//2
        )
        
        # 音频预处理滤波器
        self.lowpass_filter = self._create_lowpass_filter()
        
    def _create_lowpass_filter(self):
        """创建低通滤波器，去除高频噪声"""
        nyquist = self.sample_rate / 2
        # 确保截止频率不超过奈奎斯特频率的0.95倍，避免数字滤波器参数错误
        max_cutoff = nyquist * 0.95
        cutoff = min(8000, max_cutoff)  # 取8kHz和最大允许频率的较小值
        
        # 确保归一化频率在有效范围内 (0, 1)
        normalized_cutoff = cutoff / nyquist
        if normalized_cutoff >= 1.0:
            normalized_cutoff = 0.95
        elif normalized_cutoff <= 0.0:
            normalized_cutoff = 0.1
            
        b, a = butter(5, normalized_cutoff, btype='low')
        return (b, a)
    
    def preprocess_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        音频预处理：去噪、归一化、滤波
        
        Args:
            audio: 原始音频数据
            
        Returns:
            预处理后的音频
        """
        # 归一化
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # 低通滤波
        b, a = self.lowpass_filter
        audio = filtfilt(b, a, audio)
        
        # 预加重
        audio = np.append(audio[0], audio[1:] - 0.97 * audio[:-1])
        
        return audio
    
    def extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """
        提取梅尔频谱图
        
        Args:
            audio: 音频数据
            
        Returns:
            梅尔频谱图 [n_mels, n_frames]
        """
        # 短时傅里叶变换
        stft = librosa.stft(
            audio,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann'
        )
        
        # 功率谱
        magnitude = np.abs(stft) ** 2
        
        # 梅尔滤波
        mel_spec = np.dot(self.mel_basis, magnitude)
        
        # 对数变换
        mel_spec = np.log(mel_spec + 1e-8)
        
        return mel_spec
    
    def extract_prosodic_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        提取韵律特征：基频、能量、语速等
        
        Args:
            audio: 音频数据
            
        Returns:
            韵律特征字典
        """
        # 基频提取
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=self.sample_rate,
            hop_length=self.hop_length
        )
        
        # 能量计算
        energy = librosa.feature.rms(
            y=audio,
            hop_length=self.hop_length,
            frame_length=self.win_length
        )[0]
        
        # 零交叉率
        zcr = librosa.feature.zero_crossing_rate(
            audio,
            hop_length=self.hop_length,
            frame_length=self.win_length
        )[0]
        
        # 谱质心
        spectral_centroid = librosa.feature.spectral_centroid(
            y=audio,
            sr=self.sample_rate,
            hop_length=self.hop_length
        )[0]
        
        return {
            'f0': f0,
            'voiced_flag': voiced_flag,
            'voiced_probs': voiced_probs,
            'energy': energy,
            'zcr': zcr,
            'spectral_centroid': spectral_centroid
        }

class PhonemeAligner:
    """
    音素对齐器
    实现音频与音素序列的精确时间对齐
    """
    
    def __init__(self, sample_rate: int = 16000, hop_length: int = 160):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.frame_rate = sample_rate / hop_length  # 帧率
        
    def align_phonemes_with_audio(self, 
                                 audio: np.ndarray,
                                 phoneme_sequence: List[Tuple[str, float, float]]) -> List[Tuple[str, int, int]]:
        """
        将音素序列与音频帧对齐
        
        Args:
            audio: 音频数据
            phoneme_sequence: 音素序列 [(phoneme, start_time, duration), ...]
            
        Returns:
            对齐后的音素序列 [(phoneme, start_frame, end_frame), ...]
        """
        audio_duration = len(audio) / self.sample_rate
        total_frames = int(audio_duration * self.frame_rate)
        
        aligned_sequence = []
        
        for phoneme, start_time, duration in phoneme_sequence:
            # 转换为帧索引
            start_frame = int(start_time * self.frame_rate)
            end_frame = int((start_time + duration) * self.frame_rate)
            
            # 确保帧索引在有效范围内
            start_frame = max(0, min(start_frame, total_frames - 1))
            end_frame = max(start_frame + 1, min(end_frame, total_frames))
            
            aligned_sequence.append((phoneme, start_frame, end_frame))
        
        return aligned_sequence
    
    def refine_alignment_with_energy(self, 
                                   audio: np.ndarray,
                                   aligned_sequence: List[Tuple[str, int, int]],
                                   energy: np.ndarray) -> List[Tuple[str, int, int]]:
        """
        基于能量信息精细化音素对齐
        
        Args:
            audio: 音频数据
            aligned_sequence: 初始对齐序列
            energy: 能量特征
            
        Returns:
            精细化对齐序列
        """
        refined_sequence = []
        
        for phoneme, start_frame, end_frame in aligned_sequence:
            # 对于爆破音，寻找能量峰值
            if phoneme in ['p', 'b', 't', 'd', 'k', 'g']:
                # 在原始范围内寻找能量峰值
                search_start = max(0, start_frame - 5)
                search_end = min(len(energy), end_frame + 5)
                
                if search_end > search_start:
                    energy_segment = energy[search_start:search_end]
                    peak_idx = np.argmax(energy_segment) + search_start
                    
                    # 调整起始位置到能量峰值附近
                    start_frame = max(start_frame, peak_idx - 2)
                    end_frame = max(start_frame + 3, end_frame)
            
            # 对于元音，寻找稳定的能量区域
            elif phoneme in ['a', 'e', 'i', 'o', 'u']:
                if end_frame > start_frame + 5:
                    energy_segment = energy[start_frame:end_frame]
                    # 寻找能量相对稳定的区域
                    smooth_energy = np.convolve(energy_segment, np.ones(3)/3, mode='same')
                    stable_start = np.argmin(np.abs(smooth_energy - np.mean(smooth_energy)))
                    start_frame = start_frame + max(0, stable_start - 2)
            
            refined_sequence.append((phoneme, start_frame, end_frame))
        
        return refined_sequence

class TemporalSynchronizer:
    """
    时序同步器
    处理音频与视觉的时间同步，减少延迟
    """
    
    def __init__(self, 
                 target_fps: int = 25,
                 audio_frame_rate: float = 100.0,
                 lookahead_frames: int = 3):
        self.target_fps = target_fps
        self.audio_frame_rate = audio_frame_rate
        self.lookahead_frames = lookahead_frames
        
        # 时间同步参数
        self.video_frame_duration = 1.0 / target_fps  # 视频帧持续时间
        self.audio_frame_duration = 1.0 / audio_frame_rate  # 音频帧持续时间
        
        # 同步缓冲区
        self.sync_buffer = deque(maxlen=lookahead_frames * 2)
        
    def synchronize_audio_visual(self, 
                               audio_features: np.ndarray,
                               phoneme_alignment: List[Tuple[str, int, int]],
                               target_video_fps: int = 25) -> List[Dict[str, any]]:
        """
        同步音频特征与视频帧
        
        Args:
            audio_features: 音频特征 [n_frames, n_features]
            phoneme_alignment: 音素对齐信息
            target_video_fps: 目标视频帧率
            
        Returns:
            同步后的帧级特征
        """
        n_audio_frames = audio_features.shape[0]
        audio_duration = n_audio_frames / self.audio_frame_rate
        n_video_frames = int(audio_duration * target_video_fps)
        
        synchronized_frames = []
        
        for video_frame_idx in range(n_video_frames):
            # 计算对应的音频时间
            video_time = video_frame_idx / target_video_fps
            audio_frame_idx = int(video_time * self.audio_frame_rate)
            
            # 获取当前和未来几帧的音频特征（预测性同步）
            current_features = []
            for offset in range(-1, self.lookahead_frames):
                frame_idx = audio_frame_idx + offset
                if 0 <= frame_idx < n_audio_frames:
                    current_features.append(audio_features[frame_idx])
                else:
                    current_features.append(np.zeros_like(audio_features[0]))
            
            # 找到当前时间对应的音素
            current_phoneme = 'sil'
            for phoneme, start_frame, end_frame in phoneme_alignment:
                if start_frame <= audio_frame_idx < end_frame:
                    current_phoneme = phoneme
                    break
            
            # 计算音素在当前帧的进度
            phoneme_progress = 0.0
            for phoneme, start_frame, end_frame in phoneme_alignment:
                if start_frame <= audio_frame_idx < end_frame:
                    phoneme_progress = (audio_frame_idx - start_frame) / max(1, end_frame - start_frame)
                    break
            
            synchronized_frames.append({
                'video_frame': video_frame_idx,
                'audio_frame': audio_frame_idx,
                'audio_features': np.concatenate(current_features),
                'phoneme': current_phoneme,
                'phoneme_progress': phoneme_progress,
                'timestamp': video_time
            })
        
        return synchronized_frames
    
    def apply_temporal_smoothing(self, 
                               synchronized_frames: List[Dict[str, any]],
                               smoothing_window: int = 3) -> List[Dict[str, any]]:
        """
        应用时序平滑，减少帧间跳跃
        
        Args:
            synchronized_frames: 同步帧数据
            smoothing_window: 平滑窗口大小
            
        Returns:
            平滑后的帧数据
        """
        if len(synchronized_frames) < smoothing_window:
            return synchronized_frames
        
        smoothed_frames = []
        half_window = smoothing_window // 2
        
        for i, frame in enumerate(synchronized_frames):
            # 获取窗口范围内的帧
            start_idx = max(0, i - half_window)
            end_idx = min(len(synchronized_frames), i + half_window + 1)
            window_frames = synchronized_frames[start_idx:end_idx]
            
            # 对音频特征进行平滑
            feature_stack = np.stack([f['audio_features'] for f in window_frames])
            smoothed_features = np.mean(feature_stack, axis=0)
            
            # 创建平滑后的帧
            smoothed_frame = frame.copy()
            smoothed_frame['audio_features'] = smoothed_features
            smoothed_frame['smoothed'] = True
            
            smoothed_frames.append(smoothed_frame)
        
        return smoothed_frames

class LipSyncOptimizer:
    """
    唇形同步优化器主类
    整合音频处理、音素对齐、时序同步等功能
    """
    
    def __init__(self, 
                 sample_rate: int = 16000,
                 target_fps: int = 25,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.sample_rate = sample_rate
        self.target_fps = target_fps
        self.device = device
        
        # 初始化子模块
        self.feature_extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        self.phoneme_aligner = PhonemeAligner(sample_rate=sample_rate)
        self.temporal_synchronizer = TemporalSynchronizer(target_fps=target_fps)
        
        # 性能统计
        self.processing_times = deque(maxlen=100)
        
        logger.info(f"唇形同步优化器初始化完成 - 设备: {device}")
    
    def optimize_lip_sync(self, 
                         audio_path: str,
                         phoneme_sequence: List[Tuple[str, float, float]] = None) -> Dict[str, any]:
        """
        优化唇形同步
        
        Args:
            audio_path: 音频文件路径
            phoneme_sequence: 音素序列（可选）
            
        Returns:
            优化后的同步数据
        """
        start_time = time.time()
        
        try:
            # 1. 加载和预处理音频
            logger.info("加载音频文件...")
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            audio = self.feature_extractor.preprocess_audio(audio)
            
            # 2. 提取音频特征
            logger.info("提取音频特征...")
            mel_spec = self.feature_extractor.extract_mel_spectrogram(audio)
            prosodic_features = self.feature_extractor.extract_prosodic_features(audio)
            
            # 合并特征
            audio_features = np.concatenate([
                mel_spec.T,  # 转置为 [n_frames, n_mels]
                prosodic_features['energy'].reshape(-1, 1),
                prosodic_features['f0'].reshape(-1, 1),
                prosodic_features['zcr'].reshape(-1, 1)
            ], axis=1)
            
            # 3. 音素对齐（如果提供了音素序列）
            if phoneme_sequence:
                logger.info("执行音素对齐...")
                aligned_phonemes = self.phoneme_aligner.align_phonemes_with_audio(
                    audio, phoneme_sequence
                )
                # 基于能量精细化对齐
                aligned_phonemes = self.phoneme_aligner.refine_alignment_with_energy(
                    audio, aligned_phonemes, prosodic_features['energy']
                )
            else:
                # 生成默认音素序列
                audio_duration = len(audio) / self.sample_rate
                aligned_phonemes = [('sil', 0, int(audio_duration * 100))]
            
            # 4. 时序同步
            logger.info("执行时序同步...")
            synchronized_frames = self.temporal_synchronizer.synchronize_audio_visual(
                audio_features, aligned_phonemes, self.target_fps
            )
            
            # 5. 时序平滑
            logger.info("应用时序平滑...")
            smoothed_frames = self.temporal_synchronizer.apply_temporal_smoothing(
                synchronized_frames, smoothing_window=3
            )
            
            # 6. 计算同步质量指标
            sync_quality = self._calculate_sync_quality(smoothed_frames)
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            logger.info(f"唇形同步优化完成 - 耗时: {processing_time:.3f}s")
            
            return {
                'synchronized_frames': smoothed_frames,
                'audio_features': audio_features,
                'phoneme_alignment': aligned_phonemes,
                'sync_quality': sync_quality,
                'processing_time': processing_time,
                'audio_duration': len(audio) / self.sample_rate,
                'n_frames': len(smoothed_frames)
            }
            
        except Exception as e:
            logger.error(f"唇形同步优化失败: {e}")
            raise
    
    def _calculate_sync_quality(self, synchronized_frames: List[Dict[str, any]]) -> Dict[str, float]:
        """
        计算同步质量指标
        
        Args:
            synchronized_frames: 同步帧数据
            
        Returns:
            质量指标字典
        """
        if len(synchronized_frames) < 2:
            return {'temporal_consistency': 0.0, 'feature_smoothness': 0.0}
        
        # 时序一致性：检查帧间特征变化的平滑度
        feature_diffs = []
        for i in range(1, len(synchronized_frames)):
            prev_features = synchronized_frames[i-1]['audio_features']
            curr_features = synchronized_frames[i]['audio_features']
            diff = np.mean(np.abs(curr_features - prev_features))
            feature_diffs.append(diff)
        
        temporal_consistency = 1.0 / (1.0 + np.mean(feature_diffs))
        
        # 特征平滑度：检查特征的方差
        all_features = np.stack([f['audio_features'] for f in synchronized_frames])
        feature_variance = np.mean(np.var(all_features, axis=0))
        feature_smoothness = 1.0 / (1.0 + feature_variance)
        
        return {
            'temporal_consistency': float(temporal_consistency),
            'feature_smoothness': float(feature_smoothness),
            'overall_quality': float((temporal_consistency + feature_smoothness) / 2)
        }
    
    def get_performance_stats(self) -> Dict[str, float]:
        """
        获取性能统计信息
        
        Returns:
            性能统计字典
        """
        if not self.processing_times:
            return {'avg_processing_time': 0.0, 'min_time': 0.0, 'max_time': 0.0}
        
        times = list(self.processing_times)
        return {
            'avg_processing_time': float(np.mean(times)),
            'min_processing_time': float(np.min(times)),
            'max_processing_time': float(np.max(times)),
            'std_processing_time': float(np.std(times))
        }

class RealTimeLipSyncOptimizer:
    """
    实时唇形同步优化器
    支持流式音频处理和实时同步
    """
    
    def __init__(self, 
                 optimizer: LipSyncOptimizer,
                 buffer_size: int = 1600,  # 100ms at 16kHz
                 overlap_size: int = 320):  # 20ms overlap
        self.optimizer = optimizer
        self.buffer_size = buffer_size
        self.overlap_size = overlap_size
        
        # 实时处理缓冲区
        self.audio_buffer = deque(maxlen=buffer_size * 3)
        self.feature_buffer = deque(maxlen=50)
        self.result_buffer = deque(maxlen=10)
        
        # 处理线程
        self.processing_thread = None
        self.is_processing = False
        
    def start_realtime_processing(self):
        """启动实时处理"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._realtime_processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        logger.info("实时唇形同步处理已启动")
    
    def stop_realtime_processing(self):
        """停止实时处理"""
        self.is_processing = False
        if self.processing_thread:
            self.processing_thread.join()
        
        logger.info("实时唇形同步处理已停止")
    
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """添加音频块到处理队列"""
        self.audio_buffer.extend(audio_chunk)
    
    def get_latest_result(self) -> Optional[Dict[str, any]]:
        """获取最新的处理结果"""
        if self.result_buffer:
            return self.result_buffer.popleft()
        return None
    
    def _realtime_processing_loop(self):
        """实时处理循环"""
        while self.is_processing:
            try:
                # 检查是否有足够的音频数据
                if len(self.audio_buffer) >= self.buffer_size:
                    # 提取音频段
                    audio_segment = np.array(list(self.audio_buffer)[:self.buffer_size])
                    
                    # 预处理音频
                    processed_audio = self.optimizer.feature_extractor.preprocess_audio(audio_segment)
                    
                    # 提取特征
                    mel_spec = self.optimizer.feature_extractor.extract_mel_spectrogram(processed_audio)
                    prosodic_features = self.optimizer.feature_extractor.extract_prosodic_features(processed_audio)
                    
                    # 合并特征
                    features = np.concatenate([
                        mel_spec.T,
                        prosodic_features['energy'].reshape(-1, 1),
                        prosodic_features['f0'].reshape(-1, 1),
                        prosodic_features['zcr'].reshape(-1, 1)
                    ], axis=1)
                    
                    # 添加到结果缓冲区
                    result = {
                        'features': features,
                        'timestamp': time.time(),
                        'energy': prosodic_features['energy'],
                        'f0': prosodic_features['f0']
                    }
                    
                    self.result_buffer.append(result)
                    
                    # 移除已处理的音频（保留重叠部分）
                    for _ in range(self.buffer_size - self.overlap_size):
                        if self.audio_buffer:
                            self.audio_buffer.popleft()
                
                time.sleep(0.01)  # 10ms间隔
                
            except Exception as e:
                logger.error(f"实时处理错误: {e}")
                time.sleep(0.1)

# 使用示例
if __name__ == "__main__":
    # 创建优化器
    optimizer = LipSyncOptimizer()
    
    # 测试音频文件（需要替换为实际路径）
    audio_path = "test_audio.wav"
    
    # 示例音素序列
    phoneme_sequence = [
        ('n', 0.0, 0.2),
        ('i', 0.2, 0.3),
        ('h', 0.5, 0.2),
        ('ao', 0.7, 0.4)
    ]
    
    try:
        # 执行优化
        result = optimizer.optimize_lip_sync(audio_path, phoneme_sequence)
        
        print(f"处理完成:")
        print(f"- 音频时长: {result['audio_duration']:.2f}s")
        print(f"- 生成帧数: {result['n_frames']}")
        print(f"- 处理时间: {result['processing_time']:.3f}s")
        print(f"- 同步质量: {result['sync_quality']['overall_quality']:.3f}")
        
        # 性能统计
        stats = optimizer.get_performance_stats()
        print(f"- 平均处理时间: {stats['avg_processing_time']:.3f}s")
        
    except Exception as e:
        print(f"处理失败: {e}")
    
    # 实时处理示例
    print("\n启动实时处理测试...")
    realtime_optimizer = RealTimeLipSyncOptimizer(optimizer)
    realtime_optimizer.start_realtime_processing()
    
    # 模拟实时音频输入
    for i in range(10):
        # 生成模拟音频块
        audio_chunk = np.random.randn(160)  # 10ms at 16kHz
        realtime_optimizer.add_audio_chunk(audio_chunk)
        
        # 获取结果
        result = realtime_optimizer.get_latest_result()
        if result:
            print(f"实时处理结果 {i}: 特征维度 {result['features'].shape}")
        
        time.sleep(0.1)
    
    realtime_optimizer.stop_realtime_processing()
    print("实时处理测试完成")