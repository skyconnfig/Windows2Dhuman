#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时推理速度优化模块
包含模型量化、推理加速、内存优化和批处理优化
针对数字人唇形生成的实时性能优化
"""

import torch
import torch.nn as nn
import torch.quantization as quantization
from torch.jit import script, trace
import numpy as np
import time
import os
from collections import deque
import threading
from typing import Optional, Tuple, List
import logging

# 导入项目模块
from models.DINet import DINet_five_Ref
from models.DINet_with_attention import DINet_five_Ref_with_Attention

class InferenceOptimizer:
    """
    推理速度优化器
    提供模型量化、JIT编译、内存优化等功能
    """
    
    def __init__(self, model_path: str, device: str = 'auto'):
        self.device = self._setup_device(device)
        self.model_path = model_path
        self.original_model = None
        self.optimized_model = None
        self.quantized_model = None
        self.jit_model = None
        
        # 性能统计
        self.inference_times = deque(maxlen=100)
        self.memory_usage = deque(maxlen=50)
        
        self._setup_logging()
        
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                # 启用CUDA优化
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
            else:
                device = 'cpu'
                # 启用CPU优化
                torch.set_num_threads(4)
        
        return torch.device(device)
        
    def _setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_model(self, use_attention: bool = True) -> nn.Module:
        """加载原始模型"""
        self.logger.info(f"加载模型: {self.model_path}")
        
        if use_attention:
            # 加载带注意力机制的模型
            attention_config = {
                'spatial_attention': True,
                'mouth_focused_attention': True,
                'multi_scale_attention': False,  # 推理时关闭多尺度以提升速度
                'adaptive_fusion': True,
                'mouth_region_weight': 8.0
            }
            model = DINet_five_Ref_with_Attention(
                source_channel=3,
                ref_channel=15,
                attention_config=attention_config
            )
        else:
            # 加载基础模型
            model = DINet_five_Ref(source_channel=3, ref_channel=15)
        
        # 加载权重
        if os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        self.original_model = model
        return model
        
    def optimize_model_structure(self) -> nn.Module:
        """优化模型结构以提升推理速度"""
        if self.original_model is None:
            raise ValueError("请先加载模型")
            
        self.logger.info("优化模型结构...")
        
        # 创建优化版本的模型
        optimized_model = self._create_inference_optimized_model()
        
        # 复制权重
        self._transfer_weights(self.original_model, optimized_model)
        
        self.optimized_model = optimized_model
        return optimized_model
        
    def _create_inference_optimized_model(self) -> nn.Module:
        """创建推理优化的模型结构"""
        class InferenceOptimizedDINet(nn.Module):
            """推理优化的DINet模型"""
            
            def __init__(self, original_model):
                super().__init__()
                
                # 复制核心组件，移除训练时的额外模块
                self.source_in_conv = original_model.source_in_conv
                self.ref_in_conv = original_model.ref_in_conv
                self.trans_conv = original_model.trans_conv
                
                # 简化的注意力机制（如果存在）
                if hasattr(original_model, 'attention_modules'):
                    # 只保留最重要的注意力模块
                    self.mouth_attention = original_model.attention_modules.get('mouth_focused')
                else:
                    self.mouth_attention = None
                
                # 核心处理模块
                self.appearance_conv_list = original_model.appearance_conv_list
                self.AdaAT_list = original_model.AdaAT_list
                self.out_conv = original_model.out_conv
                
                # 缓存常用计算
                self.register_buffer('ref_features_cache', torch.zeros(1, 512, 16, 16))
                self.cache_valid = False
                
            def forward(self, source_img, ref_imgs=None, use_cache=True):
                """优化的前向传播"""
                batch_size = source_img.size(0)
                
                # 源图像编码
                source_feature = self.source_in_conv(source_img)
                
                # 参考图像特征（使用缓存）
                if ref_imgs is not None and (not use_cache or not self.cache_valid):
                    ref_feature = self.ref_in_conv(ref_imgs)
                    if use_cache:
                        self.ref_features_cache = ref_feature.detach()
                        self.cache_valid = True
                else:
                    ref_feature = self.ref_features_cache.expand(batch_size, -1, -1, -1)
                
                # 特征变换
                trans_feature = self.trans_conv(source_feature, ref_feature)
                
                # 应用简化的注意力机制
                if self.mouth_attention is not None:
                    trans_feature = self.mouth_attention(trans_feature)
                
                # 外观特征处理（优化循环）
                for i, (conv, adat) in enumerate(zip(self.appearance_conv_list, self.AdaAT_list)):
                    trans_feature = conv(trans_feature)
                    trans_feature = adat(trans_feature, ref_feature)
                
                # 输出
                result = self.out_conv(trans_feature)
                return result
                
            def clear_cache(self):
                """清除缓存"""
                self.cache_valid = False
        
        return InferenceOptimizedDINet(self.original_model)
        
    def _transfer_weights(self, source_model: nn.Module, target_model: nn.Module):
        """转移模型权重"""
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        
        # 只转移匹配的权重
        filtered_dict = {
            k: v for k, v in source_dict.items() 
            if k in target_dict and v.shape == target_dict[k].shape
        }
        
        target_dict.update(filtered_dict)
        target_model.load_state_dict(target_dict)
        
        self.logger.info(f"转移了 {len(filtered_dict)} 个权重参数")
        
    def quantize_model(self, quantization_type: str = 'dynamic') -> nn.Module:
        """模型量化"""
        if self.optimized_model is None:
            model_to_quantize = self.original_model
        else:
            model_to_quantize = self.optimized_model
            
        self.logger.info(f"开始模型量化: {quantization_type}")
        
        if quantization_type == 'dynamic':
            # 动态量化（推荐用于推理）
            quantized_model = quantization.quantize_dynamic(
                model_to_quantize,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
        elif quantization_type == 'static':
            # 静态量化（需要校准数据）
            model_to_quantize.qconfig = quantization.get_default_qconfig('fbgemm')
            quantized_model = quantization.prepare(model_to_quantize)
            # 这里需要用校准数据运行模型
            # calibrate_model(quantized_model, calibration_data)
            quantized_model = quantization.convert(quantized_model)
        else:
            raise ValueError(f"不支持的量化类型: {quantization_type}")
        
        self.quantized_model = quantized_model
        return quantized_model
        
    def compile_jit_model(self, example_inputs: Tuple[torch.Tensor, ...]) -> torch.jit.ScriptModule:
        """JIT编译模型"""
        model_to_compile = self.quantized_model or self.optimized_model or self.original_model
        
        self.logger.info("JIT编译模型...")
        
        try:
            # 尝试脚本化
            jit_model = script(model_to_compile)
        except Exception as e:
            self.logger.warning(f"脚本化失败，尝试追踪: {e}")
            # 回退到追踪
            model_to_compile.eval()
            with torch.no_grad():
                jit_model = trace(model_to_compile, example_inputs)
        
        # 优化JIT模型
        jit_model = torch.jit.optimize_for_inference(jit_model)
        
        self.jit_model = jit_model
        return jit_model
        
    def benchmark_model(self, model: nn.Module, input_shape: Tuple[int, ...], 
                       num_runs: int = 100, warmup_runs: int = 10) -> dict:
        """模型性能基准测试"""
        self.logger.info(f"开始性能测试: {num_runs} 次运行")
        
        # 创建测试输入
        if len(input_shape) == 2:  # (source, ref)
            source_input = torch.randn(1, 3, input_shape[0], input_shape[1]).to(self.device)
            ref_input = torch.randn(1, 15, input_shape[0], input_shape[1]).to(self.device)
            test_inputs = (source_input, ref_input)
        else:
            test_inputs = (torch.randn(1, 3, *input_shape).to(self.device),)
        
        model.eval()
        
        # 预热
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(*test_inputs)
        
        # 同步GPU
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 性能测试
        inference_times = []
        memory_usage = []
        
        with torch.no_grad():
            for i in range(num_runs):
                # 记录内存使用
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                    memory_before = torch.cuda.memory_allocated()
                
                # 计时
                start_time = time.perf_counter()
                output = model(*test_inputs)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.perf_counter()
                
                inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                inference_times.append(inference_time)
                
                # 记录内存使用
                if self.device.type == 'cuda':
                    memory_after = torch.cuda.memory_allocated()
                    memory_usage.append((memory_after - memory_before) / 1024 / 1024)  # MB
        
        # 计算统计信息
        stats = {
            'mean_time_ms': np.mean(inference_times),
            'std_time_ms': np.std(inference_times),
            'min_time_ms': np.min(inference_times),
            'max_time_ms': np.max(inference_times),
            'fps': 1000 / np.mean(inference_times),
            'memory_mb': np.mean(memory_usage) if memory_usage else 0
        }
        
        return stats
        
    def save_optimized_model(self, save_path: str, model_type: str = 'jit'):
        """保存优化后的模型"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if model_type == 'jit' and self.jit_model is not None:
            self.jit_model.save(save_path)
        elif model_type == 'quantized' and self.quantized_model is not None:
            torch.save(self.quantized_model.state_dict(), save_path)
        elif model_type == 'optimized' and self.optimized_model is not None:
            torch.save(self.optimized_model.state_dict(), save_path)
        else:
            raise ValueError(f"无法保存模型类型: {model_type}")
        
        self.logger.info(f"保存优化模型: {save_path}")


class RealTimeInferenceEngine:
    """
    实时推理引擎
    支持批处理、异步处理和帧缓存
    """
    
    def __init__(self, model_path: str, max_batch_size: int = 4):
        self.optimizer = InferenceOptimizer(model_path)
        self.max_batch_size = max_batch_size
        
        # 加载和优化模型
        self.model = self.optimizer.load_model(use_attention=True)
        self.model = self.optimizer.optimize_model_structure()
        
        # 批处理队列
        self.batch_queue = deque()
        self.result_queue = deque()
        
        # 异步处理
        self.processing_thread = None
        self.is_running = False
        
    def start_async_processing(self):
        """启动异步处理线程"""
        if not self.is_running:
            self.is_running = True
            self.processing_thread = threading.Thread(target=self._async_process_loop)
            self.processing_thread.start()
            
    def stop_async_processing(self):
        """停止异步处理"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
            
    def _async_process_loop(self):
        """异步处理循环"""
        while self.is_running:
            if len(self.batch_queue) >= self.max_batch_size:
                # 处理批次
                batch_data = []
                for _ in range(self.max_batch_size):
                    if self.batch_queue:
                        batch_data.append(self.batch_queue.popleft())
                
                if batch_data:
                    results = self._process_batch(batch_data)
                    self.result_queue.extend(results)
            
            time.sleep(0.001)  # 短暂休眠
            
    def _process_batch(self, batch_data: List[dict]) -> List[torch.Tensor]:
        """处理批次数据"""
        # 组合批次输入
        source_batch = torch.stack([item['source'] for item in batch_data])
        ref_batch = torch.stack([item['reference'] for item in batch_data])
        
        # 推理
        with torch.no_grad():
            output_batch = self.model(source_batch, ref_batch)
        
        # 分离结果
        return [output_batch[i] for i in range(output_batch.size(0))]
        
    def infer_single(self, source_img: torch.Tensor, 
                    ref_imgs: torch.Tensor) -> torch.Tensor:
        """单帧推理"""
        with torch.no_grad():
            return self.model(source_img.unsqueeze(0), ref_imgs.unsqueeze(0)).squeeze(0)
            
    def add_to_batch_queue(self, source_img: torch.Tensor, ref_imgs: torch.Tensor):
        """添加到批处理队列"""
        self.batch_queue.append({
            'source': source_img,
            'reference': ref_imgs
        })
        
    def get_result(self) -> Optional[torch.Tensor]:
        """获取处理结果"""
        if self.result_queue:
            return self.result_queue.popleft()
        return None


def main():
    """主函数 - 演示优化流程"""
    model_path = "checkpoints/best_model.pth"
    
    print("=== 数字人唇形生成推理速度优化 ===")
    
    # 创建优化器
    optimizer = InferenceOptimizer(model_path)
    
    # 加载模型
    print("\n1. 加载原始模型...")
    original_model = optimizer.load_model(use_attention=True)
    
    # 结构优化
    print("\n2. 优化模型结构...")
    optimized_model = optimizer.optimize_model_structure()
    
    # 模型量化
    print("\n3. 模型量化...")
    quantized_model = optimizer.quantize_model('dynamic')
    
    # JIT编译
    print("\n4. JIT编译...")
    example_source = torch.randn(1, 3, 256, 256).to(optimizer.device)
    example_ref = torch.randn(1, 15, 256, 256).to(optimizer.device)
    jit_model = optimizer.compile_jit_model((example_source, example_ref))
    
    # 性能测试
    print("\n5. 性能基准测试...")
    
    models_to_test = [
        ("原始模型", original_model),
        ("结构优化", optimized_model),
        ("量化模型", quantized_model),
        ("JIT编译", jit_model)
    ]
    
    results = {}
    for name, model in models_to_test:
        print(f"\n测试 {name}...")
        stats = optimizer.benchmark_model(model, (256, 256), num_runs=50)
        results[name] = stats
        
        print(f"  平均推理时间: {stats['mean_time_ms']:.2f} ms")
        print(f"  FPS: {stats['fps']:.1f}")
        print(f"  内存使用: {stats['memory_mb']:.1f} MB")
    
    # 保存最优模型
    print("\n6. 保存优化模型...")
    optimizer.save_optimized_model("optimized_models/jit_model.pt", "jit")
    
    # 性能对比
    print("\n=== 性能对比总结 ===")
    baseline_fps = results["原始模型"]['fps']
    
    for name, stats in results.items():
        speedup = stats['fps'] / baseline_fps
        print(f"{name}: {stats['fps']:.1f} FPS (加速 {speedup:.2f}x)")
    
    print("\n优化建议:")
    print("1. 使用JIT编译模型获得最佳性能")
    print("2. 在GPU上启用混合精度训练")
    print("3. 考虑使用TensorRT进一步优化")
    print("4. 实时应用中使用批处理和异步处理")


if __name__ == '__main__':
    main()