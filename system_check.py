#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
系统硬件配置检测和优化建议脚本
适用于2D数字人项目的Windows部署

使用方法:
    python system_check.py
"""

import os
import sys
import platform
import psutil
import subprocess
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

class SystemChecker:
    def __init__(self):
        self.results = {
            'system_info': {},
            'hardware_check': {},
            'software_check': {},
            'performance_test': {},
            'recommendations': []
        }
    
    def check_system_info(self):
        """检查系统基本信息"""
        print("🔍 检查系统信息...")
        
        system_info = {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.architecture()[0],
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
        
        self.results['system_info'] = system_info
        
        print(f"   操作系统: {system_info['os']} {system_info['os_version']}")
        print(f"   架构: {system_info['architecture']}")
        print(f"   Python版本: {system_info['python_version']}")
        
        return system_info
    
    def check_hardware(self):
        """检查硬件配置"""
        print("\n🔧 检查硬件配置...")
        
        # CPU信息
        cpu_count = psutil.cpu_count(logical=False)
        cpu_count_logical = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_gb = round(memory.total / (1024**3), 2)
        
        # 磁盘信息
        disk_usage = psutil.disk_usage('/')
        disk_free_gb = round(disk_usage.free / (1024**3), 2)
        
        hardware_info = {
            'cpu_cores_physical': cpu_count,
            'cpu_cores_logical': cpu_count_logical,
            'cpu_frequency_mhz': cpu_freq.current if cpu_freq else 0,
            'memory_total_gb': memory_gb,
            'memory_available_gb': round(memory.available / (1024**3), 2),
            'disk_free_gb': disk_free_gb
        }
        
        self.results['hardware_check'] = hardware_info
        
        print(f"   CPU核心: {cpu_count}物理核心, {cpu_count_logical}逻辑核心")
        print(f"   CPU频率: {cpu_freq.current:.0f} MHz" if cpu_freq else "   CPU频率: 未知")
        print(f"   内存: {memory_gb:.1f}GB 总计, {hardware_info['memory_available_gb']:.1f}GB 可用")
        print(f"   磁盘空间: {disk_free_gb:.1f}GB 可用")
        
        return hardware_info
    
    def check_gpu(self):
        """检查GPU配置"""
        print("\n🎮 检查GPU配置...")
        
        gpu_info = {
            'cuda_available': False,
            'gpu_count': 0,
            'gpu_names': [],
            'gpu_memory': []
        }
        
        if TORCH_AVAILABLE:
            gpu_info['cuda_available'] = torch.cuda.is_available()
            if gpu_info['cuda_available']:
                gpu_info['gpu_count'] = torch.cuda.device_count()
                for i in range(gpu_info['gpu_count']):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_info['gpu_names'].append(gpu_name)
                    gpu_info['gpu_memory'].append(round(gpu_memory, 2))
                    print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
            else:
                print("   CUDA不可用，将使用CPU模式")
        else:
            print("   PyTorch未安装，无法检测GPU")
        
        self.results['hardware_check']['gpu'] = gpu_info
        return gpu_info
    
    def check_software_dependencies(self):
        """检查软件依赖"""
        print("\n📦 检查软件依赖...")
        
        dependencies = {
            'torch': TORCH_AVAILABLE,
            'opencv': OPENCV_AVAILABLE,
            'gradio': self._check_package('gradio'),
            'mediapipe': self._check_package('mediapipe'),
            'numpy': self._check_package('numpy'),
            'scipy': self._check_package('scipy'),
            'scikit-learn': self._check_package('sklearn'),
            'kaldi_native_fbank': self._check_package('kaldi_native_fbank')
        }
        
        self.results['software_check'] = dependencies
        
        for package, available in dependencies.items():
            status = "✅" if available else "❌"
            print(f"   {status} {package}")
        
        return dependencies
    
    def _check_package(self, package_name):
        """检查Python包是否安装"""
        try:
            __import__(package_name)
            return True
        except ImportError:
            return False
    
    def performance_test(self):
        """简单的性能测试"""
        print("\n⚡ 进行性能测试...")
        
        import time
        import numpy as np
        
        # CPU测试
        print("   测试CPU性能...")
        start_time = time.time()
        
        # 矩阵运算测试
        for _ in range(100):
            a = np.random.rand(500, 500)
            b = np.random.rand(500, 500)
            c = np.dot(a, b)
        
        cpu_time = time.time() - start_time
        
        performance_info = {
            'cpu_matrix_test_time': round(cpu_time, 3)
        }
        
        # GPU测试（如果可用）
        if TORCH_AVAILABLE and torch.cuda.is_available():
            print("   测试GPU性能...")
            device = torch.device('cuda')
            start_time = time.time()
            
            for _ in range(100):
                a = torch.rand(500, 500, device=device)
                b = torch.rand(500, 500, device=device)
                c = torch.mm(a, b)
                torch.cuda.synchronize()
            
            gpu_time = time.time() - start_time
            performance_info['gpu_matrix_test_time'] = round(gpu_time, 3)
            
            print(f"   CPU矩阵运算: {cpu_time:.3f}秒")
            print(f"   GPU矩阵运算: {gpu_time:.3f}秒")
            print(f"   GPU加速比: {cpu_time/gpu_time:.1f}x")
        else:
            print(f"   CPU矩阵运算: {cpu_time:.3f}秒")
        
        self.results['performance_test'] = performance_info
        return performance_info
    
    def generate_recommendations(self):
        """生成优化建议"""
        print("\n💡 生成优化建议...")
        
        recommendations = []
        
        # 检查CPU
        cpu_cores = self.results['hardware_check']['cpu_cores_physical']
        if cpu_cores < 6:
            recommendations.append({
                'type': 'warning',
                'message': f'CPU核心数({cpu_cores})低于推荐配置(6核心+)，可能影响性能'
            })
        
        # 检查内存
        memory_gb = self.results['hardware_check']['memory_total_gb']
        if memory_gb < 8:
            recommendations.append({
                'type': 'error',
                'message': f'内存容量({memory_gb:.1f}GB)低于最低要求(8GB)，建议升级内存'
            })
        elif memory_gb < 16:
            recommendations.append({
                'type': 'info',
                'message': f'内存容量({memory_gb:.1f}GB)满足最低要求，建议升级到16GB以获得更好性能'
            })
        
        # 检查磁盘空间
        disk_free = self.results['hardware_check']['disk_free_gb']
        if disk_free < 5:
            recommendations.append({
                'type': 'error',
                'message': f'磁盘可用空间({disk_free:.1f}GB)不足，建议清理磁盘空间'
            })
        elif disk_free < 10:
            recommendations.append({
                'type': 'warning',
                'message': f'磁盘可用空间({disk_free:.1f}GB)较少，建议预留更多空间'
            })
        
        # 检查GPU
        if 'gpu' in self.results['hardware_check']:
            gpu_info = self.results['hardware_check']['gpu']
            if not gpu_info['cuda_available']:
                recommendations.append({
                    'type': 'info',
                    'message': '未检测到CUDA支持，将使用CPU模式运行，性能可能较慢'
                })
            elif gpu_info['gpu_count'] > 0:
                for i, memory in enumerate(gpu_info['gpu_memory']):
                    if memory < 2:
                        recommendations.append({
                            'type': 'warning',
                            'message': f'GPU {i} 显存({memory:.1f}GB)较少，可能影响大模型运行'
                        })
        
        # 检查软件依赖
        missing_packages = []
        for package, available in self.results['software_check'].items():
            if not available:
                missing_packages.append(package)
        
        if missing_packages:
            recommendations.append({
                'type': 'error',
                'message': f'缺少必要依赖包: {", ".join(missing_packages)}，请运行 pip install -r requirements.txt'
            })
        
        # 性能建议
        if 'cpu_matrix_test_time' in self.results['performance_test']:
            cpu_time = self.results['performance_test']['cpu_matrix_test_time']
            if cpu_time > 5.0:
                recommendations.append({
                    'type': 'warning',
                    'message': f'CPU性能测试耗时({cpu_time:.1f}秒)较长，建议升级CPU或启用GPU加速'
                })
        
        self.results['recommendations'] = recommendations
        
        # 打印建议
        for rec in recommendations:
            icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}[rec['type']]
            print(f"   {icon} {rec['message']}")
        
        if not recommendations:
            print("   ✅ 系统配置良好，无需特别优化")
        
        return recommendations
    
    def save_report(self, filename='system_check_report.json'):
        """保存检测报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n📄 检测报告已保存到: {filename}")
    
    def run_full_check(self):
        """运行完整的系统检测"""
        print("🚀 开始系统硬件配置检测...\n")
        
        self.check_system_info()
        self.check_hardware()
        self.check_gpu()
        self.check_software_dependencies()
        self.performance_test()
        self.generate_recommendations()
        
        print("\n✅ 系统检测完成！")
        
        # 保存报告
        self.save_report()
        
        return self.results

def main():
    """主函数"""
    print("="*60)
    print("    2D数字人项目 - Windows系统配置检测工具")
    print("="*60)
    
    checker = SystemChecker()
    results = checker.run_full_check()
    
    print("\n" + "="*60)
    print("检测完成！请查看上述建议进行系统优化。")
    print("如需技术支持，请将生成的报告文件一并提供。")
    print("="*60)

if __name__ == '__main__':
    main()