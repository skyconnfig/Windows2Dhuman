#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查MediaPipe模型文件目录
"""

import os
import mediapipe as mp

def check_mediapipe_models():
    """
    检查MediaPipe模型文件和目录结构
    """
    print("=== 检查MediaPipe模型文件 ===")
    
    # 获取MediaPipe安装路径
    mp_path = mp.__file__
    mp_dir = os.path.dirname(mp_path)
    print(f"MediaPipe目录: {mp_dir}")
    
    # 检查modules目录
    modules_dir = os.path.join(mp_dir, 'modules')
    print(f"\nModules目录: {modules_dir}")
    print(f"Modules目录存在: {os.path.exists(modules_dir)}")
    
    if os.path.exists(modules_dir):
        try:
            modules_content = os.listdir(modules_dir)
            print(f"Modules目录内容数量: {len(modules_content)}")
            print("前10个文件:")
            for i, item in enumerate(modules_content[:10]):
                print(f"  {i+1}. {item}")
        except Exception as e:
            print(f"无法读取modules目录: {e}")
    
    # 检查其他可能的模型目录
    possible_dirs = [
        os.path.join(mp_dir, 'data'),
        os.path.join(mp_dir, 'models'),
        os.path.join(mp_dir, 'framework'),
        os.path.join(mp_dir, 'python', 'solutions', 'models'),
    ]
    
    print("\n=== 检查其他可能的模型目录 ===")
    for dir_path in possible_dirs:
        exists = os.path.exists(dir_path)
        print(f"{dir_path}: {'存在' if exists else '不存在'}")
        if exists:
            try:
                content = os.listdir(dir_path)
                print(f"  内容数量: {len(content)}")
                if content:
                    print(f"  示例文件: {content[:3]}")
            except Exception as e:
                print(f"  无法读取: {e}")
    
    # 检查环境变量
    print("\n=== 检查相关环境变量 ===")
    env_vars = [
        'MEDIAPIPE_DISABLE_GPU',
        'GLOG_logtostderr',
        'GLOG_v',
        'TF_CPP_MIN_LOG_LEVEL'
    ]
    
    for var in env_vars:
        value = os.environ.get(var, '未设置')
        print(f"{var}: {value}")
    
    # 尝试获取MediaPipe内部路径信息
    print("\n=== 尝试获取内部路径信息 ===")
    try:
        from mediapipe.python.solution_base import SolutionBase
        print("✅ SolutionBase导入成功")
    except Exception as e:
        print(f"❌ SolutionBase导入失败: {e}")
    
    try:
        from mediapipe.python import solution_base
        print("✅ solution_base模块导入成功")
    except Exception as e:
        print(f"❌ solution_base模块导入失败: {e}")

if __name__ == "__main__":
    check_mediapipe_models()