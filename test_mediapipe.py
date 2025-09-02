#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe测试脚本
用于验证MediaPipe是否能正常导入和使用
"""

import sys
import os

print("=== MediaPipe 测试开始 ===")
print(f"Python版本: {sys.version}")
print(f"当前工作目录: {os.getcwd()}")
print(f"虚拟环境: {sys.prefix}")
print()

try:
    print("正在导入 MediaPipe...")
    import mediapipe as mp
    print(f"✅ MediaPipe 导入成功! 版本: {mp.__version__}")
    
    # 测试基本功能
    print("\n正在测试 MediaPipe 基本功能...")
    
    # 测试人脸检测
    print("测试人脸检测模块...")
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils
    print("✅ 人脸检测模块加载成功")
    
    # 测试手部检测
    print("测试手部检测模块...")
    mp_hands = mp.solutions.hands
    print("✅ 手部检测模块加载成功")
    
    # 测试姿态检测
    print("测试姿态检测模块...")
    mp_pose = mp.solutions.pose
    print("✅ 姿态检测模块加载成功")
    
    # 测试人脸网格
    print("测试人脸网格模块...")
    mp_face_mesh = mp.solutions.face_mesh
    print("✅ 人脸网格模块加载成功")
    
    print("\n=== MediaPipe 所有测试通过! ===")
    print("MediaPipe 已成功安装并可以正常使用")
    
except ImportError as e:
    print(f"❌ MediaPipe 导入失败: {e}")
    print("\n可能的解决方案:")
    print("1. 确保已安装 Microsoft Visual C++ Redistributable")
    print("2. 重新安装 MediaPipe: pip uninstall mediapipe && pip install mediapipe")
    print("3. 检查Python版本兼容性")
    
except Exception as e:
    print(f"❌ MediaPipe 测试过程中出现错误: {e}")
    print(f"错误类型: {type(e).__name__}")
    
print("\n=== 测试结束 ===")
input("按回车键退出...")