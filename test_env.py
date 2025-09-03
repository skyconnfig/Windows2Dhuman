#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟环境测试脚本
测试Python虚拟环境是否正确配置
"""

import sys
import os
print("=== Python虚拟环境测试 ===")
print(f"Python版本: {sys.version}")
print(f"Python路径: {sys.executable}")
print(f"当前工作目录: {os.getcwd()}")
print()

# 测试基础库
print("=== 基础库测试 ===")
try:
    import numpy as np
    print(f"✅ NumPy {np.__version__} - 正常")
except ImportError as e:
    print(f"❌ NumPy导入失败: {e}")

try:
    import torch
    print(f"✅ PyTorch {torch.__version__} - 正常")
except ImportError as e:
    print(f"❌ PyTorch导入失败: {e}")

try:
    import cv2
    print(f"✅ OpenCV {cv2.__version__} - 正常")
except ImportError as e:
    print(f"❌ OpenCV导入失败: {e}")

try:
    import gradio as gr
    print(f"✅ Gradio {gr.__version__} - 正常")
except ImportError as e:
    print(f"❌ Gradio导入失败: {e}")

print()
print("=== 虚拟环境状态 ===")
if 'venv' in sys.executable:
    print("✅ 当前运行在虚拟环境中")
else:
    print("⚠️  可能未在虚拟环境中运行")

print("\n=== 测试完成 ===")
print("如果所有库都显示正常，说明虚拟环境配置成功！")