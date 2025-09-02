#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
尝试修复MediaPipe路径问题
"""

import os
import sys
import mediapipe as mp

def fix_mediapipe_path():
    """
    尝试修复MediaPipe路径问题
    """
    print("=== 尝试修复MediaPipe路径问题 ===")
    
    # 方法1: 设置更多环境变量
    print("\n方法1: 设置环境变量")
    os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
    os.environ['GLOG_logtostderr'] = '1'
    os.environ['GLOG_v'] = '0'  # 减少日志输出
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    
    # 获取MediaPipe路径并添加到系统路径
    mp_dir = os.path.dirname(mp.__file__)
    modules_dir = os.path.join(mp_dir, 'modules')
    
    if modules_dir not in sys.path:
        sys.path.insert(0, modules_dir)
        print(f"✅ 添加modules目录到sys.path: {modules_dir}")
    
    # 方法2: 尝试直接导入并初始化
    print("\n方法2: 尝试直接初始化")
    try:
        # 先导入所有相关模块
        from mediapipe.python.solutions import face_detection
        from mediapipe.python import solution_base
        
        print("✅ 模块导入成功")
        
        # 尝试创建对象
        detector = face_detection.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5
        )
        print("✅ FaceDetection对象创建成功！")
        return detector
        
    except Exception as e:
        print(f"❌ 方法2失败: {e}")
    
    # 方法3: 尝试使用不同的参数
    print("\n方法3: 尝试不同参数")
    try:
        detector = face_detection.FaceDetection()
        print("✅ 使用默认参数创建成功！")
        return detector
    except Exception as e:
        print(f"❌ 方法3失败: {e}")
    
    # 方法4: 尝试重新导入
    print("\n方法4: 重新导入模块")
    try:
        # 清除模块缓存
        if 'mediapipe.python.solutions.face_detection' in sys.modules:
            del sys.modules['mediapipe.python.solutions.face_detection']
        if 'mediapipe.python.solution_base' in sys.modules:
            del sys.modules['mediapipe.python.solution_base']
        
        # 重新导入
        from mediapipe.python.solutions import face_detection
        detector = face_detection.FaceDetection(model_selection=0)
        print("✅ 重新导入后创建成功！")
        return detector
    except Exception as e:
        print(f"❌ 方法4失败: {e}")
    
    print("\n❌ 所有修复方法都失败了")
    return None

def test_opencv_alternative():
    """
    测试OpenCV作为替代方案
    """
    print("\n=== 测试OpenCV人脸检测替代方案 ===")
    try:
        import cv2
        print(f"✅ OpenCV版本: {cv2.__version__}")
        
        # 尝试加载Haar级联分类器
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            print("❌ Haar级联分类器加载失败")
        else:
            print("✅ Haar级联分类器加载成功")
            return face_cascade
    except ImportError:
        print("❌ OpenCV未安装")
    except Exception as e:
        print(f"❌ OpenCV测试失败: {e}")
    
    return None

if __name__ == "__main__":
    # 尝试修复MediaPipe
    detector = fix_mediapipe_path()
    
    if detector is None:
        # 如果MediaPipe修复失败，测试OpenCV替代方案
        opencv_detector = test_opencv_alternative()
        if opencv_detector is not None:
            print("\n✅ 可以使用OpenCV作为人脸检测的替代方案")
        else:
            print("\n❌ MediaPipe和OpenCV都无法使用")
    else:
        print("\n✅ MediaPipe修复成功！")