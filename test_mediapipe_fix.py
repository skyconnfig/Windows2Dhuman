#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MediaPipe人脸检测功能详细测试脚本
用于定位MediaPipe初始化问题的具体原因
"""

import os
import cv2
import mediapipe as mp
import numpy as np

def test_mediapipe_step_by_step():
    """
    逐步测试MediaPipe各个组件
    return: bool - 测试是否成功
    """
    try:
        print("=== 逐步测试MediaPipe组件 ===")
        print(f"MediaPipe版本: {mp.__version__}")
        
        # 步骤1: 导入solutions
        print("\n步骤1: 导入solutions模块")
        solutions = mp.solutions
        print("✅ solutions模块导入成功")
        
        # 步骤2: 获取face_detection
        print("\n步骤2: 获取face_detection模块")
        face_detection = solutions.face_detection
        print("✅ face_detection模块获取成功")
        
        # 步骤3: 获取drawing_utils
        print("\n步骤3: 获取drawing_utils模块")
        drawing_utils = solutions.drawing_utils
        print("✅ drawing_utils模块获取成功")
        
        # 步骤4: 设置环境变量
        print("\n步骤4: 设置环境变量")
        os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'
        print("✅ 环境变量设置完成")
        
        # 步骤5: 尝试创建FaceDetection对象（最小配置）
        print("\n步骤5: 创建FaceDetection对象（最小配置）")
        try:
            with face_detection.FaceDetection() as face_detector:
                print("✅ FaceDetection对象创建成功（默认参数）")
                return True
        except Exception as e:
            print(f"❌ FaceDetection对象创建失败: {str(e)}")
            
            # 步骤6: 尝试不同的参数配置
            print("\n步骤6: 尝试不同的参数配置")
            try:
                with face_detection.FaceDetection(
                    model_selection=0,
                    min_detection_confidence=0.5
                ) as face_detector:
                    print("✅ FaceDetection对象创建成功（指定参数）")
                    return True
            except Exception as e2:
                print(f"❌ FaceDetection对象创建失败（指定参数）: {str(e2)}")
                
                # 步骤7: 尝试静态模式
                print("\n步骤7: 尝试静态模式")
                try:
                    detector = face_detection.FaceDetection(
                        model_selection=1,
                        min_detection_confidence=0.7
                    )
                    print("✅ FaceDetection对象创建成功（静态模式）")
                    detector.close()
                    return True
                except Exception as e3:
                    print(f"❌ FaceDetection对象创建失败（静态模式）: {str(e3)}")
                    return False
                    
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {str(e)}")
        return False

def check_mediapipe_installation():
    """
    检查MediaPipe安装和依赖
    """
    print("=== 检查MediaPipe安装状态 ===")
    
    try:
        # 检查MediaPipe包路径
        import mediapipe
        mp_path = mediapipe.__file__
        print(f"MediaPipe安装路径: {mp_path}")
        
        # 检查关键文件
        mp_dir = os.path.dirname(mp_path)
        print(f"MediaPipe目录: {mp_dir}")
        
        # 检查python子目录
        python_dir = os.path.join(mp_dir, 'python')
        if os.path.exists(python_dir):
            print(f"✅ python目录存在: {python_dir}")
        else:
            print(f"❌ python目录不存在: {python_dir}")
            
        # 检查solutions子目录
        solutions_dir = os.path.join(python_dir, 'solutions')
        if os.path.exists(solutions_dir):
            print(f"✅ solutions目录存在: {solutions_dir}")
        else:
            print(f"❌ solutions目录不存在: {solutions_dir}")
            
        # 检查face_detection.py文件
        face_detection_file = os.path.join(solutions_dir, 'face_detection.py')
        if os.path.exists(face_detection_file):
            print(f"✅ face_detection.py存在: {face_detection_file}")
        else:
            print(f"❌ face_detection.py不存在: {face_detection_file}")
            
    except Exception as e:
        print(f"❌ 检查安装状态时发生错误: {str(e)}")

if __name__ == "__main__":
    print("=== MediaPipe详细诊断测试 ===")
    
    # 检查安装状态
    check_mediapipe_installation()
    
    print("\n" + "="*50)
    
    # 逐步测试
    success = test_mediapipe_step_by_step()
    
    print("\n" + "="*50)
    print("=== 最终测试结果 ===")
    if success:
        print("🎉 MediaPipe功能测试成功！")
    else:
        print("⚠️  MediaPipe功能测试失败，需要进一步排查。")
        print("\n建议尝试:")
        print("1. 重新安装MediaPipe")
        print("2. 检查系统环境变量")
        print("3. 确认Python版本兼容性")
        print("4. 检查网络连接（模型文件下载）")