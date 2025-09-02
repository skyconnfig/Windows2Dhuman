#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查MediaPipe face_detection模型文件
"""

import os
import mediapipe as mp

def check_face_detection_models():
    """
    检查face_detection模型文件的详细信息
    """
    print("=== 检查face_detection模型文件 ===")
    
    # 获取MediaPipe目录
    mp_dir = os.path.dirname(mp.__file__)
    face_detection_dir = os.path.join(mp_dir, 'modules', 'face_detection')
    
    print(f"face_detection目录: {face_detection_dir}")
    print(f"目录存在: {os.path.exists(face_detection_dir)}")
    
    if os.path.exists(face_detection_dir):
        try:
            content = os.listdir(face_detection_dir)
            print(f"\n目录内容数量: {len(content)}")
            print("所有文件和目录:")
            for i, item in enumerate(content):
                item_path = os.path.join(face_detection_dir, item)
                is_dir = os.path.isdir(item_path)
                size = os.path.getsize(item_path) if not is_dir else 'DIR'
                print(f"  {i+1}. {item} ({'目录' if is_dir else f'{size} bytes'})")
                
                # 如果是.tflite文件，显示更多信息
                if item.endswith('.tflite'):
                    print(f"     -> 模型文件: {item}")
        except Exception as e:
            print(f"无法读取face_detection目录: {e}")
    
    # 尝试手动创建FaceDetection对象并捕获详细错误
    print("\n=== 尝试创建FaceDetection对象 ===")
    try:
        # 设置详细日志
        os.environ['GLOG_logtostderr'] = '1'
        os.environ['GLOG_v'] = '2'
        
        from mediapipe.python.solutions import face_detection as fd
        print("✅ face_detection模块导入成功")
        
        # 尝试创建对象
        print("尝试创建FaceDetection对象...")
        face_detection = fd.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        print("✅ FaceDetection对象创建成功！")
        
    except Exception as e:
        print(f"❌ 创建FaceDetection对象失败: {e}")
        print(f"错误类型: {type(e).__name__}")
        
        # 尝试获取更详细的错误信息
        import traceback
        print("\n详细错误堆栈:")
        traceback.print_exc()
    
    # 检查可能的模型文件路径
    print("\n=== 检查可能的模型文件路径 ===")
    possible_model_files = [
        'face_detection_short_range.tflite',
        'face_detection_full_range.tflite',
        'face_detection_front.tflite',
        'face_detection_back.tflite'
    ]
    
    for model_file in possible_model_files:
        model_path = os.path.join(face_detection_dir, model_file)
        exists = os.path.exists(model_path)
        print(f"{model_file}: {'存在' if exists else '不存在'}")
        if exists:
            size = os.path.getsize(model_path)
            print(f"  文件大小: {size} bytes ({size/1024/1024:.2f} MB)")

if __name__ == "__main__":
    check_face_detection_models()