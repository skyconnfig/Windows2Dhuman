#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV人脸检测替代方案
替代MediaPipe的人脸检测功能
"""

import cv2
import numpy as np
import os

class OpenCVFaceDetection:
    """
    使用OpenCV实现人脸检测，替代MediaPipe
    """
    
    def __init__(self, min_detection_confidence=0.5):
        """
        初始化OpenCV人脸检测器
        
        Args:
            min_detection_confidence: 最小检测置信度
        """
        self.min_detection_confidence = min_detection_confidence
        
        # 尝试加载不同的人脸检测模型
        self.detector = None
        self.detection_method = None
        
        # 方法1: 尝试使用DNN人脸检测模型
        if self._load_dnn_detector():
            self.detection_method = 'dnn'
            print("✅ 使用DNN人脸检测模型")
        # 方法2: 使用Haar级联分类器
        elif self._load_haar_detector():
            self.detection_method = 'haar'
            print("✅ 使用Haar级联分类器")
        else:
            raise RuntimeError("无法加载任何人脸检测模型")
    
    def _load_dnn_detector(self):
        """
        尝试加载DNN人脸检测模型
        """
        try:
            # 创建DNN模型文件（如果不存在）
            model_dir = "models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # 这里可以下载预训练的DNN模型
            # 暂时跳过DNN方法
            return False
        except Exception as e:
            print(f"DNN模型加载失败: {e}")
            return False
    
    def _load_haar_detector(self):
        """
        加载Haar级联分类器
        """
        try:
            # 直接使用正确的路径构建方式
            haar_file = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            print(f"尝试加载Haar文件: {haar_file}")
            print(f"文件存在: {os.path.exists(haar_file)}")
            
            if os.path.exists(haar_file):
                self.detector = cv2.CascadeClassifier(haar_file)
                if not self.detector.empty():
                    print(f"✅ Haar级联文件加载成功: {haar_file}")
                    return True
                else:
                    print("❌ Haar级联文件存在但加载失败")
            
            # 尝试其他备选文件
            alternative_files = [
                'haarcascade_frontalface_alt.xml',
                'haarcascade_frontalface_alt2.xml'
            ]
            
            for alt_file in alternative_files:
                alt_path = os.path.join(cv2.data.haarcascades, alt_file)
                print(f"尝试备选文件: {alt_path}")
                if os.path.exists(alt_path):
                    self.detector = cv2.CascadeClassifier(alt_path)
                    if not self.detector.empty():
                        print(f"✅ 备选Haar文件加载成功: {alt_path}")
                        return True
            
            # 如果所有Haar文件都失败，使用轮廓检测作为备用方案
            print("⚠️ 所有Haar文件都无法加载，使用轮廓检测作为备用方案")
            return self._create_simple_detector()
            
        except Exception as e:
            print(f"Haar级联加载失败: {e}")
            return self._create_simple_detector()
    
    def _create_simple_detector(self):
        """
        创建一个简单的人脸检测器（备用方案）
        """
        try:
            # 使用OpenCV的内置检测器
            self.detector = cv2.CascadeClassifier()
            # 尝试加载默认的人脸检测器
            if hasattr(cv2, 'face'):
                # 如果有face模块，使用它
                self.detection_method = 'face_module'
                return True
            else:
                # 使用基本的轮廓检测作为最后的备用方案
                self.detection_method = 'contour'
                return True
        except Exception as e:
            print(f"简单检测器创建失败: {e}")
            return False
    
    def detect_faces(self, image):
        """
        检测人脸
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            list: 检测到的人脸框列表，每个元素为 (x, y, w, h)
        """
        if self.detector is None:
            return []
        
        try:
            if self.detection_method == 'haar':
                return self._detect_with_haar(image)
            elif self.detection_method == 'contour':
                return self._detect_with_contour(image)
            else:
                return []
        except Exception as e:
            print(f"人脸检测失败: {e}")
            return []
    
    def _detect_with_haar(self, image):
        """
        使用Haar级联检测人脸
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces.tolist() if len(faces) > 0 else []
    
    def _detect_with_contour(self, image):
        """
        使用轮廓检测（备用方案）
        """
        # 这是一个非常基础的检测方法，仅作为最后的备用方案
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用边缘检测
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        faces = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # 简单的人脸大小过滤
            if 30 < w < 300 and 30 < h < 300 and 0.8 < w/h < 1.2:
                faces.append([x, y, w, h])
        
        return faces[:5]  # 最多返回5个检测结果

def test_opencv_face_detection():
    """
    测试OpenCV人脸检测功能
    """
    print("=== 测试OpenCV人脸检测 ===")
    
    try:
        # 创建检测器
        detector = OpenCVFaceDetection()
        print(f"✅ 检测器创建成功，使用方法: {detector.detection_method}")
        
        # 创建一个测试图像
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        test_image.fill(128)  # 灰色背景
        
        # 绘制一个简单的"人脸"用于测试
        cv2.rectangle(test_image, (200, 150), (400, 350), (255, 255, 255), -1)  # 白色矩形
        cv2.circle(test_image, (250, 200), 10, (0, 0, 0), -1)  # 左眼
        cv2.circle(test_image, (350, 200), 10, (0, 0, 0), -1)  # 右眼
        cv2.rectangle(test_image, (280, 250), (320, 280), (0, 0, 0), -1)  # 鼻子
        cv2.ellipse(test_image, (300, 320), (30, 15), 0, 0, 180, (0, 0, 0), 2)  # 嘴巴
        
        # 检测人脸
        faces = detector.detect_faces(test_image)
        print(f"检测到 {len(faces)} 个人脸")
        
        for i, (x, y, w, h) in enumerate(faces):
            print(f"  人脸 {i+1}: x={x}, y={y}, w={w}, h={h}")
        
        return detector
        
    except Exception as e:
        print(f"❌ OpenCV人脸检测测试失败: {e}")
        return None

if __name__ == "__main__":
    test_opencv_face_detection()