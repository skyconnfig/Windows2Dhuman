#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高质量数据集准备脚本
为160px高分辨率训练准备和验证数据集质量
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from PIL import Image
import json
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataQualityAnalyzer:
    """数据质量分析器"""
    
    def __init__(self, target_resolution: int = 160):
        self.target_resolution = target_resolution
        self.quality_metrics = {
            'sharpness_threshold': 100,      # 拉普拉斯方差阈值
            'contrast_threshold': 30,        # 对比度阈值
            'brightness_range': (50, 200),   # 亮度范围
            'min_face_size': 64,            # 最小人脸尺寸
            'blur_threshold': 50,           # 模糊度阈值
        }
    
    def calculate_sharpness(self, image: np.ndarray) -> float:
        """计算图像清晰度（拉普拉斯方差）"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def calculate_contrast(self, image: np.ndarray) -> float:
        """计算图像对比度"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return gray.std()
    
    def calculate_brightness(self, image: np.ndarray) -> float:
        """计算图像亮度"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return gray.mean()
    
    def detect_blur(self, image: np.ndarray) -> float:
        """检测图像模糊程度"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def detect_face_landmarks(self, image: np.ndarray) -> Optional[Dict]:
        """检测人脸关键点（简化版本）"""
        try:
            # 使用OpenCV的人脸检测器
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
            
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # 返回最大的人脸
                largest_face = max(faces, key=lambda x: x[2] * x[3])
                x, y, w, h = largest_face
                
                # 估算嘴部区域（人脸下半部分）
                mouth_x = x + w // 4
                mouth_y = y + int(h * 0.6)
                mouth_w = w // 2
                mouth_h = int(h * 0.3)
                
                return {
                    'face_bbox': (x, y, w, h),
                    'mouth_bbox': (mouth_x, mouth_y, mouth_w, mouth_h),
                    'face_size': w * h
                }
        except Exception as e:
            logger.warning(f"人脸检测失败: {e}")
        
        return None
    
    def analyze_image_quality(self, image_path: str) -> Dict:
        """分析单张图像质量"""
        try:
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return {'error': '无法读取图像'}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image_rgb.shape[:2]
            
            # 基础质量指标
            sharpness = self.calculate_sharpness(image_rgb)
            contrast = self.calculate_contrast(image_rgb)
            brightness = self.calculate_brightness(image_rgb)
            blur_score = self.detect_blur(image_rgb)
            
            # 人脸检测
            face_info = self.detect_face_landmarks(image_rgb)
            
            # 质量评估
            quality_score = 0
            issues = []
            
            # 清晰度检查
            if sharpness >= self.quality_metrics['sharpness_threshold']:
                quality_score += 25
            else:
                issues.append(f"清晰度不足: {sharpness:.1f} < {self.quality_metrics['sharpness_threshold']}")
            
            # 对比度检查
            if contrast >= self.quality_metrics['contrast_threshold']:
                quality_score += 25
            else:
                issues.append(f"对比度不足: {contrast:.1f} < {self.quality_metrics['contrast_threshold']}")
            
            # 亮度检查
            if self.quality_metrics['brightness_range'][0] <= brightness <= self.quality_metrics['brightness_range'][1]:
                quality_score += 25
            else:
                issues.append(f"亮度异常: {brightness:.1f} 不在 {self.quality_metrics['brightness_range']} 范围内")
            
            # 人脸检查
            if face_info and face_info['face_size'] >= self.quality_metrics['min_face_size'] ** 2:
                quality_score += 25
            else:
                issues.append("人脸尺寸不足或未检测到人脸")
            
            # 分辨率检查
            resolution_adequate = min(w, h) >= self.target_resolution
            
            return {
                'image_path': image_path,
                'resolution': (w, h),
                'resolution_adequate': resolution_adequate,
                'sharpness': sharpness,
                'contrast': contrast,
                'brightness': brightness,
                'blur_score': blur_score,
                'face_info': face_info,
                'quality_score': quality_score,
                'issues': issues,
                'recommended': quality_score >= 75 and resolution_adequate
            }
            
        except Exception as e:
            return {'error': f'分析失败: {str(e)}'}

class DatasetPreprocessor:
    """数据集预处理器"""
    
    def __init__(self, target_resolution: int = 160, output_dir: str = None):
        self.target_resolution = target_resolution
        self.output_dir = output_dir or f"processed_dataset_{target_resolution}px"
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'images'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'metadata'), exist_ok=True)
    
    def resize_and_crop_image(self, image: np.ndarray, target_size: int) -> np.ndarray:
        """智能调整图像尺寸和裁剪"""
        h, w = image.shape[:2]
        
        # 计算缩放比例，保持长宽比
        scale = max(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        # 缩放图像
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # 中心裁剪到目标尺寸
        start_x = (new_w - target_size) // 2
        start_y = (new_h - target_size) // 2
        cropped = resized[start_y:start_y + target_size, start_x:start_x + target_size]
        
        return cropped
    
    def enhance_image_quality(self, image: np.ndarray) -> np.ndarray:
        """增强图像质量"""
        # 转换到LAB色彩空间进行处理
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        
        # 应用CLAHE增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # 合并通道并转换回RGB
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        
        # 轻微锐化
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 混合原图和锐化图
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
    
    def process_image(self, image_path: str, quality_info: Dict) -> Optional[str]:
        """处理单张图像"""
        try:
            if not quality_info.get('recommended', False):
                return None
            
            # 读取图像
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 调整尺寸
            processed = self.resize_and_crop_image(image_rgb, self.target_resolution)
            
            # 质量增强
            if quality_info['quality_score'] < 90:  # 只对质量不够高的图像进行增强
                processed = self.enhance_image_quality(processed)
            
            # 保存处理后的图像
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(self.output_dir, 'images', f"{name}_processed{ext}")
            
            processed_bgr = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, processed_bgr)
            
            return output_path
            
        except Exception as e:
            logger.error(f"处理图像失败 {image_path}: {e}")
            return None

class HighQualityDatasetPreparer:
    """高质量数据集准备器"""
    
    def __init__(self, target_resolution: int = 160, num_workers: int = 4):
        self.target_resolution = target_resolution
        self.num_workers = num_workers
        self.analyzer = DataQualityAnalyzer(target_resolution)
        self.preprocessor = DatasetPreprocessor(target_resolution)
        
    def scan_directory(self, directory: str) -> List[str]:
        """扫描目录中的图像文件"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if Path(file).suffix.lower() in image_extensions:
                    image_files.append(os.path.join(root, file))
        
        return image_files
    
    def analyze_dataset_quality(self, image_paths: List[str]) -> Dict:
        """分析数据集质量"""
        logger.info(f"开始分析 {len(image_paths)} 张图像的质量...")
        
        quality_results = []
        
        # 使用多线程处理
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_path = {executor.submit(self.analyzer.analyze_image_quality, path): path 
                            for path in image_paths}
            
            for future in tqdm(as_completed(future_to_path), total=len(image_paths), desc="质量分析"):
                result = future.result()
                if 'error' not in result:
                    quality_results.append(result)
        
        # 统计分析
        total_images = len(quality_results)
        recommended_images = sum(1 for r in quality_results if r.get('recommended', False))
        
        quality_scores = [r['quality_score'] for r in quality_results]
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        resolution_adequate = sum(1 for r in quality_results if r.get('resolution_adequate', False))
        
        # 问题统计
        all_issues = []
        for result in quality_results:
            all_issues.extend(result.get('issues', []))
        
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue.split(':')[0]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        summary = {
            'total_images': total_images,
            'recommended_images': recommended_images,
            'recommendation_rate': recommended_images / total_images if total_images > 0 else 0,
            'average_quality_score': avg_quality,
            'resolution_adequate_count': resolution_adequate,
            'resolution_adequate_rate': resolution_adequate / total_images if total_images > 0 else 0,
            'common_issues': issue_counts,
            'detailed_results': quality_results
        }
        
        return summary
    
    def prepare_dataset(self, input_directory: str, output_directory: str = None) -> Dict:
        """准备高质量数据集"""
        if output_directory:
            self.preprocessor.output_dir = output_directory
            os.makedirs(output_directory, exist_ok=True)
            os.makedirs(os.path.join(output_directory, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_directory, 'metadata'), exist_ok=True)
        
        # 扫描图像文件
        logger.info(f"扫描目录: {input_directory}")
        image_paths = self.scan_directory(input_directory)
        logger.info(f"找到 {len(image_paths)} 张图像")
        
        if not image_paths:
            return {'error': '未找到图像文件'}
        
        # 分析质量
        quality_summary = self.analyze_dataset_quality(image_paths)
        
        # 处理推荐的图像
        logger.info("开始处理推荐的高质量图像...")
        processed_paths = []
        
        recommended_results = [r for r in quality_summary['detailed_results'] if r.get('recommended', False)]
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_result = {executor.submit(self.preprocessor.process_image, r['image_path'], r): r 
                              for r in recommended_results}
            
            for future in tqdm(as_completed(future_to_result), total=len(recommended_results), desc="图像处理"):
                processed_path = future.result()
                if processed_path:
                    processed_paths.append(processed_path)
        
        # 保存元数据
        metadata = {
            'target_resolution': self.target_resolution,
            'input_directory': input_directory,
            'output_directory': self.preprocessor.output_dir,
            'processing_summary': quality_summary,
            'processed_images': len(processed_paths),
            'processed_paths': processed_paths
        }
        
        metadata_path = os.path.join(self.preprocessor.output_dir, 'metadata', 'dataset_info.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # 生成质量报告
        self.generate_quality_report(quality_summary, self.preprocessor.output_dir)
        
        logger.info(f"数据集准备完成！")
        logger.info(f"输出目录: {self.preprocessor.output_dir}")
        logger.info(f"处理了 {len(processed_paths)} 张高质量图像")
        
        return metadata
    
    def generate_quality_report(self, quality_summary: Dict, output_dir: str):
        """生成质量报告"""
        report_path = os.path.join(output_dir, 'metadata', 'quality_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("数据集质量分析报告\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"目标分辨率: {self.target_resolution}px\n")
            f.write(f"总图像数量: {quality_summary['total_images']}\n")
            f.write(f"推荐图像数量: {quality_summary['recommended_images']}\n")
            f.write(f"推荐率: {quality_summary['recommendation_rate']:.2%}\n")
            f.write(f"平均质量分数: {quality_summary['average_quality_score']:.1f}/100\n")
            f.write(f"分辨率达标数量: {quality_summary['resolution_adequate_count']}\n")
            f.write(f"分辨率达标率: {quality_summary['resolution_adequate_rate']:.2%}\n\n")
            
            f.write("常见问题统计:\n")
            for issue, count in quality_summary['common_issues'].items():
                f.write(f"- {issue}: {count} 次\n")
            
            f.write("\n建议:\n")
            if quality_summary['recommendation_rate'] < 0.5:
                f.write("- 数据集质量较低，建议收集更高质量的图像\n")
            if quality_summary['resolution_adequate_rate'] < 0.8:
                f.write("- 大部分图像分辨率不足，建议使用更高分辨率的源图像\n")
            if '清晰度不足' in quality_summary['common_issues']:
                f.write("- 存在较多模糊图像，建议提高拍摄质量\n")
            if '对比度不足' in quality_summary['common_issues']:
                f.write("- 存在较多低对比度图像，建议改善拍摄光照条件\n")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='高质量数据集准备工具')
    parser.add_argument('--input_dir', type=str, required=True, help='输入图像目录')
    parser.add_argument('--output_dir', type=str, help='输出目录')
    parser.add_argument('--resolution', type=int, default=160, help='目标分辨率')
    parser.add_argument('--workers', type=int, default=4, help='并行处理线程数')
    parser.add_argument('--analyze_only', action='store_true', help='仅分析质量，不处理图像')
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    # 初始化准备器
    preparer = HighQualityDatasetPreparer(
        target_resolution=args.resolution,
        num_workers=args.workers
    )
    
    if args.analyze_only:
        # 仅分析质量
        image_paths = preparer.scan_directory(args.input_dir)
        quality_summary = preparer.analyze_dataset_quality(image_paths)
        
        print("\n质量分析结果:")
        print(f"总图像数量: {quality_summary['total_images']}")
        print(f"推荐图像数量: {quality_summary['recommended_images']}")
        print(f"推荐率: {quality_summary['recommendation_rate']:.2%}")
        print(f"平均质量分数: {quality_summary['average_quality_score']:.1f}/100")
        
    else:
        # 完整处理
        result = preparer.prepare_dataset(args.input_dir, args.output_dir)
        
        if 'error' in result:
            logger.error(f"处理失败: {result['error']}")
        else:
            print("\n处理完成！")
            print(f"输出目录: {result['output_directory']}")
            print(f"处理图像数量: {result['processed_images']}")

if __name__ == "__main__":
    # 如果没有命令行参数，运行测试
    if len(sys.argv) == 1:
        print("高质量数据集准备工具")
        print("=" * 50)
        print("\n使用方法:")
        print("python prepare_high_quality_dataset.py --input_dir <输入目录> [选项]")
        print("\n选项:")
        print("  --output_dir <输出目录>    指定输出目录")
        print("  --resolution <分辨率>     目标分辨率 (默认: 160)")
        print("  --workers <线程数>        并行处理线程数 (默认: 4)")
        print("  --analyze_only           仅分析质量，不处理图像")
        print("\n示例:")
        print("  python prepare_high_quality_dataset.py --input_dir ./raw_images --output_dir ./processed_160px --resolution 160")
        print("  python prepare_high_quality_dataset.py --input_dir ./raw_images --analyze_only")
        
        # 运行简单测试
        print("\n运行组件测试...")
        analyzer = DataQualityAnalyzer(160)
        print("✓ 数据质量分析器初始化成功")
        
        preprocessor = DatasetPreprocessor(160)
        print("✓ 数据预处理器初始化成功")
        
        preparer = HighQualityDatasetPreparer(160)
        print("✓ 数据集准备器初始化成功")
        
        print("\n所有组件测试通过！")
    else:
        main()