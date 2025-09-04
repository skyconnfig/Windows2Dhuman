#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数字人唇形优化项目依赖安装脚本
支持Windows、Linux、macOS跨平台安装
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description=""):
    """
    执行命令并处理错误
    """
    if description:
        print(f"\n{description}...")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print(f"✓ 成功: {description}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ 失败: {description}")
        print(f"错误信息: {e.stderr}")
        return False

def check_python_version():
    """
    检查Python版本
    """
    version = sys.version_info
    print(f"当前Python版本: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 9 or version.minor > 11:
        print("⚠️  警告: 推荐使用Python 3.9-3.11版本以获得最佳兼容性")
        print("   特别是MediaPipe可能在其他版本上出现问题")
        
        response = input("是否继续安装? (y/N): ")
        if response.lower() != 'y':
            return False
    
    return True

def install_basic_dependencies():
    """
    安装基础依赖
    """
    print("\n" + "="*50)
    print("安装基础依赖包")
    print("="*50)
    
    # 升级pip
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "升级pip"):
        return False
    
    # 安装requirements.txt中的依赖
    requirements_file = Path("requirements.txt")
    if requirements_file.exists():
        if not run_command(f"{sys.executable} -m pip install -r requirements.txt", 
                          "安装requirements.txt中的依赖"):
            return False
    else:
        print("⚠️  未找到requirements.txt文件")
    
    return True

def install_additional_dependencies():
    """
    安装项目特定的额外依赖
    """
    print("\n" + "="*50)
    print("安装项目特定依赖包")
    print("="*50)
    
    additional_packages = [
        ("audiomentations>=0.30.0", "音频数据增强库"),
        ("dominate>=2.6.0", "HTML生成库"),
        ("librosa>=0.9.0", "音频处理库"),
        ("thop>=0.1.1", "模型分析工具"),
        ("psutil>=5.9.0", "系统监控库"),
        ("imageio>=2.25.0", "图像IO库"),
        ("beautifulsoup4>=4.11.0", "网页解析库"),
    ]
    
    success_count = 0
    for package, description in additional_packages:
        if run_command(f"{sys.executable} -m pip install {package}", f"安装{description}"):
            success_count += 1
    
    print(f"\n额外依赖安装完成: {success_count}/{len(additional_packages)} 个包成功安装")
    return success_count == len(additional_packages)

def verify_installation():
    """
    验证关键包的安装状态
    """
    print("\n" + "="*50)
    print("验证关键包安装状态")
    print("="*50)
    
    test_imports = [
        ("torch", "PyTorch"),
        ("cv2", "OpenCV"),
        ("mediapipe", "MediaPipe"),
        ("gradio", "Gradio"),
        ("numpy", "NumPy"),
        ("librosa", "Librosa"),
        ("audiomentations", "AudioMentations"),
    ]
    
    success_count = 0
    for module_name, display_name in test_imports:
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', '未知版本')
            print(f"✓ {display_name}: {version}")
            success_count += 1
        except ImportError:
            print(f"✗ {display_name}: 未安装或导入失败")
    
    print(f"\n验证结果: {success_count}/{len(test_imports)} 个关键包可用")
    return success_count >= len(test_imports) - 2  # 允许2个包失败

def main():
    """
    主安装流程
    """
    print("数字人唇形优化项目依赖安装脚本")
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"Python路径: {sys.executable}")
    
    # 检查Python版本
    if not check_python_version():
        print("安装已取消")
        return False
    
    # 安装基础依赖
    if not install_basic_dependencies():
        print("基础依赖安装失败，请检查网络连接和权限")
        return False
    
    # 安装额外依赖
    install_additional_dependencies()
    
    # 验证安装
    if verify_installation():
        print("\n" + "="*50)
        print("🎉 依赖安装完成！")
        print("="*50)
        print("\n现在可以运行以下命令启动项目:")
        print("python app.py")
        print("\n如果遇到问题，请检查:")
        print("1. Python版本是否为3.9-3.11")
        print("2. 网络连接是否正常")
        print("3. 是否有足够的磁盘空间")
        print("4. Windows用户是否安装了Visual C++ Redistributable")
        return True
    else:
        print("\n⚠️  部分依赖可能安装失败，但核心功能应该可用")
        print("如果遇到运行时错误，请手动安装缺失的包")
        return False

if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n安装已被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n安装过程中出现未预期的错误: {e}")
        sys.exit(1)