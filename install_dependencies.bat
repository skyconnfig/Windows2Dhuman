@echo off
echo ========================================
echo 数字人唇形优化项目依赖安装脚本
echo ========================================
echo.

echo 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo 错误：未找到Python，请先安装Python 3.9-3.11
    pause
    exit /b 1
)

echo.
echo 升级pip到最新版本...
python -m pip install --upgrade pip

echo.
echo 安装基础依赖包...
pip install -r requirements.txt

echo.
echo 安装项目特定依赖包...
echo 安装音频数据增强库...
pip install audiomentations>=0.30.0

echo 安装HTML生成库...
pip install dominate>=2.6.0

echo 安装音频处理库...
pip install librosa>=0.9.0

echo 安装模型分析工具...
pip install thop>=0.1.1

echo 安装系统监控库...
pip install psutil>=5.9.0

echo 安装图像IO库...
pip install imageio>=2.25.0

echo 安装网页解析库...
pip install beautifulsoup4>=4.11.0

echo.
echo ========================================
echo 依赖安装完成！
echo ========================================
echo.
echo 验证关键包安装状态...
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import cv2; print(f'OpenCV版本: {cv2.__version__}')"
python -c "import mediapipe; print(f'MediaPipe版本: {mediapipe.__version__}')"
python -c "import gradio; print(f'Gradio版本: {gradio.__version__}')"
python -c "import numpy; print(f'NumPy版本: {numpy.__version__}')"

echo.
echo 如果上述验证都成功，说明环境配置完成！
echo 现在可以运行: python app.py
echo.
pause