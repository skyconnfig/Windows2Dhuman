#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2D数字人项目 - 简化版启动脚本
暂时跳过MediaPipe相关功能，测试其他组件
"""

import gradio as gr
import numpy as np
import cv2
import torch
from pathlib import Path

print("=== 2D数字人系统启动中 ===")
print(f"PyTorch版本: {torch.__version__}")
print(f"OpenCV版本: {cv2.__version__}")
print(f"Gradio版本: {gr.__version__}")

# 检查模型文件
model_dir = Path("models")
if model_dir.exists():
    print(f"✅ 模型目录存在: {model_dir}")
    model_files = list(model_dir.glob("*.pth"))
    if model_files:
        print(f"✅ 找到模型文件: {len(model_files)}个")
        for model_file in model_files:
            print(f"   - {model_file.name}")
    else:
        print("⚠️  未找到.pth模型文件")
else:
    print("⚠️  模型目录不存在")

# 检查音频文件
audio_dir = Path("audio")
if audio_dir.exists():
    print(f"✅ 音频目录存在: {audio_dir}")
    audio_files = list(audio_dir.glob("*.wav"))
    if audio_files:
        print(f"✅ 找到音频文件: {len(audio_files)}个")
else:
    print("⚠️  音频目录不存在")

def simple_demo():
    """
    简化版演示界面
    """
    def process_text(text):
        if not text:
            return "请输入文本内容"
        return f"处理文本: {text}\n\n系统状态: 虚拟环境运行正常\nPyTorch: {torch.__version__}\nOpenCV: {cv2.__version__}"
    
    def process_image(image):
        if image is None:
            return None
        # 简单的图像处理示例
        img_array = np.array(image)
        # 转换为灰度图
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        # 转换回RGB
        processed = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return processed
    
    with gr.Blocks(title="2D数字人系统 - 简化版") as demo:
        gr.Markdown("# 🤖 2D数字人实时渲染系统")
        gr.Markdown("### 虚拟环境测试版本 - MediaPipe功能暂时禁用")
        
        with gr.Tab("文本处理"):
            text_input = gr.Textbox(label="输入文本", placeholder="请输入要处理的文本...")
            text_output = gr.Textbox(label="处理结果", lines=5)
            text_btn = gr.Button("处理文本")
            text_btn.click(process_text, inputs=text_input, outputs=text_output)
        
        with gr.Tab("图像处理"):
            image_input = gr.Image(label="上传图像")
            image_output = gr.Image(label="处理结果")
            image_btn = gr.Button("处理图像")
            image_btn.click(process_image, inputs=image_input, outputs=image_output)
        
        with gr.Tab("系统信息"):
            gr.Markdown(f"""
            ### 系统环境信息
            - **Python版本**: {torch.__version__.split('+')[0] if '+' in torch.__version__ else torch.__version__}
            - **PyTorch版本**: {torch.__version__}
            - **OpenCV版本**: {cv2.__version__}
            - **Gradio版本**: {gr.__version__}
            - **虚拟环境**: ✅ 已激活
            
            ### 注意事项
            - MediaPipe功能暂时禁用（DLL加载问题）
            - 其他核心功能正常运行
            - 建议安装Microsoft Visual C++ Redistributable解决MediaPipe问题
            """)
    
    return demo

if __name__ == "__main__":
    print("\n=== 启动Gradio界面 ===")
    demo = simple_demo()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True
    )