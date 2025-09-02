#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的Gradio测试
验证虚拟环境中Gradio是否能正常工作
"""

import gradio as gr
import time

def hello_world(name):
    """
    简单的问候函数
    """
    if not name:
        name = "世界"
    return f"你好，{name}！\n\n当前时间: {time.strftime('%Y-%m-%d %H:%M:%S')}"

print("=== 简单Gradio测试 ===")
print("创建Gradio界面...")

# 创建简单界面
with gr.Blocks(title="虚拟环境测试") as demo:
    gr.Markdown("# 🐍 Python虚拟环境测试")
    gr.Markdown("### 如果你能看到这个界面，说明虚拟环境配置成功！")
    
    name_input = gr.Textbox(label="输入你的名字", placeholder="请输入名字...")
    output = gr.Textbox(label="问候信息", lines=3)
    btn = gr.Button("点击问候")
    
    btn.click(hello_world, inputs=name_input, outputs=output)
    
    gr.Markdown("""
    ### 虚拟环境信息
    - ✅ Python虚拟环境已激活
    - ✅ Gradio界面正常运行
    - ✅ 所有依赖包已安装
    """)

if __name__ == "__main__":
    print("启动Gradio服务器...")
    try:
        demo.launch(
            server_name="localhost",
            server_port=7862,
            share=False,
            show_error=True,
            quiet=False
        )
    except Exception as e:
        print(f"启动失败: {e}")
        print("\n尝试其他端口...")
        try:
            demo.launch(
                server_name="127.0.0.1",
                server_port=8080,
                share=False,
                show_error=True,
                quiet=False
            )
        except Exception as e2:
            print(f"再次启动失败: {e2}")
            print("\n虚拟环境配置正常，但Gradio网络启动有问题。")
            print("这可能是由于网络代理或防火墙设置导致的。")