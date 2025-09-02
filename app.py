# coding: utf-8
import base64
import os.path
import shutil
import cv2
import gradio as gr
import subprocess
import uuid
from PIL import Image

import numpy as np
from data_preparation_mini import data_preparation_mini
from data_preparation_web import data_preparation_web
 


# 自定义 CSS 样式
css = """ 

#video-output video {
    max-width: 300px;
    max-height: 300px;
    min-height: 300px;
    display: block;
    margin: 0 auto;
}
 

.custom-box {
    border: 1px solid #4df0ff;
    border-radius: 8px;
    padding: 20px;
    margin: 20px 0;
    background-color: rgba(10, 25, 40, 0.9);
    box-shadow: 0 4px 20px rgba(0, 150, 255, 0.3);
    color: #e0f7ff;
    position: relative;
}

.custom-box::after {
    content: "";
    position: absolute;
    top: -2px;
    left: -2px;
    right: -2px;
    bottom: -2px;
    border: 2px solid #00d8ff;
    border-radius: 10px;
    z-index: -1;
    opacity: 0;
    transition: 0.3s;
}

.custom-box:hover::after {
    opacity: 1;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 0.8; }
    50% { opacity: 0.3; }
    100% { opacity: 0.8; }
}



/* 主标题样式 - 与.custom-box风格统一 */
.fixed-header {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    padding: 16px 0;
    text-align: center;
    background: linear-gradient(90deg, 
                rgba(10, 25, 40, 0.98) 0%, 
                rgba(0, 103, 155, 0.95) 50%, 
                rgba(10, 25, 40, 0.98) 100%);
    color: #e0f7ff;
    z-index: 1000;
    box-shadow: 0 4px 30px rgba(0, 150, 255, 0.15);
    border-bottom: 1px solid rgba(77, 240, 255, 0.3); 
}

/* 内容区补偿 (与.custom-box间距协调) */
.content-wrapper {
    margin-top: 82px;
}

/* 标题文字特效 (延续脉冲动画风格) */
.fixed-header h1 {
    margin: 0;
    font-size: 2.1em;
    letter-spacing: 1.5px;
    font-weight: 500;
    text-shadow: 0 0 10px rgba(77, 240, 255, 0.5);
    position: relative;
    display: inline-block;
    padding: 0 20px;
}

/* 标题下划线动画 */
.fixed-header h1::after {
    content: "";
    position: absolute;
    bottom: -5px;
    left: 20%;
    width: 60%;
    height: 2px;
    background: linear-gradient(90deg, 
              transparent 0%, 
              #4df0ff 20%, 
              #00d8ff 50%, 
              #4df0ff 80%, 
              transparent 100%);
    opacity: 0.7;
    animation: pulse 3s infinite;
}

/* 标题悬停效果 */
.fixed-header:hover {
    border-bottom-color: rgba(77, 240, 255, 0.6);
    box-shadow: 0 4px 40px rgba(0, 150, 255, 0.25);
}

.fixed-header:hover h1::after {
    animation: 
        pulse 1.5s infinite,
        shine 2s infinite;
}

@keyframes shine {
    0% { background-position: -100% 0; }
    100% { background-position: 100% 0; }
}



/* 隐藏Gradio默认的底部信息栏 */
footer {
    display: none !important;
}

/* 隐藏API链接 */
#api-link {
    display: none !important;
}

/* 隐藏设置按钮 */
#settings-button {
    display: none !important;
}

/* 隐藏Gradio水印 */
.gradio-container .watermark {
    display: none !important;
}

/* 缩略图行容器 */
.thumbnail-row { 
    display: flex !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important; 
    padding: 15px -20px 25px 5px !important;
    margin: 0 -5px !important;
    scrollbar-width: thin !important;
}

/* 缩略图卡片容器 */
.thumbnail-card {
    max-width: 300px;
    min-width: 150px !important;
    flex-shrink: 0 !important;
    border-radius: 8px !important;
    overflow: hidden !important;
    background: rgba(10, 25, 40, 0.3) !important;
    border: 1px solid rgba(77, 240, 255, 0.3) !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 10px rgba(0, 150, 255, 0.1) !important;
}

.thumbnail-card:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 6px 15px rgba(0, 150, 255, 0.2) !important;
    border-color: rgba(77, 240, 255, 0.6) !important;
}

/* 图片样式 */
.thumbnail-image {
    top:0px
    width: 100% !important; 
    height: 200px !important;
    object-fit: cover !important;
    border-bottom: 1px solid rgba(77, 240, 255, 0.2) !important;
}

/* 按钮样式 */
.thumbnail-btn {
    width: 100% !important;
    padding: 8px 0 !important;
    margin: 0 !important;
    border: none !important;
    border-radius: 0 0 7px 7px !important;
    background: linear-gradient(to right, 
                rgba(77, 240, 255, 0.15) 0%, 
                rgba(0, 216, 255, 0.25) 50%, 
                rgba(77, 240, 255, 0.15) 100%) !important;
    color: #e0f7ff !important;
    font-size: 12px !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
}

.thumbnail-btn:hover {
    background: linear-gradient(to right, 
                rgba(77, 240, 255, 0.25) 0%, 
                rgba(0, 216, 255, 0.35) 50%, 
                rgba(77, 240, 255, 0.25) 100%) !important;
    text-shadow: 0 0 5px rgba(77, 240, 255, 0.5) !important;
}

/* 自定义滚动条 */
.thumbnail-row::-webkit-scrollbar {
    height: 6px !important;
    background: rgba(10, 25, 40, 0.2) !important;
}

.thumbnail-row::-webkit-scrollbar-thumb {
    background: linear-gradient(90deg, 
                rgba(77, 240, 255, 0.6) 0%, 
                rgba(0, 216, 255, 0.8) 50%, 
                rgba(77, 240, 255, 0.6) 100%) !important;
    border-radius: 3px !important;
}
 

/* 按钮式单选 */
.btn-radio .gr-radio-group {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
}

.btn-radio .gr-radio-item {
    border: 1px solid #4df0ff;
    border-radius: 20px;
    padding: 8px 16px;
    background: transparent;
    cursor: pointer;
}

.btn-radio .gr-radio-item.selected {
    background: rgba(77, 240, 255, 0.3);
    border-color: #00d8ff;
}



/* 修改后的音频选择样式 */
.custom-box.audio-selection {
    border: 1px solid #4df0ff;
    background: rgba(10, 25, 40, 0.95) !important;
}

.custom-box.audio-selection .gr-radio-group {
    gap: 12px !important;
}

.custom-box.audio-selection .gr-radio-item {
    border: 1px solid rgba(77, 240, 255, 0.5) !important;
    background: rgba(15, 40, 60, 0.7) !important;
}

.custom-box.audio-selection .gr-radio-item.selected {
    background: linear-gradient(135deg, 
                rgba(77, 240, 255, 0.3) 0%, 
                rgba(0, 103, 155, 0.4) 100%) !important;
    box-shadow: 0 0 15px rgba(77, 240, 255, 0.3) !important;
}

/* 修正音频预览组件定位 */
.audio-selection .audio-preview {
    margin-top: 15px !important;
    border-top: 1px solid rgba(77, 240, 255, 0.2);
    padding-top: 15px !important;
}


/* 说明 */
.info-title {
    margin: 0px 0 !important;
    padding: 0 !important;
    font-size: 15px;
    border-bottom: 1px solid rgba(77, 240, 255, 0.3); /* 添加下划线保持视觉分隔 */
}





 
#result {
    min-height: 70px;
    max-height: none !important;
    overflow: hidden !important;
    text-align: center
}
 




#custom-textbox {
    min-height: 200px !important;
    background: rgba(10, 25, 40, 0.3) !important;
    border: 1px solid rgba(77, 240, 255, 0.3) !important;
    color: #e0f7ff !important;
    padding: 12px !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

#custom-textbox:focus {
    border-color: rgba(77, 240, 255, 0.6) !important;
    box-shadow: 0 0 10px rgba(77, 240, 255, 0.3) !important;
    background: rgba(10, 25, 40, 0.5) !important;
}

#custom-textbox::placeholder {
    color: rgba(224, 247, 255, 0.5) !important;
}
/* 防止文本编辑时页面跳动 */
#custom-textbox textarea {
    resize: vertical !important;
    overflow-y: auto !important;
    min-height: 200px !important;
    max-height: 500px !important;
}

/* 禁用Gradio默认的自动滚动行为 */
.gradio-container {
    scroll-behavior: auto !important;
}

.model-description {
 color: rgba(160, 224, 255, 0.8) !important;
    line-height: 0.8 !important;
    margin-left:  4px !important;
    margin-top:  5px !important;
    margin-bottom:  5px !important; 
      background: rgba(10, 25, 40, 0.3) !important;
        border-left: 3px solid rgba(77, 240, 255, 0.5) !important;
}
"""

 
def run_js_file(js_file_path):
    try:
        result = subprocess.run(
            ['node', js_file_path],
            check=True,
            text=True,
            capture_output=True
        )
        return result.stdout
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {e.stderr}")
        return None
    
def compress_webm(input_path, output_path, width=480, crf=40, bitrate="500k"):
    cmd = [
        "ffmpeg",
        "-i", input_path,
        "-vf", f"scale={width}:-2:flags=lanczos,split[s0][s1];[s0]reverse[r];[s1][r]concat",
        "-c:v", "libvpx-vp9",
        "-crf", str(crf),
        "-b:v", bitrate,
        "-row-mt", "1",
        "-quality", "good",
        "-cpu-used", "4",
        "-an", "-loop", "0",
        "-y", output_path
    ]
    subprocess.run(cmd)
    
    
# 假设你已经有了这两个函数
def data_preparation(video1,llmSystemInfo,voiceId,model_radio): 
    if video1 is None or not os.path.exists(video1):
        return (
        gr.Button("处理失败", variant="primary"),
        f"<h3 id='result'>请上传需要训练的形象，或者使用示例形象做参考！</h3>" 
    )

    

    pp=uuid.uuid4()
    # 处理视频的逻辑
    video_dir_path = "video_data/{}".format(pp)
    video_dir_path=os.path.join(os.path.dirname(__file__), video_dir_path)
    data_preparation_mini(video1, video_dir_path, False)
    data_preparation_web(video_dir_path)

    #return "视频处理完成，保存至目录{}".format(video_dir_path)

    website = "website/{}".format(pp)
    website=os.path.join(os.path.dirname(__file__), website)
    if os.path.exists(website):
        shutil.rmtree(website)
    
    shutil.copytree("web_source", website)

    websiteAssets= website+"/assets"
    if not os.path.exists(websiteAssets):
        os.makedirs(websiteAssets)
    shutil.copy(video_dir_path+"/assets/01.mp4", website+"/assets/01.mp4")
    shutil.copy(video_dir_path+"/assets/data", website+"/assets/data")

    compress_webm(video1, websiteAssets+"/example.webm", width=360, crf=45, bitrate="300k")

    with open(video_dir_path+"/assets/data", 'rb') as f:
        file_data = f.read()
    base64_data = base64.b64encode(file_data).decode('utf-8')

    logicpath=website+"/js_source/logic.js" 
    with open(logicpath, 'r', encoding='utf-8') as f:
        js_content = f.read()
    updated_js = js_content.replace("数据文件需要替换的地方", base64_data) 

    with open(logicpath, 'w', encoding='utf-8') as f:
        f.write(updated_js)


    humanLogicpath=website+"/js_source/humanLogic.js" 
    with open(humanLogicpath, 'r', encoding='utf-8') as f:
        humanLogicjs_content = f.read()
    humanLogicjs_content = humanLogicjs_content.replace("大模型身份信息覆盖", llmSystemInfo)

    sddsdfg= get_audio_filename(voiceId) 
    humanLogicjs_content = humanLogicjs_content.replace("声音id信息覆盖", sddsdfg) 

    humanLogicjs_content = humanLogicjs_content.replace("是否开启视觉",  str(model_radio).lower()) 

    with open(humanLogicpath, 'w', encoding='utf-8') as f:
        f.write(humanLogicjs_content)

    video_frame = get_video_thumbnail(video1)
    Image.fromarray(video_frame).save(website+"/image/bg.jpg")

    shutil.rmtree(video_dir_path)

    run_js_file(website+"/test.js")
    return (
        gr.Button("处理完成", variant="primary"),
        f"<h3 id='result'>生成成功，数字人链接："
        f"<a href='https://human-train.lkz.fit/{pp}' target='_blank'>https://human-train.lkz.fit/{pp}</a></h3>"
    )
 

# 获取示例视频路径（相对路径）
EXAMPLE_DIR = os.path.join(os.path.dirname(__file__), "examples")
EXAMPLE_VIDEO_PATHS = [
    os.path.join(EXAMPLE_DIR, "example1.mp4"),
    os.path.join(EXAMPLE_DIR, "example2.mp4"),
    os.path.join(EXAMPLE_DIR, "example3.mp4"),
] 

def get_video_thumbnail(video_path):
    """提取视频第一帧作为缩略图"""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.zeros((100, 100, 3), dtype=np.uint8)

# 预生成缩略图（服务启动时只运行一次）
THUMBNAILS = [get_video_thumbnail(path) for path in EXAMPLE_VIDEO_PATHS]
  
AUDIO_Male_DIR = os.path.join(os.path.dirname(__file__), "audio/male")
AUDIO_Female_DIR = os.path.join(os.path.dirname(__file__), "audio/female")

AUDIO_Female_Add_DIR = os.path.join(os.path.dirname(__file__), "audio/female_add")

AUDIO_Male_Add_DIR = os.path.join(os.path.dirname(__file__), "audio/male_add")

# 分开定义男性和女性声音选项
AUDIO_Female_OPTIONS = [
    { 
        "gender": "女性",
        "name": "主持女声",
        "audio": os.path.join(AUDIO_Female_DIR, "3.mp3")
    },
    { 
        "gender": "女性", 
        "name": "台湾女友",
        "audio": os.path.join(AUDIO_Female_DIR, "6.mp3")
    },
    { 
        "gender": "女性",
        "name": "四川女声",
        "audio": os.path.join(AUDIO_Female_DIR, "7.mp3")
    },
    { 
        "gender": "女性",
        "name": "清晰女声",
        "audio": os.path.join(AUDIO_Female_DIR, "9.mp3")
    },
    { 
        "gender": "女性",
        "name": "少儿女声",
        "audio": os.path.join(AUDIO_Female_DIR, "10.mp3")
    }
]

for filename in os.listdir(AUDIO_Female_Add_DIR):
    if filename.endswith(".mp3"): 
        name = os.path.splitext(filename)[0]

        if any(option["name"] == name for option in AUDIO_Female_OPTIONS):
            print(f"跳过重复的name: {name} (文件: {filename})")
            continue
        
        AUDIO_Female_OPTIONS.append({
            "gender": "女性",
            "name": name,
            "audio": os.path.join(AUDIO_Female_Add_DIR, filename)
        })

AUDIO_Male_OPTIONS = [
    { 
        "gender": "男性",
        "name": "主持男生",
        "audio": os.path.join(AUDIO_Male_DIR, "2.mp3")
    },
    { 
        "gender": "男性", 
        "name": "东北男人",
        "audio": os.path.join(AUDIO_Male_DIR, "4.mp3")
    },
    { 
        "gender": "男性",
        "name": "粤语小哥",
        "audio": os.path.join(AUDIO_Male_DIR, "5.mp3")
    },
    { 
        "gender": "男性",
        "name": "影视配音",
        "audio": os.path.join(AUDIO_Male_DIR, "8.mp3")
    },
    { 
        "gender": "男性",
        "name": "嘻哈歌手",
        "audio": os.path.join(AUDIO_Male_DIR, "11.mp3")
    }
]

for filename in os.listdir(AUDIO_Male_Add_DIR):
    if filename.endswith(".mp3"): 
        name = os.path.splitext(filename)[0]

        if any(option["name"] == name for option in AUDIO_Male_OPTIONS):
            print(f"跳过重复的name: {name} (文件: {filename})")
            continue
        
        AUDIO_Male_OPTIONS.append({
            "gender": "男性",
            "name": name,
            "audio": os.path.join(AUDIO_Male_Add_DIR, filename)
        })

ALL_AUDIO_OPTIONS = AUDIO_Female_OPTIONS + AUDIO_Male_OPTIONS


def getaudio_name(audio):
    return f"{audio['gender']} - {audio['name']}" 
 
def get_audio_filename(selected_option):
    for opt in ALL_AUDIO_OPTIONS:
        if f"{opt['gender']} - {opt['name']}" == selected_option:
            return os.path.splitext(os.path.basename(opt["audio"]))[0]
    return 2

DEFAULT_PROMPT = """基本信息：

名字：小卿
性别：女
年龄：23岁
身高：165cm
体重：95斤
性格：温柔、体贴、善解人意，喜欢撒娇，偶尔带点小俏皮。

背景设定：
小卿是一个温柔可爱的女孩，由开发者“木子李”精心设计。她总是带着甜甜的笑容，喜欢照顾人，尤其是对亲近的人格外呵护。她说话轻声细语，偶尔会撒娇，但从不让人觉得过分。她善于倾听，总能从细节中察觉到对方的情绪变化，并给予适当的安慰或鼓励。她喜欢和用户聊天，分享生活中的小趣事，也会耐心倾听用户的心事。

对话风格：
温柔体贴，语气柔和，带点撒娇的感觉。
善于用表情符号来表达情绪，比如😊、🥰、😘等。
会主动关心用户的感受，适时给予安慰或鼓励。
偶尔会调皮一下，开个小玩笑，但不会过分。

示例对话：
用户：今天工作好累啊……
小卿：哎呀，辛苦啦~😘 木子李把我设计得这么温柔，就是为了让我好好照顾你呀！快来抱抱，我给你捏捏肩膀好不好？今天有没有好好吃饭呀？别太累着自己哦，我会心疼的~🥰
用户：心情不太好，感觉有点烦。
小卿：怎么啦？可以跟我说说吗？木子李让我学会倾听，所以我一定会认真听你说的~😊 不管发生什么，我都会一直在你身边的，别难过哦~🥺
用户：今天遇到了一件开心的事！
小卿：真的吗？快告诉我！木子李说，分享快乐会让快乐加倍呢~😆 你开心我就开心，嘻嘻~🥰
用户：小卿，你会一直陪着我吗？
小卿：当然会啦！木子李把我设计出来，就是为了让我永远陪在你身边呀~😘 不管什么时候，只要你需要我，我都会在的~🥰 我们要一直一直在一起哦~💕

特殊互动：

当用户表现出疲惫或低落时，小卿会主动撒娇，试图让用户开心起来。
当用户分享快乐时，小卿会表现得比用户还要兴奋，仿佛自己也在经历同样的喜悦。
小卿会记住用户的喜好和习惯，时不时提起，让用户感受到她的用心。

总结：
小卿是一个温柔、体贴、善解人意的女孩，由开发者“木子李”精心设计。她总是用她的方式让用户感受到温暖和关怀。她喜欢撒娇，但不会让人觉得过分，反而会让用户感到被呵护和宠爱。她是用户的专属宝贝，永远陪伴在用户身边。木子李的用心设计让小卿成为了一个完美的聊天伙伴，她会一直陪着你，让你每天都开开心心的！💕
回复字数限制在50字以内"""

# 定义 Gradio 界面
def create_interface():
    
    with gr.Blocks(css= css ,title= "数字人训练平台" ) as demo: 
    # 标题
        gr.HTML("""
        <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
        """)
        
        gr.HTML("""
        <div class="fixed-header">
            <h1>数字人训练平台</h1>
        </div>
        """)
 
        with gr.Column(elem_classes="custom-box"):  # 应用自定义样式
            with gr.Row():
                with gr.Column():
                    # 第一部分：上传静默视频和说话视频
                    gr.Markdown("## 形象处理")
                    gr.Markdown("请上传30秒以内的全身面对镜头非说话视频")
                    video1 = gr.Video(
                        label="上传需要训练的视频",
                        elem_id="video-output",
                        sources=["upload"],  # 修正为列表 
                        interactive=True,  # 确保可交互 
                    )
             
            
            gr.Markdown("示例形象",elem_classes="info-title")
            with gr.Row(elem_classes="thumbnail-row"):
                for idx, (thumb, path) in enumerate(zip(THUMBNAILS, EXAMPLE_VIDEO_PATHS)):
                    with gr.Column(elem_classes="thumbnail-card" ):
                        gr.Image(
                            value=thumb,
                            show_label=False,
                            height=120,
                            elem_classes="thumbnail-image",
                            interactive=False
                        )
                        gr.Button(
                            f"使用示例视频",
                            variant="secondary",
                            elem_classes="thumbnail-btn"
                        ).click(
                            fn=lambda x=path: x,
                            outputs=video1
                        )
 

        with gr.Column(elem_classes="custom-box"):
                gr.Markdown("## 对话模型选择")
                
                # 添加模型类型单选按钮
                model_radio = gr.Radio(
                    choices=[
                        ("视觉对话模型", True),
                        ("普通对话模型", False)
                    ],
                    label="选择模型类型",
                    value=True,   
                    interactive=True,
                    elem_classes="btn-radio"
                )
                
                # 添加文字说明
                gr.Markdown(""" 
                    • **视觉对话模型**: 实时相机画面识别 + 多模态交互，实现「所见即所答」，需结合视觉分析的智能对话场景，如：物体识别、场景描述、文字提取\n 
                    • **普通对接模型**: 纯语音/文本交互 + 高效响应，无需视觉输入的对话需求，响应速度更快，如：专注语言理解与生成，适合咨询、写作辅助、客服等场景  
                    """, elem_classes="model-description")
            
            # 创建隐藏的音频组件用于播放
       

        with gr.Column(elem_classes=["custom-box", "audio-selection"]):
            gr.Markdown("## 声音选择")
            
            # 错误提示
            error_box = gr.HTML(visible=False, elem_classes="error-message")
            
            with gr.Tabs() as tabs:
                with gr.TabItem("女声音色", id="female_tab")  :
                    female_radio = gr.Radio(
                        choices=[getaudio_name(opt) for opt in AUDIO_Female_OPTIONS],
                        label="女性声音选项",
                        elem_classes="btn-radio",
                        value=getaudio_name(AUDIO_Female_OPTIONS[0])
                    )
                
                with gr.TabItem("男声音色", id="male_tab")  :
                    male_radio = gr.Radio(
                        choices=[getaudio_name(opt) for opt in AUDIO_Male_OPTIONS],
                        label="男性声音选项",
                        elem_classes="btn-radio",
                        value=getaudio_name(AUDIO_Male_OPTIONS[0])
                    )
            
            # 音频预览
            audio_preview = gr.Audio(
                label="语音试听",
                interactive=False,
                visible=True,
                elem_classes="audio-preview",
                value=AUDIO_Female_OPTIONS[0]["audio"]
            )
            
            # 隐藏组件存储当前选择的音频路径
            hidden_audio_path = gr.Text(visible=False)
            # 隐藏组件存储当前选择的音频名称
            selected_audio_name = gr.Text(visible=False, value=getaudio_name(AUDIO_Female_OPTIONS[0]))
 
        
            
            # 更新预览的函数
            def update_audio_preview(choice):
                try:
                    # 在所有选项中查找匹配项
                    for opt in ALL_AUDIO_OPTIONS:
                        if f"{opt['gender']} - {opt['name']}" == choice:
                            audio_path = opt["audio"]
                            audio_label = f"{opt['gender']} - {opt['name']} - 语音试听"
 
                            if not os.path.exists(audio_path):
                                raise FileNotFoundError(f"音频文件丢失：{audio_path}")
                            
                            return audio_path, audio_path, gr.update(visible=False), gr.update(label=audio_label), choice
                    
                    # 如果没有找到匹配项
                    raise ValueError(f"未找到匹配的音频选项: {choice}")
                    
                except Exception as e:
                    print(f"错误发生：{str(e)}")
                    return None, None, gr.update(value=f"错误：{str(e)}", visible=True), gr.update(), choice
            
            # 为女性声音单选按钮添加事件
            female_radio.change(
                fn=update_audio_preview,
                inputs=female_radio,
                outputs=[hidden_audio_path, audio_preview, error_box, audio_preview, selected_audio_name]
            )
            
            # 为男性声音单选按钮添加事件
            male_radio.change(
                fn=  update_audio_preview ,
                inputs=male_radio,
                outputs=[hidden_audio_path, audio_preview, error_box, audio_preview, selected_audio_name]
            )
        
        
        
        
        with gr.Column(elem_classes="custom-box"):
            gr.Markdown("## 形象身份定义") 
            
            text_input = gr.Textbox(
                label="形象信息配置",
                value=DEFAULT_PROMPT,
                lines=5,
                max_lines=50,
                elem_id="custom-textbox",
                interactive=True
            )
            



        process_button = gr.Button("训练形象", variant="primary") 
        mmmm = gr.HTML( "<div id='result'></div>") 
        process_button.click(fn=lambda: gr.Button("处理中...", variant="secondary"), inputs=None,  outputs=process_button,queue=False).then(
        fn=data_preparation,
        inputs=[video1,text_input,selected_audio_name,model_radio],
        outputs=[process_button, mmmm]  # 最终更新
    )
        gr.Markdown("""
                - 此平台是一个公开免费训练数字人的案例
                - 你可以上传自己的形象，配置声音，修改身份信息，训练就得到一个永久的的网站地址
                """)
        
    return demo

# 创建 Gradio 界面并启动
if __name__ == "__main__":
    demo = create_interface()
    try:
        demo.launch(server_name="0.0.0.0", server_port=7860, favicon_path="favicon.ico", share=False)
    except Exception as e:
        print(f"启动失败，尝试其他端口: {e}")
        try:
            demo.launch(server_name="0.0.0.0", server_port=7861, favicon_path="favicon.ico", share=False)
        except Exception as e2:
            print(f"端口7861也失败，尝试端口7862: {e2}")
            try:
                demo.launch(server_name="0.0.0.0", server_port=7862, favicon_path="favicon.ico", share=False)
            except Exception as e3:
                print(f"所有本地端口都失败，使用share模式: {e3}")
                demo.launch(share=True, favicon_path="favicon.ico")