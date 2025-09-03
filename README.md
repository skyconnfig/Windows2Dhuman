# Windows2Dhuman - 2D数字人项目

## 项目简介

这是一个基于Web技术的2D数字人交互系统，支持实时语音识别、文本对话和数字人动画展示。项目集成了语音识别、自然语言处理和数字人渲染技术，为用户提供沉浸式的数字人交互体验。

## 主要功能

- **实时语音识别**: 支持麦克风输入，实时转换语音为文字
- **智能对话系统**: 集成大语言模型，提供自然的对话体验
- **数字人动画**: 2D数字人角色动画展示
- **多语音类型**: 支持多种语音合成选项
- **Web界面**: 现代化的Web用户界面
- **WebSocket通信**: 实时双向通信支持

## 技术栈

### 后端
- Python 3.x
- Gradio (Web界面框架)
- WebSocket (实时通信)
- 语音识别和合成相关库

### 前端
- HTML5/CSS3/JavaScript
- WebSocket客户端
- 现代浏览器API (麦克风访问等)

### 依赖管理
- Node.js (前端构建工具)
- npm (包管理)

## 安装说明

### 环境要求
- Python 3.8+
- Node.js 16+
- 现代浏览器 (Chrome, Firefox, Edge等)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/skyconnfig/Windows2Dhuman.git
   cd Windows2Dhuman
   ```

2. **创建Python虚拟环境**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **安装Python依赖**
   ```bash
   pip install -r requirements.txt
   ```

4. **安装Node.js依赖**
   ```bash
   npm install
   ```

## 使用方法

### 启动应用

1. **启动Python后端服务**
   ```bash
   python app.py
   ```

2. **启动Web服务器**
   - 进入 `website` 目录
   - 运行相应的启动脚本
   - 默认访问地址: `http://localhost:8082`

### 功能使用

1. **语音交互**
   - 点击麦克风按钮开始录音
   - 说话后系统会自动识别并转换为文字
   - 数字人会根据对话内容做出相应回应

2. **文本输入**
   - 在输入框中直接输入文字
   - 按回车或点击发送按钮提交

## 项目结构

```
Windows2Dhuman/
├── app.py                 # 主应用入口
├── requirements.txt       # Python依赖
├── package.json          # Node.js依赖
├── web_source/           # Web前端源码
│   ├── jsCode15/        # 生产环境JS代码
│   ├── js_source/       # 开发环境JS源码
│   └── index.html       # 主页面
├── audio/               # 音频资源
├── models/              # AI模型文件
├── data/                # 数据文件
└── venv/                # Python虚拟环境
```

## 最近更新

### v1.1.0 (最新)
- **修复WebSocket连接稳定性问题**: 解决了"Unchecked runtime.lastError: The message port closed before a response was received"错误
- **增强错误处理机制**: 添加了WebSocket重连机制和安全消息发送功能
- **优化连接状态检查**: 改进了连接状态判断逻辑
- **更新依赖包**: 确保所有Node.js依赖包为最新版本

### 技术改进
- 实现了`sendWebSocketMessage`安全消息发送函数
- 添加了`createWebSocketConnection`连接管理函数
- 优化了`StartMicrophone`和`handleInput`函数的WebSocket使用
- 增加了指数退避重连机制

## 开发说明

### 开发环境设置
1. 确保安装了所有依赖
2. 使用开发模式启动服务
3. 前端代码修改后需要从`js_source`复制到`jsCode15`

### 代码规范
- 遵循Python PEP8规范
- JavaScript使用ES6+语法
- 添加适当的注释和文档

## 故障排除

### 常见问题

1. **WebSocket连接失败**
   - 检查防火墙设置
   - 确认端口未被占用
   - 查看浏览器控制台错误信息

2. **麦克风权限问题**
   - 确保浏览器已授予麦克风权限
   - 使用HTTPS或localhost访问

3. **依赖安装失败**
   - 检查Python和Node.js版本
   - 尝试清除缓存后重新安装

## 贡献指南

欢迎提交Issue和Pull Request来改进项目。请确保：
- 代码符合项目规范
- 添加适当的测试
- 更新相关文档

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，请通过以下方式联系：
- 提交GitHub Issue
- 发送邮件至项目维护者

---

*最后更新时间: 2024年12月*