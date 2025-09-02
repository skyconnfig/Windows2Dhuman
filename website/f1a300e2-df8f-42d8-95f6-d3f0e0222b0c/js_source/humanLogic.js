async function Initialized() {
  await new Promise(resolve => setTimeout(resolve, 500));
  document.getElementById('overlayButton').innerHTML = '<span style="font-size: 28px; position: relative; top: -100px; ">数字人在呼叫你</span> <img src="image/receive1.png" width="100" height="100" style=" position: relative; bottom:  -200px; "/> <span style="font-size: 23px; position: relative; bottom:  -190px; ">聊天</span>';
  document.getElementById('overlayButton').style.pointerEvents = 'auto'
}


isCreateAudioFinish = false;
isVLM = true;
 
path = ".";
const mp4url = path + '/assets/01.mp4';

let audioContext;
let isPlaying = false;
let IsRecogition = true;
let currentInputModelIsVoice = true;

let TextInputDiv = document.getElementById('TextInputDiv')

let VoiceInputDiv = document.getElementById('VoiceInputDiv')

let microphoneStream = null



if (isVLM) {

  currentFacingMode = "environment"
  cameraContainer = document.createElement('div');
  cameraContainer.id = 'camera-container';

  cameraView = document.createElement('video');
  cameraView.id = 'camera-view';
  cameraView.autoplay = true;
  cameraView.playsInline = true;
  cameraView.disablePictureInPicture = true;


  switchCameraBtn = document.createElement('button');
  switchCameraBtn.id = 'camera-toggle-btn';
  switchCameraBtn.setAttribute('aria-label', '切换摄像头');
  switchCameraBtn.innerHTML = '↻';


  // 添加到DOM
  cameraContainer.appendChild(cameraView);
  cameraContainer.appendChild(switchCameraBtn);
  document.body.appendChild(cameraContainer);



  let isDragging = false;
  let offsetX, offsetY;
  cameraContainer.addEventListener('mousedown', startDrag);
  document.addEventListener('mousemove', drag);
  document.addEventListener('mouseup', endDrag);

  // 触摸事件处理
  switchCameraBtn.addEventListener('touchstart', handleButtonTouch, { passive: true });
  cameraContainer.addEventListener('touchstart', handleTouchStart, { passive: false });
  document.addEventListener('touchmove', handleTouchMove, { passive: false });
  document.addEventListener('touchend', handleTouchEnd);


  switchCameraBtn.addEventListener('click', toggleCamera);

  function toggleCamera() {
    currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';


    startCamera();
  }


  function handleButtonTouch(e) {
    // 阻止事件冒泡，确保按钮能接收到触摸事件
    e.stopPropagation();
    // 添加触摸反馈
    switchCameraBtn.style.transform = 'scale(0.9)';
    switchCameraBtn.style.backgroundColor = 'rgba(0, 0, 0, 0.7)';
  }


  // 鼠标拖动函数
  function startDrag(e) {
    isDragging = true;
    offsetX = e.clientX - cameraContainer.getBoundingClientRect().left;
    offsetY = e.clientY - cameraContainer.getBoundingClientRect().top;
    cameraContainer.style.cursor = 'grabbing';
    e.preventDefault();
  }

  function drag(e) {
    if (!isDragging) return;
    updatePosition(e.clientX, e.clientY);
    e.preventDefault();
  }

  function endDrag() {
    isDragging = false;
    cameraContainer.style.cursor = 'move';
  }

  // 触摸拖动函数
  function handleTouchStart(e) {
    if (e.touches.length === 1) {
      isDragging = true;
      const touch = e.touches[0];
      offsetX = touch.clientX - cameraContainer.getBoundingClientRect().left;
      offsetY = touch.clientY - cameraContainer.getBoundingClientRect().top;
      e.preventDefault(); // 阻止默认滚动行为
    }
  }

  function handleTouchMove(e) {
    if (isDragging && e.touches.length === 1) {
      const touch = e.touches[0];
      updatePosition(touch.clientX, touch.clientY);
      e.preventDefault(); // 阻止默认滚动行为
    }
  }

  function handleTouchEnd() {
    isDragging = false;
  }

  // 更新位置函数（共用）
  function updatePosition(clientX, clientY) {
    let newLeft = clientX - offsetX;
    let newTop = clientY - offsetY;

    // 获取窗口尺寸和容器尺寸
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const containerWidth = cameraContainer.offsetWidth;
    const containerHeight = cameraContainer.offsetHeight;

    // 限制边界
    newLeft = Math.max(0, Math.min(newLeft, windowWidth - containerWidth));
    newTop = Math.max(0, Math.min(newTop, windowHeight - containerHeight));

    cameraContainer.style.left = `${newLeft}px`;
    cameraContainer.style.top = `${newTop}px`;
  }

  let currentCameraStream = null;
  function startCamera() {

    stopCameraStream();
    const constraints = {
      video: {
        facingMode: currentFacingMode
      },
      audio: false
    };



    navigator.mediaDevices.getUserMedia(constraints)
      .then(stream => {
        currentCameraStream = stream
        cameraView.srcObject = stream;

        // 禁用画中画功能
        if ('disablePictureInPicture' in cameraView) {
          cameraView.disablePictureInPicture = true;
        }
      })
      .catch(err => {
        console.error("摄像头错误:", err);
        alert(`视觉识别需要访问摄像头`);
      });

  }

  function stopCameraStream() {
    if (currentCameraStream) {
      currentCameraStream.getTracks().forEach(track => track.stop());
      currentCameraStream = null;
    }
  }


  startCamera()


  const magicImageHeader = new Uint8Array([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09]);
  captureInterval = setInterval(async () => {
    if (IsRecogition) {
      imageBlob = await getCameraFrameAsJpeg();

      const arrayBuffer = await imageBlob.arrayBuffer();
      const imageData = new Uint8Array(arrayBuffer);
      const combinedData = new Uint8Array(magicImageHeader.length + imageData.length);

      combinedData.set(magicImageHeader, 0);          // 写入头部
      combinedData.set(imageData, magicImageHeader.length); // 写入图像数据


      socket.send(combinedData);
    }

  }, 1000)


}

function getCameraFrameAsJpeg() {
  return new Promise((resolve, reject) => {

    const CameraCanvas = document.createElement('canvas');
    CameraCanvas.width = cameraView.videoWidth;
    CameraCanvas.height = cameraView.videoHeight;
    const CameraCtx = CameraCanvas.getContext('2d');
    CameraCtx.drawImage(cameraView, 0, 0, CameraCanvas.width, CameraCanvas.height);

    // 转换为WebP格式的base64
    CameraCanvas.toBlob(
      (blob) => {
        if (!blob) reject(new Error('无法创建Blob'));
        else resolve(blob); // 直接返回Blob，避免Base64转换
      },
      'image/jpeg',
      0.8
    );
  });
}


function StartMicrophone() {
  if (microphoneStream == null) {
    console.log("startMicrophoneStream")
    navigator.mediaDevices
      .getUserMedia({
        audio: true
      })
      .then(stream => {
        microphoneStream = stream;

        const audioContext = new window.AudioContext();
        const microphoneSource = audioContext.createMediaStreamSource(microphoneStream);


        const scriptProcessor = audioContext.createScriptProcessor(4096, 1, 1);

        const biquadFilter = audioContext.createBiquadFilter();
        biquadFilter.type = "lowpass";
        biquadFilter.frequency.setValueAtTime(8000, audioContext.currentTime); // Set cut-off frequency to 8kHz

        microphoneSource.connect(biquadFilter);
        biquadFilter.connect(scriptProcessor);

        scriptProcessor.onaudioprocess = event => {
          if (!IsRecogition)
            return;

          else if (socket == null || socket.readyState != WebSocket.OPEN)
            return;

          let audioData = event.inputBuffer.getChannelData(0);
          const sampleRateRatio = audioContext.sampleRate / 16000;
          const newLength = Math.round(audioData.length / sampleRateRatio);
          const resampledData = new Float32Array(newLength);

          for (let i = 0; i < newLength; i++) {
            resampledData[i] = audioData[Math.round(i * sampleRateRatio)];
          }

          socket.send(floatTo16BitPCM(resampledData));
        };

        scriptProcessor.connect(audioContext.destination);
      })
      .catch(error => {
        console.error('error:', error);
      });
  }
}

function floatTo16BitPCM(samples) {
  const length = samples.length;
  const int16Array = new Int16Array(length);

  for (let i = 0; i < length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
  }

  return int16Array;
}


const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);


const videoOverlay = document.getElementById('videoOverlay');
const overlayButton = document.getElementById('overlayButton');

if (isMobile) {
  videoOverlay.style.display = 'block';
  videoOverlay.remove();
}

// 监听按钮点击事件
overlayButton.addEventListener('click', async () => {
  
  overlayButton.remove();
  switchInputMode(true)
 
  document.getElementsByClassName('poster')[0].remove();
  videoOverlay.remove();


});


// 获取滚动容器
const scrollContainer = document.getElementById('scrollContainer');

// 动态添加元素
function addItem(position) {
  // 创建外层容器
  const item = document.createElement('div');
  item.className = 'item';
  if (position === 'left') {
    item.classList.add('align-left'); // 靠左
  } else {
    item.classList.add('align-right'); // 靠右
  }
  // 创建左边或右边元素
  const element = document.createElement('div');
  element.className = position; // 设置类名（left 或 right）
  //element.textContent = `这是一个${position === 'left' ? '左边' : '右边'}元素`; // 设置内容

  // 将元素添加到外层容器
  item.appendChild(element);

  // 将外层容器添加到滚动容器中
  scrollContainer.appendChild(item);

  // 滚动到底部
  scrollContainer.scrollTo({
    top: scrollContainer.scrollHeight, // 滚动到底部
    behavior: 'smooth' // 平滑滚动
  });

  return element;
}

let audioQueue = [];


// 监听播放完成事件
async function PlayEnd() {

  isPlaying = false;

  if (audioQueue.length > 0) {

    PlayWav()
  } else if (isCreateAudioFinish) {
    await new Promise(resolve => setTimeout(resolve, 500));
    IsRecogition = currentInputModelIsVoice;
  }

};

//const socket = new WebSocket('wss://2dhuman.lkz.fit/recognition?isSendConfig=true&isFree=true');
const socket = new WebSocket('ws://localhost:19465/recognition?isSendConfig=true&isFree=true');
socket.addEventListener('open', (event) => {
  const systemMessage = `基本信息：

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
回复字数限制在50字以内`;
  const voiceType = "卿卿女声"; 

  const data = {
    systemMessage: systemMessage,
    voiceType: voiceType, 
    isvlm: isVLM
  };

  const jsonString = JSON.stringify(data);

  socket.send(jsonString)

});


let rightContent = null;
let llmShowContent;
socket.addEventListener('message', (event) => {
  //  console.log(event.data) 
  try {
    // 解析接收到的 JSON 数据
    const jsonData = JSON.parse(event.data);

    if (jsonData.DataType == "TTS") {

      const data = jsonData.Data;
      const dataJson = JSON.parse(data);

      if (!(/^\s*$/.test(dataJson.AudioData))) {
        const audioUint8Array = base64ToArrayBuffer(dataJson.AudioData);
        audioQueue.push(audioUint8Array)
        PlayWav();
      }

      llmShowContent.textContent += dataJson.Text;

      // voiceInfoDiv.textContent = "正在讲话..."

      // 滚动到底部
      scrollContainer.scrollTo({
        top: scrollContainer.scrollHeight, // 滚动到底部
        behavior: 'smooth' // 平滑滚动
      });
    }
    else if (jsonData.DataType == "VoiceRecognitionDelta") { 
      const data = jsonData.Data;
      if (rightContent == null) {
        rightContent = addItem('right')
        rightContent.textContent = data;
      }
      else {
        rightContent.textContent += data;
      }
    }
    else if (jsonData.DataType == "StartLLM") {
      isCreateAudioFinish = false;
      const data = jsonData.Data;
      if (rightContent == null) 
        rightContent = addItem('right')
      rightContent.textContent = data;
      rightContent = null;

      IsRecogition = false;

      llmShowContent = addItem('left')

      //  voiceInfoDiv.textContent = "正在思考..." 

      audioQueue = [];

      if (!audioContext || audioContext.state === 'closed') {
        audioContext = new (window.AudioContext || window.webkitAudioContext)({
          latencyHint: 'interactive',
        });
      } else if (audioContext.state === 'suspended') {
        audioContext.resume(); // 如果处于暂停状态，则恢复
      }
    }
    else if (jsonData.DataType == "Finish") {
      isCreateAudioFinish = true;
    }

  } catch (error) {
    console.error('解析 JSON 失败:', error);
  }
});

const userInput = document.getElementById('userInput');
const submitButton = document.getElementById('submitButton');

document.querySelector('button[name="switchVoice"]').addEventListener('click', () => {
  switchInputMode(true)
});
document.querySelector('button[name="switchText"]').addEventListener('click', () => {
  switchInputMode(false)
});

function handleInput() {
  const inputValue = userInput.value; // 获取输入框内容并去除首尾空格
  if (!(!inputValue || inputValue.trim() === '')) {
    socket.send(inputValue);
  }
  userInput.value = ""
}

// 监听输入框的回车键事件
userInput.addEventListener('keydown', (event) => {
  if (event.key === 'Enter') { // 如果按下的是回车键
    handleInput(); // 调用回调函数
  }
});

// 监听按钮的点击事件
submitButton.addEventListener('click', handleInput);


window.onload = function () { };

const canvas = document.getElementById('canvas_video');
const ctx = canvas.getContext('2d');


function handleKeyDown(event) {
  if (event.key === 'Enter') { // 如果按下的是回车键
    handleInput(); // 调用处理函数
  }
}

function switchInputMode(isVoice) {
  IsRecogition = currentInputModelIsVoice = isVoice;

  VoiceInputDiv.classList.remove('hide-with-animation');
  VoiceInputDiv.classList.remove('show-with-animation');

  TextInputDiv.classList.remove('hide-with-animation');
  TextInputDiv.classList.remove('show-with-animation');
  if (IsRecogition) {

    StartMicrophone();


    VoiceInputDiv.style.display = "flex"

    VoiceInputDiv.style.pointerEvents = "auto";
    VoiceInputDiv.classList.add('show-with-animation');


    TextInputDiv.classList.add('hide-with-animation');

  } else {
    TextInputDiv.style.display = "flex"
    TextInputDiv.classList.add('show-with-animation');
    VoiceInputDiv.style.pointerEvents = "none";
    VoiceInputDiv.classList.add('hide-with-animation');

  }

}



function base64ToArrayBuffer(base64) { 
  const binaryString = atob(base64);
 
  const length = binaryString.length;
  const bytes = new Uint8Array(length);
 
  for (let i = 0; i < length; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
 
  return bytes.buffer;
}


function PlayWav() {

  if (isPlaying)
    return;

  isPlaying = true;


  const audioUint8Array = audioQueue.shift();

  playWavAudio(audioUint8Array)
  audioContext.decodeAudioData(audioUint8Array, function (audioBuffer) {

    // 创建BufferSource节点
    const source = audioContext.createBufferSource();
    source.buffer = audioBuffer;
    // 连接到输出并播放
    source.connect(audioContext.destination);
    source.start(0);


    // 当音频播放结束时释放资源
    source.onended = PlayEnd;

  })
}

switchInputMode(false)

document.addEventListener('contextmenu', function (e) {
  e.preventDefault();
});
document.addEventListener('keydown', function (e) {
  // 禁用 F12
  if (e.key === 'F12') {
    e.preventDefault();
  }
  // 禁用 Ctrl+Shift+I
  if (e.ctrlKey && e.shiftKey && e.key === 'I') {
    e.preventDefault();
  }
  // 禁用 Ctrl+U
  if (e.ctrlKey && e.key === 'u') {
    e.preventDefault();
  }
});

setInterval(function () {
  const startTime = performance.now();
  debugger;
  const endTime = performance.now();
  if (endTime - startTime > 100) { // 如果时间差较大，说明可能打开了开发者工具
    alert('开发者工具已打开，请关闭！');
    window.location.reload();
  }
}, 1000);
// window.addEventListener('resize', resizeCanvas);



addItem('left').textContent = `你好呀！我是你的数字人朋友，有什么可以帮到你的！`;