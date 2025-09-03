async function Initialized() {
  await new Promise(resolve => setTimeout(resolve, 500));
  document.getElementById('overlayButton').innerHTML = '<span style="font-size: 28px; position: relative; top: -100px; ">数字人在呼叫你</span> <img src="image/receive1.png" width="100" height="100" style=" position: relative; bottom:  -200px; "/> <span style="font-size: 23px; position: relative; bottom:  -190px; ">聊天</span>';
  document.getElementById('overlayButton').style.pointerEvents = 'auto'
}


isCreateAudioFinish = false;
isVLM = 是否开启视觉;
 
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

const socket = new WebSocket('wss://2dhuman.lkz.fit/recognition?isSendConfig=true&isFree=true');
//const socket = new WebSocket('ws://localhost:19465/recognition?isSendConfig=true&isFree=true');
socket.addEventListener('open', (event) => {
  const systemMessage = `大模型身份信息覆盖`;
  const voiceType = "声音id信息覆盖"; 

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