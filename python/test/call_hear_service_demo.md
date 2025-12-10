# 调用 HeAR Docker 服务示例

这个文档演示了如何通过 HTTP API 调用运行在 Docker 容器中的 HeAR 服务。

## 前提条件

1. **确保 Docker 容器正在运行**：
   ```bash
   docker run -d -p 8089:8080 --gpus device=7 \
     -v /home/ubuntu/shared:/shared \
     -v /home/ubuntu/code:/code \
     hear-serving
   ```

2. **检查服务状态**：
   ```bash
   curl http://localhost:8089/health
   ```
   应该返回：`ok`

## API 端点

- **健康检查**: `GET http://localhost:8089/health`
- **预测**: `POST http://localhost:8089/predict`

## 请求格式

### 方式 1: 使用 base64 编码的 WAV 文件

```python
import requests
import base64
import librosa
import numpy as np

# 加载音频文件
audio, sr = librosa.load("audio.wav", sr=16000, mono=True, duration=2.0)

# 转换为 base64 WAV（需要实现转换函数，见 call_hear_service_example.py）
wav_base64 = audio_to_base64_wav(audio)

# 构建请求
request_data = {
    "instances": [
        {
            "input_bytes": wav_base64
        }
    ]
}

# 发送请求
response = requests.post("http://localhost:8089/predict", json=request_data)
result = response.json()

# 获取嵌入向量
embedding = result['predictions'][0]['embedding']
print(f"嵌入向量维度: {len(embedding)}")  # 应该是 512
```

### 方式 2: 使用音频数组

```python
import requests
import numpy as np

# 准备音频数组（必须是 32000 个样本，即 2 秒 16kHz）
audio_array = np.random.normal(size=32000).astype(np.float32)

# 构建请求
request_data = {
    "instances": [
        {
            "input_array": audio_array.tolist()
        }
    ]
}

# 发送请求
response = requests.post("http://localhost:8089/predict", json=request_data)
result = response.json()

# 获取嵌入向量
embedding = result['predictions'][0]['embedding']
```

### 方式 3: 批量预测

```python
import requests

# 准备多个音频实例
request_data = {
    "instances": [
        {"input_array": audio1.tolist()},
        {"input_bytes": wav_base64_2},
        {"input_array": audio3.tolist()},
    ]
}

# 发送请求
response = requests.post("http://localhost:8089/predict", json=request_data)
result = response.json()

# 获取所有嵌入向量
for i, prediction in enumerate(result['predictions']):
    embedding = prediction['embedding']
    print(f"实例 {i+1} 的嵌入向量维度: {len(embedding)}")
```

## 响应格式

成功响应：
```json
{
  "predictions": [
    {
      "embedding": [0.123, -0.456, 0.789, ...]  // 512 维向量
    }
  ]
}
```

错误响应：
```json
{
  "error": {
    "description": "错误描述信息"
  }
}
```

## 完整示例

查看 `call_hear_service_example.py` 获取完整的 Python 脚本示例。

## 与本地模型调用的区别

| 特性 | 本地模型（hear_event_detector_demo.ipynb） | Docker 服务（本示例） |
|------|------------------------------------------|---------------------|
| 模型加载 | 从 Hugging Face 直接加载 | 模型已在容器中加载 |
| 推理方式 | 直接调用模型函数 | HTTP API 调用 |
| 适用场景 | 开发、实验、单机使用 | 生产环境、多客户端、分布式 |
| 资源隔离 | 与 Python 环境共享 | 独立的容器环境 |
| 扩展性 | 受限于单机资源 | 可以水平扩展 |

