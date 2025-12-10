#!/usr/bin/env python3
"""
示例：如何调用 HeAR Docker 服务

这个脚本演示了如何通过 HTTP API 调用运行在 Docker 容器中的 HeAR 服务。

服务端点：
- 健康检查: http://localhost:8080/health
- 预测: http://localhost:8080/predict

API 返回格式：

1. 成功响应：
   {
     "predictions": [
       {
         "embedding": [0.123, -0.456, 0.789, ...]  # 512 维浮点数列表
       }
     ]
   }

2. 错误响应（某个实例处理失败）：
   {
     "predictions": [
       {
         "error": {
           "description": "错误描述信息"
         }
       }
     ]
   }

注意：
- 每个请求中的 instance 对应一个 prediction
- embedding 是 512 维的浮点数向量
- 批量请求时，predictions 数组的长度等于 instances 数组的长度

使用方法：
1. 确保 Docker 容器正在运行：
   docker run -d -p 8080:8080 --gpus device=7 \
     -v /home/ubuntu/shared:/shared \
     -v /home/ubuntu/code:/code \
     hear-serving

2. 运行此脚本：
   python3 call_hear_service_example.py
"""

import base64
import json
import requests
import numpy as np
import librosa
from pathlib import Path


# 服务配置
SERVICE_URL = "http://localhost:8080"
HEALTH_ENDPOINT = f"{SERVICE_URL}/health"
PREDICT_ENDPOINT = f"{SERVICE_URL}/predict"

# HeAR 模型要求
SAMPLE_RATE = 16000  # 16kHz
CLIP_DURATION = 2    # 2秒
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # 32000 个样本


def check_health():
    """检查服务健康状态"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        if response.status_code == 200:
            print(f"✓ 服务健康: {response.text}")
            return True
        else:
            print(f"✗ 服务不健康: {response.status_code} - {response.text}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"✗ 无法连接到服务: {e}")
        return False


def audio_to_base64_wav(audio_array: np.ndarray, sample_rate: int = SAMPLE_RATE) -> str:
    """将音频数组转换为 base64 编码的 WAV 文件"""
    import io
    import wave
    
    # 确保音频是单声道、16kHz、2秒
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    # 归一化到 [-1, 1] 范围
    if audio_array.max() > 1.0 or audio_array.min() < -1.0:
        audio_array = audio_array / np.max(np.abs(audio_array))
    
    # 转换为 int16
    audio_int16 = (audio_array * 32767).astype(np.int16)
    
    # 创建 WAV 文件
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # 单声道
        wav_file.setsampwidth(2)  # 16位 = 2字节
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    # 转换为 base64
    wav_bytes = wav_buffer.getvalue()
    return base64.b64encode(wav_bytes).decode('utf-8')


def predict_with_audio_file(audio_file_path: str) -> dict:
    """
    使用音频文件进行预测
    
    Args:
        audio_file_path: 音频文件路径
        
    Returns:
        包含嵌入向量的响应字典
    """
    # 加载音频文件
    print(f"\n加载音频文件: {audio_file_path}")
    audio, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
    
    # 如果音频长度不足 2 秒，进行填充
    if len(audio) < CLIP_LENGTH:
        audio = np.pad(audio, (0, CLIP_LENGTH - len(audio)), mode='constant')
    elif len(audio) > CLIP_LENGTH:
        audio = audio[:CLIP_LENGTH]
    
    # 转换为 base64 WAV
    wav_base64 = audio_to_base64_wav(audio, SAMPLE_RATE)
    
    # 构建请求
    request_data = {
        "instances": [
            {
                "input_bytes": wav_base64
            }
        ]
    }
    
    # 发送请求
    print(f"发送预测请求到: {PREDICT_ENDPOINT}")
    response = requests.post(PREDICT_ENDPOINT, json=request_data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 预测成功")
        return result
    else:
        print(f"✗ 预测失败: {response.status_code}")
        print(f"  错误信息: {response.text}")
        return None


def predict_with_audio_array(audio_array: np.ndarray) -> dict:
    """
    使用音频数组进行预测
    
    Args:
        audio_array: 形状为 (32000,) 的 numpy 数组，表示 2 秒 16kHz 音频
        
    Returns:
        包含嵌入向量的响应字典
    """
    # 确保音频格式正确
    if len(audio_array.shape) > 1:
        audio_array = np.mean(audio_array, axis=1)
    
    if len(audio_array) != CLIP_LENGTH:
        raise ValueError(f"音频数组长度必须是 {CLIP_LENGTH}，当前是 {len(audio_array)}")
    
    # 构建请求 - 使用 input_array 格式
    request_data = {
        "instances": [
            {
                "input_array": audio_array.tolist()
            }
        ]
    }
    
    # 发送请求
    print(f"发送预测请求到: {PREDICT_ENDPOINT}")
    response = requests.post(PREDICT_ENDPOINT, json=request_data, timeout=30)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 预测成功")
        return result
    else:
        print(f"✗ 预测失败: {response.status_code}")
        print(f"  错误信息: {response.text}")
        return None


def predict_batch(audio_files: list[str]) -> dict:
    """
    批量预测多个音频文件
    
    Args:
        audio_files: 音频文件路径列表
        
    Returns:
        包含所有嵌入向量的响应字典
    """
    instances = []
    
    for audio_file in audio_files:
        # 加载音频
        audio, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True, duration=CLIP_DURATION)
        
        # 确保长度正确
        if len(audio) < CLIP_LENGTH:
            audio = np.pad(audio, (0, CLIP_LENGTH - len(audio)), mode='constant')
        elif len(audio) > CLIP_LENGTH:
            audio = audio[:CLIP_LENGTH]
        
        # 转换为 base64
        wav_base64 = audio_to_base64_wav(audio, SAMPLE_RATE)
        instances.append({"input_bytes": wav_base64})
    
    # 构建请求
    request_data = {"instances": instances}
    
    # 发送请求
    print(f"发送批量预测请求 ({len(instances)} 个实例) 到: {PREDICT_ENDPOINT}")
    response = requests.post(PREDICT_ENDPOINT, json=request_data, timeout=60)
    
    if response.status_code == 200:
        result = response.json()
        print(f"✓ 批量预测成功，返回 {len(result.get('predictions', []))} 个结果")
        return result
    else:
        print(f"✗ 批量预测失败: {response.status_code}")
        print(f"  错误信息: {response.text}")
        return None


def main():
    """主函数：演示各种调用方式"""
    print("=" * 60)
    print("HeAR 服务调用示例")
    print("=" * 60)
    
    # 1. 检查服务健康状态
    print("\n1. 检查服务健康状态")
    if not check_health():
        print("\n请确保 Docker 容器正在运行！")
        print("运行命令示例：")
        print("  docker run -d -p 8089:8080 --gpus device=7 \\")
        print("    -v /home/ubuntu/shared:/shared \\")
        print("    -v /home/ubuntu/code:/code \\")
        print("    hear-serving")
        return
    
    # 2. 使用音频文件进行预测
    print("\n2. 使用音频文件进行预测")
    # 示例：如果有音频文件
    # audio_file = "/path/to/your/audio.wav"
    # result = predict_with_audio_file(audio_file)
    # if result:
    #     embedding = result['predictions'][0]['embedding']
    #     print(f"嵌入向量维度: {len(embedding)}")
    #     print(f"嵌入向量前5个值: {embedding[:5]}")
    
    # 3. 使用随机音频数组进行预测
    print("\n3. 使用随机音频数组进行预测")
    random_audio = np.random.normal(size=CLIP_LENGTH).astype(np.float32)
    result = predict_with_audio_array(random_audio)
    if result:
        embedding = result['predictions'][0]['embedding']
        print(f"嵌入向量维度: {len(embedding)}")
        print(f"嵌入向量前5个值: {embedding[:5]}")
        print(f"嵌入向量统计: min={min(embedding):.4f}, max={max(embedding):.4f}, mean={np.mean(embedding):.4f}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

