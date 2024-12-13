'''
# This file is use to testing the effect of the combfilter
# Test result: 
# 改变延迟时间：
# 延迟时间越长，声音的重复间隔越大，回声感越明显。
# 延迟时间越短，声音会变得“紧凑”，类似金属或机械声。
# 改变反馈系数：
# 反馈系数越高，回声的音量衰减越慢，声音更加丰富。
# 反馈系数越低，回声衰减得更快，声音更加干净自然。
'''

import numpy as np
import pyaudio
import tkinter as tk
from tkinter import ttk
import threading
import wave

# 读取 WAV 文件作为音频输入
input_wav_file = "author.wav"  # 替换为您的 .wav 文件路径

# 打开 WAV 文件
wav = wave.open(input_wav_file, 'rb')
fs = wav.getframerate()  # 采样率
n_channels = wav.getnchannels()  # 通道数
sample_width = wav.getsampwidth()  # 样本宽度
frames = wav.readframes(wav.getnframes())  # 所有音频帧
wav.close()

# 将音频数据转换为 NumPy 数组
audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
audio_data /= np.max(np.abs(audio_data))  # 归一化音频数据到 [-1, 1]

# 如果是立体声，取单声道
if n_channels > 1:
    audio_data = audio_data[::n_channels]

# 默认梳状滤波器参数
default_delay_time = 0.02  # 默认延迟时间（秒）
default_feedback_coeff = 0.8  # 默认反馈系数

# 创建实时调节的全局变量
current_delay_time = default_delay_time
current_feedback_coeff = default_feedback_coeff

# 梳状滤波器实现
def feedback_comb_filter(x, delay_samples, feedback_coefficient):
    y = np.zeros_like(x)
    for n in range(len(x)):
        if n < delay_samples:
            y[n] = x[n]
        else:
            y[n] = x[n] + feedback_coefficient * y[n - delay_samples]
    return y

# 实时音频播放线程
stop_flag = False

def audio_player():
    global stop_flag, current_delay_time, current_feedback_coeff

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)

    while not stop_flag:
        # 动态计算延迟样本数
        delay_samples = int(current_delay_time * fs)
        # 应用梳状滤波器
        filtered_signal = feedback_comb_filter(audio_data, delay_samples, current_feedback_coeff)
        # 播放音频
        stream.write(filtered_signal.astype(np.float32).tobytes())

    stream.stop_stream()
    stream.close()
    p.terminate()

# 启动播放线程
player_thread = threading.Thread(target=audio_player)
player_thread.start()

# 更新滑块值的回调函数
def update_delay_time(val):
    global current_delay_time
    current_delay_time = float(val)

def update_feedback_coeff(val):
    global current_feedback_coeff
    current_feedback_coeff = float(val)

# 创建Tkinter GUI
root = tk.Tk()
root.title("Comb Filter Parameter Adjustment")

# 延迟时间滑块
delay_label = tk.Label(root, text="Delay Time (seconds)")
delay_label.pack()
delay_slider = ttk.Scale(root, from_=0.01, to=0.1, orient="horizontal", command=update_delay_time)
delay_slider.set(default_delay_time)
delay_slider.pack()

# 反馈系数滑块
feedback_label = tk.Label(root, text="Feedback Coefficient")
feedback_label.pack()
feedback_slider = ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", command=update_feedback_coeff)
feedback_slider.set(default_feedback_coeff)
feedback_slider.pack()

# 停止程序按钮
def stop_program():
    global stop_flag
    stop_flag = True
    root.destroy()

stop_button = tk.Button(root, text="Stop", command=stop_program)
stop_button.pack()

# 运行Tkinter主循环
root.mainloop()

# 等待线程结束
player_thread.join()
