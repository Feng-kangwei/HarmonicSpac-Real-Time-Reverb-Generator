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

# 默认参数
default_delay_time = 0.02  # 梳状滤波器默认延迟时间（秒）
default_feedback_coeff = 0.8  # 梳状滤波器默认反馈系数
default_allpass_coeff = 0.5  # 全通系统默认系数

# 创建实时调节的全局变量
current_delay_time = default_delay_time
current_feedback_coeff = default_feedback_coeff
current_allpass_coeff = default_allpass_coeff

# 梳状滤波器实现
def feedback_comb_filter(x, delay_samples, feedback_coefficient):
    y = np.zeros_like(x)
    for n in range(len(x)):
        if n < delay_samples:
            y[n] = x[n]
        else:
            y[n] = x[n] + feedback_coefficient * y[n - delay_samples]
    return y

# 全通系统实现
def allpass_filter(x, a):
    """
    一阶全通滤波器
    参数：
        x: 输入信号（numpy 数组）
        a: 全通系数，|a| < 1
    返回：
        y: 滤波后的输出信号
    """
    y = np.zeros_like(x)
    for n in range(1, len(x)):
        y[n] = -a * x[n] + x[n-1] + a * y[n-1]
    return y

# 实时音频播放线程
stop_flag = False

def audio_player():
    global stop_flag, current_delay_time, current_feedback_coeff, current_allpass_coeff

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)

    while not stop_flag:
        # 动态计算梳状滤波器参数
        delay_samples = int(current_delay_time * fs)
        # 应用梳状滤波器
        comb_filtered_signal = feedback_comb_filter(audio_data, delay_samples, current_feedback_coeff)
        # 应用全通滤波器
        allpass_filtered_signal = allpass_filter(comb_filtered_signal, current_allpass_coeff)
        # 播放音频
        stream.write(allpass_filtered_signal.astype(np.float32).tobytes())

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

def update_allpass_coeff(val):
    global current_allpass_coeff
    current_allpass_coeff = float(val)

# 创建Tkinter GUI
root = tk.Tk()
root.title("Comb Filter and Allpass System Adjustment")

# 梳状滤波器参数 - 延迟时间滑块
delay_label = tk.Label(root, text="Comb Filter - Delay Time (seconds)")
delay_label.pack()
delay_slider = ttk.Scale(root, from_=0.01, to=0.1, orient="horizontal", command=update_delay_time)
delay_slider.set(default_delay_time)
delay_slider.pack()

# 梳状滤波器参数 - 反馈系数滑块
feedback_label = tk.Label(root, text="Comb Filter - Feedback Coefficient")
feedback_label.pack()
feedback_slider = ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", command=update_feedback_coeff)
feedback_slider.set(default_feedback_coeff)
feedback_slider.pack()

# 全通系统参数 - 系数滑块
allpass_label = tk.Label(root, text="Allpass Filter - Coefficient")
allpass_label.pack()
allpass_slider = ttk.Scale(root, from_=-0.99, to=0.99, orient="horizontal", command=update_allpass_coeff)
allpass_slider.set(default_allpass_coeff)
allpass_slider.pack()

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
