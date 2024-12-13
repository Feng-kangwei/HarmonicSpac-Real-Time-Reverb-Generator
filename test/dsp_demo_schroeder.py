import numpy as np
import pyaudio
import wave
import tkinter as tk
from tkinter import ttk
import threading

# 读取 WAV 文件
input_wav_file = "author.wav"  # 替换为您的 WAV 文件路径
wav = wave.open(input_wav_file, 'rb')
fs = wav.getframerate()
n_channels = wav.getnchannels()
frames = wav.readframes(wav.getnframes())
wav.close()

# 转换音频数据为 NumPy 数组
audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
audio_data /= np.max(np.abs(audio_data))  # 归一化到 [-1, 1]
if n_channels > 1:
    audio_data = audio_data[::n_channels]  # 转为单声道

# 默认参数
default_comb_feedback = 0.8
default_allpass_feedback = 0.7
default_comb_delay = [0.0297, 0.0371, 0.0411, 0.0437]
default_allpass_delay = 0.005

# 全局变量用于实时调整
current_comb_feedback = default_comb_feedback
current_allpass_feedback = default_allpass_feedback
current_comb_delay = default_comb_delay
current_allpass_delay = default_allpass_delay

# 梳状滤波器实现
def feedback_comb_filter(x, delay_samples, feedback_coeff):
    y = np.zeros_like(x)
    buffer = np.zeros(delay_samples)
    for n in range(len(x)):
        delayed_value = buffer[-1]
        buffer[1:] = buffer[:-1]  # 移动缓冲区
        buffer[0] = x[n] + feedback_coeff * delayed_value
        y[n] = delayed_value
    return y

# 全通滤波器实现
def allpass_filter(x, delay_samples, feedback_coeff):
    y = np.zeros_like(x)
    buffer = np.zeros(delay_samples)
    for n in range(len(x)):
        delayed_value = buffer[-1]
        buffer[1:] = buffer[:-1]
        buffer[0] = x[n] + feedback_coeff * delayed_value
        y[n] = delayed_value - feedback_coeff * x[n]
    return y

# 混响算法实现
def reverb(audio, fs):
    global current_comb_feedback, current_allpass_feedback, current_comb_delay, current_allpass_delay
    # 多个梳状滤波器的并联
    comb_output = np.zeros_like(audio)
    for delay_time in current_comb_delay:
        delay_samples = int(delay_time * fs)
        comb_output += feedback_comb_filter(audio, delay_samples, current_comb_feedback)
    
    # 单个全通滤波器
    delay_samples = int(current_allpass_delay * fs)
    reverb_output = allpass_filter(comb_output, delay_samples, current_allpass_feedback)
    return reverb_output

# 实时音频播放线程
stop_flag = False

def audio_player():
    global stop_flag
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paFloat32, channels=1, rate=fs, output=True)
    while not stop_flag:
        reverb_audio = reverb(audio_data, fs)
        stream.write(reverb_audio.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()

# 启动音频播放线程
player_thread = threading.Thread(target=audio_player)
player_thread.start()

# 更新滑块值的回调函数
def update_comb_feedback(val):
    global current_comb_feedback
    current_comb_feedback = float(val)

def update_allpass_feedback(val):
    global current_allpass_feedback
    current_allpass_feedback = float(val)

def update_comb_delay(val):
    global current_comb_delay
    delay_value = float(val)
    current_comb_delay = [delay_value, delay_value + 0.01, delay_value + 0.02, delay_value + 0.03]

def update_allpass_delay(val):
    global current_allpass_delay
    current_allpass_delay = float(val)

# 创建 Tkinter GUI
root = tk.Tk()
root.title("Reverb Parameter Adjustment")

# 梳状滤波器反馈系数滑块
comb_feedback_label = tk.Label(root, text="Comb Filter Feedback Coefficient")
comb_feedback_label.pack()
comb_feedback_slider = ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", command=update_comb_feedback)
comb_feedback_slider.set(default_comb_feedback)
comb_feedback_slider.pack()

# 全通滤波器反馈系数滑块
allpass_feedback_label = tk.Label(root, text="Allpass Filter Feedback Coefficient")
allpass_feedback_label.pack()
allpass_feedback_slider = ttk.Scale(root, from_=0.0, to=1.0, orient="horizontal", command=update_allpass_feedback)
allpass_feedback_slider.set(default_allpass_feedback)
allpass_feedback_slider.pack()

# 梳状滤波器延迟时间滑块
comb_delay_label = tk.Label(root, text="Comb Filter Delay Time (seconds)")
comb_delay_label.pack()
comb_delay_slider = ttk.Scale(root, from_=0.01, to=0.1, orient="horizontal", command=update_comb_delay)
comb_delay_slider.set(default_comb_delay[0])
comb_delay_slider.pack()

# 全通滤波器延迟时间滑块
allpass_delay_label = tk.Label(root, text="Allpass Filter Delay Time (seconds)")
allpass_delay_label.pack()
allpass_delay_slider = ttk.Scale(root, from_=0.001, to=0.02, orient="horizontal", command=update_allpass_delay)
allpass_delay_slider.set(default_allpass_delay)
allpass_delay_slider.pack()

# 停止按钮
def stop_program():
    global stop_flag
    stop_flag = True
    root.destroy()

stop_button = tk.Button(root, text="Stop", command=stop_program)
stop_button.pack()

# 启动 GUI 主循环
root.mainloop()

# 等待线程结束
#player_thread.join()

