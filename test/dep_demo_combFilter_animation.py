import numpy as np
import pyaudio 
import tkinter as tk
from tkinter import ttk
import wave
import sounddevice as sd
import struct
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

#Algo
def feedback_comb_filter(x,delay_samples, feedback_coefficient):
    y=np.zeros_like(x)
    for n in range(len(x)):
        if n<delay_samples:
            y[n]=x[n]
        else:
            y[n]=x[n]+feedback_coefficient*y[n-delay_samples]
    return y

# GUI
def update_delay_time(val):
    global current_delay_time
    current_delay_time = float(val)

def update_feedback_coeff(val):
    global current_feedback_coeff
    current_feedback_coeff=float(val)

def stop_program():
    root.destroy()

def animation_dynamic():
    global signal_output, ani
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(8, 6))  # 创建两个子图
    fig.tight_layout(pad=3.0)

    # 时域图
    line_time, = ax_time.plot([], [], lw=2)
    ax_time.set_xlim(0, BLOCKLEN)
    ax_time.set_ylim(-1, 1)
    ax_time.set_title("Output Signal (Time Domain)")
    ax_time.set_xlabel("Sample Index")
    ax_time.set_ylabel("Amplitude")

    # 频域图
    line_freq, = ax_freq.plot([], [], lw=2)
    ax_freq.set_xlim(0, fs // 2)  # 频域范围为 0 到 Nyquist 频率
    ax_freq.set_ylim(0, 0.1)
    ax_freq.set_title("Output Signal (Frequency Domain)")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude")

    # 将 Matplotlib 图嵌入 Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    def update(frame):
        global signal_output
        delay_samples = int(current_delay_time * fs)
        signal_output = feedback_comb_filter(audio_data, delay_samples, current_feedback_coeff)

        # 时域更新
        start_idx = frame * BLOCKLEN % len(signal_output)
        end_idx = start_idx + BLOCKLEN
        if end_idx > len(signal_output):
            end_idx = len(signal_output)
            start_idx = end_idx - BLOCKLEN  # 确保窗口大小固定为 BLOCKLEN

        time_data = signal_output[start_idx:end_idx]
        line_time.set_data(range(BLOCKLEN), time_data)

        # 频域更新
        freq_data = np.abs(np.fft.rfft(time_data)) / BLOCKLEN  # 快速傅里叶变换（FFT）
        freqs = np.fft.rfftfreq(BLOCKLEN, d=1.0 / fs)  # 频率范围
        line_freq.set_data(freqs, freq_data)

        return line_time, line_freq

    ani = FuncAnimation(fig, update, interval=100, blit=True)
    canvas.draw()


def animation_allData():
    global signal_output, ani
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # 创建2x2的子图
    fig.tight_layout(pad=3.0)

    # 输入信号时域图
    ax_input_time = axs[0, 0]
    line_input_time, = ax_input_time.plot([], [], lw=2, label="Input Signal")
    ax_input_time.set_xlim(0, len(audio_data))
    ax_input_time.set_ylim(-1, 1)
    ax_input_time.set_title("Input Signal (Time Domain)")
    ax_input_time.set_xlabel("Sample Index")
    ax_input_time.set_ylabel("Amplitude")
    ax_input_time.legend()

    # 输入信号频域图
    ax_input_freq = axs[0, 1]
    line_input_freq, = ax_input_freq.plot([], [], lw=2, label="Input Spectrum")
    ax_input_freq.set_xlim(0, fs // 2)
    ax_input_freq.set_ylim(0, 0.1)
    ax_input_freq.set_title("Input Signal (Frequency Domain)")
    ax_input_freq.set_xlabel("Frequency (Hz)")
    ax_input_freq.set_ylabel("Magnitude")
    ax_input_freq.legend()

    # 输出信号时域图
    ax_output_time = axs[1, 0]
    line_output_time, = ax_output_time.plot([], [], lw=2, label="Output Signal")
    ax_output_time.set_xlim(0, len(audio_data))
    ax_output_time.set_ylim(-1, 1)
    ax_output_time.set_title("Output Signal (Time Domain)")
    ax_output_time.set_xlabel("Sample Index")
    ax_output_time.set_ylabel("Amplitude")
    ax_output_time.legend()

    # 输出信号频域图
    ax_output_freq = axs[1, 1]
    line_output_freq, = ax_output_freq.plot([], [], lw=2, label="Output Spectrum")
    ax_output_freq.set_xlim(0, fs // 2)
    ax_output_freq.set_ylim(0, 0.1)
    ax_output_freq.set_title("Output Signal (Frequency Domain)")
    ax_output_freq.set_xlabel("Frequency (Hz)")
    ax_output_freq.set_ylabel("Magnitude")
    ax_output_freq.legend()

    # 将 Matplotlib 图嵌入 Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    def update(frame):
        global signal_output
        delay_samples = int(current_delay_time * fs)  # 根据滑块计算延迟样本数
        signal_output = feedback_comb_filter(audio_data, delay_samples, current_feedback_coeff)  # 滤波

        # 输入信号时域
        line_input_time.set_data(np.arange(len(audio_data)), audio_data)

        # 输入信号频域
        input_freq_data = np.abs(np.fft.rfft(audio_data)) / len(audio_data)
        input_freqs = np.fft.rfftfreq(len(audio_data), d=1.0 / fs)
        line_input_freq.set_data(input_freqs, input_freq_data)

        # 输出信号时域
        line_output_time.set_data(np.arange(len(signal_output)), signal_output)

        # 输出信号频域
        output_freq_data = np.abs(np.fft.rfft(signal_output)) / len(signal_output)
        output_freqs = np.fft.rfftfreq(len(signal_output), d=1.0 / fs)
        line_output_freq.set_data(output_freqs, output_freq_data)

        return line_input_time, line_input_freq, line_output_time, line_output_freq

    ani = FuncAnimation(fig, update, interval=100, blit=True)
    canvas.draw()

'''
def animation_allData():
    global signal_output, ani
    fig, (ax_time, ax_freq) = plt.subplots(2, 1, figsize=(8, 6))  # 创建两个子图
    fig.tight_layout(pad=3.0)

    # 时域图
    line_time, = ax_time.plot([], [], lw=2)
    ax_time.set_xlim(0, len(audio_data))
    ax_time.set_ylim(-1, 1)
    ax_time.set_title("Output Signal (Time Domain)")
    ax_time.set_xlabel("Sample Index")
    ax_time.set_ylabel("Amplitude")

    # 频域图
    line_freq, = ax_freq.plot([], [], lw=2)
    ax_freq.set_xlim(0, fs // 2)  # 频域范围为 0 到 Nyquist 频率
    ax_freq.set_ylim(0, 0.1)
    ax_freq.set_title("Output Signal (Frequency Domain)")
    ax_freq.set_xlabel("Frequency (Hz)")
    ax_freq.set_ylabel("Magnitude")

    # 将 Matplotlib 图嵌入 Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack()

    def update(frame):
        global signal_output
        delay_samples = int(current_delay_time * fs)  # 根据滑块计算延迟样本数
        signal_output = feedback_comb_filter(audio_data, delay_samples, current_feedback_coeff)  # 滤波
        line_time.set_data(np.arange(len(signal_output)), signal_output)  # 更新折线图数据
    
        # 频域更新
        freq_data = np.abs(np.fft.rfft(signal_output)) / len(signal_output)  # 快速傅里叶变换（FFT）
        freqs = np.fft.rfftfreq(len(signal_output), d=1.0 / fs)  # 频率范围
        line_freq.set_data(freqs, freq_data)

        return line_time, line_freq

    ani = FuncAnimation(fig, update, interval=100, blit=True)
    canvas.draw()
'''




if __name__=="__main__":

    input_wav_file="author.wav"
    wav=wave.open(input_wav_file,'rb')
    fs=wav.getframerate()
    n_channels=wav.getnchannels()
    sample_width=wav.getsampwidth()
    frames=wav.readframes(wav.getnframes())
    wav.close()

    BLOCKLEN=256

    audio_data=np.frombuffer(frames,dtype=np.int16).astype(np.float32)
    audio_data/=np.max(np.abs(audio_data))

    if n_channels>1:
        audio_data=audio_data[::n_channels]

    default_delay_time=0.02
    default_feedback_coeff=0.8

    current_delay_time =default_delay_time
    current_feedback_coeff =default_feedback_coeff

    #GUI
    root=tk.Tk()
    root.title("Comb Filter Parameter Adjustment")

    # delay slider
    delay_label=tk.Label(root,text="Delay Time(seconds)")
    delay_label.pack()
    delay_slider=ttk.Scale(root,from_=0.01,to=0.1,orient="horizontal",command=update_delay_time)
    delay_slider.set(default_delay_time)
    delay_slider.pack()

    #feedback slider
    feedback_label=tk.Label(root,text="Feedback Coefficient")
    feedback_label.pack()
    feedback_slider=ttk.Scale(root,from_=0.01,to=1.0,orient="horizontal",command=update_feedback_coeff)
    feedback_slider.set(default_feedback_coeff)
    feedback_slider.pack()

    visualization_button=tk.Button(root,text="Vitualization",command=animation_allData)
    visualization_button.pack()

    stop_button=tk.Button(root,text="Stop",command=stop_program)
    stop_button.pack()


    root.mainloop() 






