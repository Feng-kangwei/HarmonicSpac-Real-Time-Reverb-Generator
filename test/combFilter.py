import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.fft import fft, fftfreq

# 创建脉冲信号
def generate_impulse_signal(length, sample_rate=44100):
    impulse = np.zeros(length)
    impulse[0] = 1  # 单位脉冲
    t = np.linspace(0, length / sample_rate, length, endpoint=False)
    return t, impulse

# 组合滤波器
def comb_filter(input_signal, delay, feedback, sample_rate=44100):
    delay_samples = int(delay * sample_rate)
    buffer = np.zeros(delay_samples)
    output_signal = np.zeros_like(input_signal)
    for n in range(len(input_signal)):
        delayed_sample = buffer[-1]
        buffer[1:] = buffer[:-1]
        buffer[0] = input_signal[n] + feedback * delayed_sample
        output_signal[n] = input_signal[n] + delayed_sample
    return output_signal

# 全通滤波器
def all_pass_filter(input_signal, delay, feedback, sample_rate=44100):
    delay_samples = int(delay * sample_rate)
    buffer = np.zeros(delay_samples)
    output_signal = np.zeros_like(input_signal)
    for n in range(len(input_signal)):
        delayed_sample = buffer[-1]
        buffer[1:] = buffer[:-1]
        buffer[0] = input_signal[n] + feedback * delayed_sample
        output_signal[n] = delayed_sample - feedback * input_signal[n]
    return output_signal

# 生成脉冲信号
sample_rate = 44100
signal_length = 44100  # 1 秒信号
t, input_signal = generate_impulse_signal(signal_length, sample_rate)

# 默认参数
default_delay = 0.02  # 默认延迟时间（秒）
default_comb_feedback = 0.5  # Comb Filter 默认反馈系数
default_allpass_feedback = 0.5  # All-Pass Filter 默认反馈系数

# 初始信号处理
comb_output_signal = comb_filter(input_signal, default_delay, default_comb_feedback, sample_rate)
allpass_output_signal = all_pass_filter(comb_output_signal, default_delay, default_allpass_feedback, sample_rate)

# 创建 GUI
fig, ax = plt.subplots(3, 2, figsize=(12, 10))
plt.subplots_adjust(left=0.25, bottom=0.3)

# 输入信号 - 时域图
ax[0, 0].plot(t[:1000], input_signal[:1000], label="Input Signal", color="blue")
ax[0, 0].set_title("Input Signal (Time Domain)")
ax[0, 0].set_xlabel("Time (s)")
ax[0, 0].legend()

# 输入信号 - 频域图
freqs = fftfreq(signal_length, 1 / sample_rate)
input_fft = np.abs(fft(input_signal))
ax[0, 1].plot(freqs[:signal_length // 2], input_fft[:signal_length // 2], color="blue")
ax[0, 1].set_title("Input Signal (Frequency Domain)")
ax[0, 1].set_xlabel("Frequency (Hz)")

# 组合滤波器 - 时域图
comb_output_plot, = ax[1, 0].plot(t[:1000], comb_output_signal[:1000], label="Comb Filter Output", color="red")
ax[1, 0].set_title("Comb Filter Output (Time Domain)")
ax[1, 0].set_xlabel("Time (s)")
ax[1, 0].legend()

# 组合滤波器 - 频域图
comb_output_fft = np.abs(fft(comb_output_signal))
comb_output_freq_plot, = ax[1, 1].plot(freqs[:signal_length // 2], comb_output_fft[:signal_length // 2], color="red")
ax[1, 1].set_title("Comb Filter Output (Frequency Domain)")
ax[1, 1].set_xlabel("Frequency (Hz)")

# 全通滤波器 - 时域图
allpass_output_plot, = ax[2, 0].plot(t[:1000], allpass_output_signal[:1000], label="All-Pass Filter Output", color="green")
ax[2, 0].set_title("All-Pass Filter Output (Time Domain)")
ax[2, 0].set_xlabel("Time (s)")
ax[2, 0].legend()

# 全通滤波器 - 频域图
allpass_output_fft = np.abs(fft(allpass_output_signal))
allpass_output_freq_plot, = ax[2, 1].plot(freqs[:signal_length // 2], allpass_output_fft[:signal_length // 2], color="green")
ax[2, 1].set_title("All-Pass Filter Output (Frequency Domain)")
ax[2, 1].set_xlabel("Frequency (Hz)")

# 添加滑块
ax_comb_delay = plt.axes([0.25, 0.2, 0.65, 0.03], facecolor="lightgray")
ax_comb_feedback = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor="lightgray")
ax_allpass_feedback = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor="lightgray")

slider_comb_delay = Slider(ax_comb_delay, "Comb Delay (s)", 0.001, 0.1, valinit=default_delay, valstep=0.001)
slider_comb_feedback = Slider(ax_comb_feedback, "Comb Feedback", 0.0, 0.99, valinit=default_comb_feedback, valstep=0.01)
slider_allpass_feedback = Slider(ax_allpass_feedback, "All-Pass Feedback", 0.0, 0.99, valinit=default_allpass_feedback, valstep=0.01)

# 滑块更新函数
def update(val):
    comb_delay = slider_comb_delay.val
    comb_feedback = slider_comb_feedback.val
    allpass_feedback = slider_allpass_feedback.val

    # 更新组合滤波器输出
    updated_comb_output = comb_filter(input_signal, comb_delay, comb_feedback, sample_rate)
    updated_comb_fft = np.abs(fft(updated_comb_output))
    comb_output_plot.set_ydata(updated_comb_output[:1000])
    comb_output_freq_plot.set_ydata(updated_comb_fft[:signal_length // 2])

    # 更新全通滤波器输出
    updated_allpass_output = all_pass_filter(updated_comb_output, comb_delay, allpass_feedback, sample_rate)
    updated_allpass_fft = np.abs(fft(updated_allpass_output))
    allpass_output_plot.set_ydata(updated_allpass_output[:1000])
    allpass_output_freq_plot.set_ydata(updated_allpass_fft[:signal_length // 2])

    fig.canvas.draw_idle()

slider_comb_delay.on_changed(update)
slider_comb_feedback.on_changed(update)
slider_allpass_feedback.on_changed(update)

plt.show()
