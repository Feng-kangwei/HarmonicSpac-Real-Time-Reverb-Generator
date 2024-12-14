import numpy as np
import pyaudio
import scipy.signal as ss
import rir_generator as rir
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk
from tkinter.filedialog import askopenfilename
import wave
from scipy.signal import butter, lfilter
import struct
import soundfile as sf
import shutil

"""
整合功能说明：
- 当用户按下Record后开始录制音频，并实时在输入信号图中显示录音波形及其频域图。
- 当用户按下Stop Record后停止录音，将录音保存为一个临时文件，然后通过add_room_reverb_file函数处理成output.wav。
- 用户按下Play后，使用与之前上传文件后的逻辑一致，对output.wav进行播放并在图中显示相应的时域和频域数据。
"""

input_wf = None
output_wf = None
playing = False
recording = False
recorded_frames = []
record_input_stream = None
recorded_filename = "recorded_temp.wav"
output_file_path = "output.wav"

OPRATE = 8000
OPBLOCKLEN=600
OPWIDTH = 2  # 假设为16-bit音频，2 bytes per sample
OPCHANNELS = 1  # 或2, 根据你的音频是单声道还是双声道



room_dim = [5.0, 4.0, 6.0]
source_pos = [2.0, 3.5, 2.0]
receiver_pos = [2.0, 1.5, 2.0]
rt60 = 0.4
cutoff=10

def highpass_filter(data, cutoff=10, fs=16000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data

def add_room_reverb(input_signal, fs, room_dim=[5.0, 4.0, 6.0], source_pos=[2.0, 3.5, 2.0], receiver_pos=[2.0, 1.5, 2.0], rt60=0.4):
    h = rir.generate(
        c=340,
        fs=fs,
        r=[receiver_pos],
        s=source_pos,
        L=room_dim,
        reverberation_time=rt60,
        nsample=4096
    )
    h = h.reshape(-1, 1)
    reverbed = ss.convolve(input_signal, h[:, 0], mode='same')
    max_val = np.max(np.abs(reverbed))
    if max_val > 1e-6:
        reverbed = reverbed / max_val
    else:
        reverbed = np.zeros_like(reverbed)
    return reverbed

def add_room_reverb_file(audio_path, save_path, room_dim=[5, 4, 6], source_pos=[2, 3.5, 2], receiver_pos=[2, 1.5, 2], rt60=0.4):
    signal, fs = sf.read(audio_path, always_2d=True)
    h = rir.generate(
        c=340,
        fs=fs,
        r=[receiver_pos],
        s=source_pos,
        L=room_dim,
        reverberation_time=rt60,
        nsample=4096
    )
    h = h.reshape(-1, 1)
    reverbed = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        reverbed[:, i] = ss.convolve(signal[:, i], h[:, 0], mode='same')
    reverbed = reverbed / np.max(np.abs(reverbed))
    sf.write(save_path, reverbed, fs)
    return reverbed, fs

def reprocess_audio():
    global input_wf, output_wf
    if file_path is not None:
        if output_wf is not None:
            output_wf.close()
        add_room_reverb_file(file_path, output_file_path, room_dim, source_pos, receiver_pos, rt60)
        output_wf = wave.open(output_file_path, 'rb')
        output_wf.rewind()

def update_plot(frame):
    if not playing and not recording:
        return line_input_time, line_input_freq, line_output_time, line_output_freq

    # 录音时显示输入波形
    if recording and record_input_stream is not None:
        input_bytes = record_input_stream.read(OPBLOCKLEN, exception_on_overflow=False)
        recorded_frames.append(input_bytes)
        input_signal_block = np.frombuffer(input_bytes, dtype=np.int16)
        line_input_time.set_data(range(OPBLOCKLEN), input_signal_block)
        input_freq_data = np.abs(np.fft.rfft(input_signal_block)) / OPBLOCKLEN
        input_freqs = np.fft.rfftfreq(OPBLOCKLEN, d=1.0 / OPRATE)
        line_input_freq.set_data(input_freqs, input_freq_data)
        # 录音时没有输出音频在播放，可以不更新输出波形
        return line_input_time, line_input_freq, line_output_time, line_output_freq

    # 播放时数据更新逻辑
    if playing and output_wf is not None and input_wf is not None:
        input_bytes = input_wf.readframes(OPBLOCKLEN)
        if len(input_bytes) < OPBLOCKLEN*OPWIDTH:
            input_wf.rewind()
            input_bytes = input_wf.readframes(OPBLOCKLEN)
        input_signal_block = struct.unpack('h' * OPBLOCKLEN, input_bytes)
        line_input_time.set_data(range(OPBLOCKLEN), input_signal_block)
        input_freq_data = np.abs(np.fft.rfft(input_signal_block)) / OPBLOCKLEN
        input_freqs = np.fft.rfftfreq(OPBLOCKLEN, d=1.0 / OPRATE)
        line_input_freq.set_data(input_freqs, input_freq_data)

        output_bytes = output_wf.readframes(OPBLOCKLEN)
        if len(output_bytes) < OPBLOCKLEN*OPWIDTH:
            output_wf.rewind()
            output_bytes = output_wf.readframes(OPBLOCKLEN)
        output_signal_block = struct.unpack('h' * OPBLOCKLEN, output_bytes)
        stream.write(output_bytes, OPBLOCKLEN)
        output_freq_data = np.abs(np.fft.rfft(output_signal_block)) / OPBLOCKLEN
        output_freqs = np.fft.rfftfreq(OPBLOCKLEN, d=1.0 / OPRATE)
        line_output_time.set_data(range(OPBLOCKLEN), output_signal_block)
        line_output_freq.set_data(output_freqs, output_freq_data)

    return line_input_time, line_input_freq, line_output_time, line_output_freq

def update_cutoff(val):
    global cutoff
    cutoff=float(val)

def update_rt60(val):
    global rt60
    rt60 = float(val)
    reprocess_audio()

def update_room_dim_x(val):
    global room_dim
    room_dim[0] = float(val)
    x_label.config(text=f"X: {room_dim[0]:.2f}")
    reprocess_audio()

def update_room_dim_y(val):
    global room_dim
    room_dim[1] = float(val)
    y_label.config(text=f"Y: {room_dim[1]:.2f}")
    reprocess_audio()

def update_room_dim_z(val):
    global room_dim
    room_dim[2] = float(val)
    z_label.config(text=f"Z: {room_dim[2]:.2f}")
    reprocess_audio()

def update_source_pos_x(val):
    global source_pos
    source_pos[0] = float(val)
    sx_label.config(text=f"X: {source_pos[0]:.2f}")
    reprocess_audio()

def update_source_pos_y(val):
    global source_pos
    source_pos[1] = float(val)
    sy_label.config(text=f"Y: {source_pos[1]:.2f}")
    reprocess_audio()

def update_source_pos_z(val):
    global source_pos
    source_pos[2] = float(val)
    sz_label.config(text=f"Z: {source_pos[2]:.2f}")
    reprocess_audio()

def update_receiver_pos_x(val):
    global receiver_pos
    receiver_pos[0] = float(val)
    rx_label.config(text=f"X: {receiver_pos[0]:.2f}")
    reprocess_audio()

def update_receiver_pos_y(val):
    global receiver_pos
    receiver_pos[1] = float(val)
    ry_label.config(text=f"Y: {receiver_pos[1]:.2f}")
    reprocess_audio()

def update_receiver_pos_z(val):
    global receiver_pos
    receiver_pos[2] = float(val)
    rz_label.config(text=f"Z: {receiver_pos[2]:.2f}")
    reprocess_audio()

def toggle_signal_source():
    if use_real_time.get():
        use_local_file.set(False)
        use_record.set(False)
    elif use_local_file.get():
        use_real_time.set(False)
        use_record.set(False)
    elif use_record.get():
        use_real_time.set(False)
        use_local_file.set(False)
    elif not use_real_time.get() and not use_local_file.get() and not use_record.get():
        pass

def upload_file():
    global file_path
    file_path = askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        use_local_file.set(False)
        use_real_time.set(False)
        process(file_path,output_file_path)

def process(file_path,output_file_path):
    global OPRATE ,OPWIDTH, OPLEN, OPCHANNELS, OPBLOCKLEN
    global input_wf,output_wf
    global p,stream,playing
    input_wf = wave.open(file_path, 'rb')
    add_room_reverb_file(file_path,output_file_path,room_dim, source_pos, receiver_pos, rt60)
    output_wf = wave.open(output_file_path, 'rb')
    OPRATE        = output_wf.getframerate()
    OPWIDTH       = output_wf.getsampwidth()
    OPLEN         = output_wf.getnframes()
    OPCHANNELS    = output_wf.getnchannels()
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(OPWIDTH),
        channels=OPCHANNELS,
        rate=OPRATE,
        input=True,
        output=True,
        frames_per_buffer=OPBLOCKLEN
    )
    playing = True

def start_recording():
    print("entered finish")
    global recording, record_input_stream, recorded_frames, p, OPCHANNELS, OPRATE, OPWIDTH, OPBLOCKLEN
    if not recording:
        recording = True
        recorded_frames = []
        record_input_format = p.get_format_from_width(OPWIDTH)
        record_input_stream = p.open(
            format=record_input_format,
            channels=OPCHANNELS,
            rate=OPRATE,
            input=True,
            output=False,
            frames_per_buffer=OPBLOCKLEN
        )
        print("Recording started.")

def stop_recording():
    global recording, record_input_stream, recorded_frames, recorded_filename
    if recording:
        recording = False
        record_input_stream.stop_stream()
        record_input_stream.close()
        record_input_stream = None
        # 保存录音为临时wav文件
        wf = wave.open(recorded_filename, 'wb')
        wf.setnchannels(OPCHANNELS)
        wf.setsampwidth(OPWIDTH)
        wf.setframerate(OPRATE)
        wf.writeframes(b''.join(recorded_frames))
        wf.close()
        print(f"Recording saved to {recorded_filename}")

        # 对录音文件进行混响处理并保存为output.wav
        process(recorded_filename, output_file_path)
        print(f"Processed recorded file saved to {output_file_path}")

def stop_play():
    global playing, stream
    if playing:
        playing = False
        stream.stop_stream()
        output_wf.rewind()
    print("Play stopped.")

    # 准备全零数组，比如长度为OPBLOCKLEN
    zeros_block = [0]*OPBLOCKLEN

    # 将输入、输出时域与频域曲线更新为全0数据
    line_input_time.set_data(range(OPBLOCKLEN), zeros_block)
    line_input_freq.set_data(range(OPBLOCKLEN//2+1), [0]*(OPBLOCKLEN//2+1))  # 频域点数量为OPBLOCKLEN//2+1
    line_output_time.set_data(range(OPBLOCKLEN), zeros_block)
    line_output_freq.set_data(range(OPBLOCKLEN//2+1), [0]*(OPBLOCKLEN//2+1))

    # 刷新图表
    canvas.draw()

def play():
    global playing, stream, output_wf
    if not playing:
        # 播放output.wav (可认为是刚录制或刚upload处理好的文件)
        # 注：如果想播放录制后的结果，不需要特殊处理，因为stop_recording已经生成了output.wav
        if output_wf is None:
            # 如果此前没有文件，则假设刚录下的文件即为output.wav
            process(output_file_path, output_file_path)
        else:
            if stream.is_stopped():
                stream.start_stream()
        playing = True
    print("Play started/resumed.")

def reset():
    print("Reset button pressed.")

def quit_application():
    root.destroy()



# GUI部分
root = tk.Tk()
root.title("Real-Time Reverb with Visualization")
root.geometry("1100x700")
use_real_time = tk.BooleanVar(value=False)
use_local_file = tk.BooleanVar(value=False)
use_record = tk.BooleanVar(value=False)
local_file_path = None
file_path = None  # 在最上方全局定义


control_frame = tk.Frame(root, highlightthickness=1, highlightbackground="black")
control_frame.pack(side=tk.LEFT, fill=tk.Y)

cutoff_label = tk.Label(control_frame, text="Cut off frequency")
cutoff_label.pack()
cutoff_value_lable=tk.Label(control_frame,text=f"cutoff: {cutoff:.2f}")
cutoff_slider = ttk.Scale(control_frame, from_=100, to=8000, orient="horizontal", command=update_cutoff)
cutoff_slider.set(cutoff)
cutoff_slider.pack()


rt60_label = tk.Label(control_frame, text="Reverb Time (RT60)")
rt60_label.pack()
rt60_value_lable=tk.Label(control_frame,text=f"rt60: {rt60:.2f}")
rt60_slider = ttk.Scale(control_frame, from_=0.4, to=2.0, orient="horizontal", command=update_rt60)
rt60_slider.set(rt60)
rt60_slider.pack()

room_frame = tk.Frame(control_frame)
room_frame.pack(fill=tk.X)
room_label = tk.Label(room_frame, text="Room Dimensions")
room_label.pack()
x_frame = tk.Frame(room_frame)
x_frame.pack(side=tk.LEFT, padx=5)
x_label = tk.Label(x_frame, text=f"X: {room_dim[0]:.2f}")
x_label.pack()
x_slider = ttk.Scale(x_frame, from_=1.0, to=10.0, orient="vertical", command=update_room_dim_x)
x_slider.set(room_dim[0])
x_slider.pack()

y_frame = tk.Frame(room_frame)
y_frame.pack(side=tk.LEFT, padx=5)
y_label = tk.Label(y_frame, text=f"Y: {room_dim[1]:.2f}")
y_label.pack()
y_slider = ttk.Scale(y_frame, from_=2.0, to=10.0, orient="vertical", command=update_room_dim_y)
y_slider.set(room_dim[1])
y_slider.pack()

z_frame = tk.Frame(room_frame)
z_frame.pack(side=tk.LEFT, padx=5)
z_label = tk.Label(z_frame, text=f"Z: {room_dim[2]:.2f}")
z_label.pack()
z_slider = ttk.Scale(z_frame, from_=1.0, to=10.0, orient="vertical", command=update_room_dim_z)
z_slider.set(room_dim[2])
z_slider.pack()

source_frame = tk.Frame(control_frame)
source_frame.pack()
source_label = tk.Label(source_frame, text="Source Position")
source_label.pack()
sx_frame=tk.Frame(source_frame)
sx_frame.pack(side=tk.LEFT, padx=5)
sx_label = tk.Label(sx_frame, text=f"X: {source_pos[0]:.2f}")
sx_label.pack()
sx_slider = ttk.Scale(sx_frame, from_=1.0, to=10.0, orient="vertical", command=update_source_pos_x)
sx_slider.set(source_pos[0])
sx_slider.pack()

sy_frame=tk.Frame(source_frame)
sy_frame.pack(side=tk.LEFT, padx=5)
sy_label = tk.Label(sy_frame, text=f"Y: {source_pos[1]:.2f}")
sy_label.pack()
sy_slider = ttk.Scale(sy_frame, from_=1.0, to=10.0, orient="vertical", command=update_source_pos_y)
sy_slider.set(source_pos[1])
sy_slider.pack()

sz_frame=tk.Frame(source_frame)
sz_frame.pack(side=tk.LEFT, padx=5)
sz_label = tk.Label(sz_frame, text=f"Z: {source_pos[2]:.2f}")
sz_label.pack()
sz_slider = ttk.Scale(sz_frame, from_=1.0, to=10.0, orient="vertical", command=update_source_pos_z)
sz_slider.set(source_pos[2])
sz_slider.pack()

receiver_frame = tk.Frame(control_frame)
receiver_frame.pack()
receiver_label = tk.Label(receiver_frame, text="Receiver Position")
receiver_label.pack()
rx_frame=tk.Frame(receiver_frame)
rx_frame.pack(side=tk.LEFT, padx=5)
rx_label = tk.Label(rx_frame, text=f"X: {receiver_pos[0]:.2f}")
rx_label.pack()
rx_slider = ttk.Scale(rx_frame, from_=1.0, to=10.0, orient="vertical", command=update_receiver_pos_x)
rx_slider.set(receiver_pos[0])
rx_slider.pack()

ry_frame=tk.Frame(receiver_frame)
ry_frame.pack(side=tk.LEFT, padx=5)
ry_label = tk.Label(ry_frame, text=f"Y: {receiver_pos[1]:.2f}")
ry_label.pack()
ry_slider = ttk.Scale(ry_frame, from_=1.0, to=10.0, orient="vertical", command=update_receiver_pos_y)
ry_slider.set(receiver_pos[1])
ry_slider.pack()

rz_frame=tk.Frame(receiver_frame)
rz_frame.pack(side=tk.LEFT, padx=5)
rz_label = tk.Label(rz_frame, text=f"Z: {receiver_pos[2]:.2f}")
rz_label.pack()
rz_slider = ttk.Scale(rz_frame, from_=1.0, to=10.0, orient="vertical", command=update_receiver_pos_z)
rz_slider.set(receiver_pos[2])
rz_slider.pack()

button_frame = tk.Frame(control_frame)
button_frame.pack()
signal_frame = tk.Frame(control_frame, highlightthickness=1, highlightbackground="black")
signal_frame.pack(pady=10)

real_time_check = tk.Checkbutton(signal_frame, text="Real Time", variable=use_real_time, command=toggle_signal_source)
real_time_check.pack(side=tk.LEFT)
local_file_check = tk.Checkbutton(signal_frame, text="Local File", variable=use_local_file, command=toggle_signal_source)
local_file_check.pack(side=tk.LEFT)
record_check = tk.Checkbutton(signal_frame, text="Record", variable=use_record, command=toggle_signal_source)
record_check.pack(side=tk.LEFT)

line1_frame = tk.Frame(button_frame)
line1_frame.pack(side=tk.TOP, fill=tk.X)
line2_frame = tk.Frame(button_frame)
line2_frame.pack(side=tk.TOP, fill=tk.X)
line2_subframe = tk.Frame(line2_frame)
line2_subframe.pack(expand=True, anchor='center')

upload_button = tk.Button(line1_frame,text="upload file",command=upload_file)
upload_button.pack(side=tk.LEFT, padx=5, pady=5)
record_button = tk.Button(line1_frame,text="Record",command=start_recording)
record_button.pack(side=tk.LEFT, padx=5, pady=5)
stop_button=tk.Button(line1_frame,text="Stop Record",command=stop_recording)
stop_button.pack(side=tk.LEFT,padx=5,pady=5)

reset_button = tk.Button(line2_subframe, text="Reset", command=reset)
reset_button.pack(side=tk.LEFT, padx=5, pady=5)
quit_button = tk.Button(line2_subframe, text="Quit", command=quit_application)
quit_button.pack(side=tk.LEFT, padx=5, pady=5)
play_button = tk.Button(line2_subframe, text="Play", command=play)
play_button.pack(side=tk.LEFT, padx=5, pady=5)
stop_play_button = tk.Button(line2_subframe, text="Stop Play", command=stop_play)
stop_play_button.pack(side=tk.LEFT, padx=5, pady=5)

chart_frame = tk.Frame(root,width=200)
chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
fig, axs = plt.subplots(2, 2, figsize=(3,2),dpi=50)
fig.tight_layout(pad=0.5)

ax_input_time = axs[0, 0]
line_input_time, = ax_input_time.plot([], [], lw=2, label="Input Signal (Time)")
ax_input_time.set_xlim(0, 100)
ax_input_time.set_ylim(-20000, 20000)
ax_input_time.set_title("Input Signal (Time Domain)")
ax_input_time.set_xlabel("Sample Index")
ax_input_time.set_ylabel("Amplitude")
ax_input_time.legend()

ax_input_freq = axs[0, 1]
line_input_freq, = ax_input_freq.plot([], [], lw=2, label="Input Signal (Freq)")
ax_input_freq.set_xlim(0, OPRATE // 2)
ax_input_freq.set_ylim(0, 2000)
ax_input_freq.set_title("Input Signal (Frequency Domain)")
ax_input_freq.set_xlabel("Frequency (Hz)")
ax_input_freq.set_ylabel("Magnitude")
ax_input_freq.legend()

ax_output_time = axs[1, 0]
line_output_time, = ax_output_time.plot([], [], lw=2, label="Reverb Signal (Time)")
ax_output_time.set_xlim(0, 100)
ax_output_time.set_ylim(-20000, 20000)
ax_output_time.set_title("Reverb Signal (Time Domain)")
ax_output_time.set_xlabel("Sample Index")
ax_output_time.set_ylabel("Amplitude")
ax_output_time.legend()

ax_output_freq = axs[1, 1]
line_output_freq, = ax_output_freq.plot([], [], lw=2, label="Reverb Signal (Freq)")
ax_output_freq.set_xlim(0, OPRATE // 2)
ax_output_freq.set_ylim(0, 2000)
ax_output_freq.set_title("Reverb Signal (Frequency Domain)")
ax_output_freq.set_xlabel("Frequency (Hz)")
ax_output_freq.set_ylabel("Magnitude")
ax_output_freq.legend()

canvas = FigureCanvasTkAgg(fig, master=chart_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

ani = FuncAnimation(fig, update_plot, interval=15, blit=True)
canvas.draw()

p = pyaudio.PyAudio()
file_path = None

try:
    root.mainloop()
except KeyboardInterrupt:
    print("Real-time reverb stopped.")
finally:
    if 'stream' in globals():
        stream.stop_stream()
        stream.close()
    p.terminate()
