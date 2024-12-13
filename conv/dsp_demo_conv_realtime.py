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

"""
1.加入了process function
2.修改了一下绘图的逻辑

"""

input_wf = None
output_wf = None

def highpass_filter(data, cutoff=10, fs=16000, order=5):

    # 计算归一化截止频率
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = lfilter(b, a, data)

    return filtered_data

def add_room_reverb(input_signal, fs,
                   room_dim=[5.0, 4.0, 6.0],
                   source_pos=[2.0, 3.5, 2.0],
                   receiver_pos=[2.0, 1.5, 2.0],
                   rt60=0.4):
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

def add_room_reverb_file(audio_path, save_path, 
                   room_dim=[5, 4, 6],
                   source_pos=[2, 3.5, 2],
                   receiver_pos=[2, 1.5, 2],
                   rt60=0.4):
    """
    添加基于房间模型的混响效果
    
    参数:
    - audio_path: 输入音频文件路径
    - save_path: 输出音频文件路径
    - room_dim: 房间尺寸 [x y z] (米)
    - source_pos: 声源位置 [x y z] (米)
    - receiver_pos: 接收器位置 [x y z] (米)
    - rt60: 混响时间(秒)
    """
    # 加载音频文件
    signal, fs = sf.read(audio_path, always_2d=True)
    
    # 生成房间冲击响应
    h = rir.generate(
        c=340,                      # 声速 (m/s)
        fs=fs,                      # 采样率
        r=[receiver_pos],           # 接收器位置
        s=source_pos,               # 声源位置
        L=room_dim,                 # 房间尺寸
        reverberation_time=rt60,    # 混响时间
        nsample=4096                # 输出样本数
    )
    
    # 应用卷积
    # 确保h形状正确 (nsample, 1)
    h = h.reshape(-1, 1)
    
    # 对每个声道进行卷积
    reverbed = np.zeros_like(signal)
    for i in range(signal.shape[1]):
        reverbed[:, i] = ss.convolve(signal[:, i], h[:, 0], mode='same')
    
    # 归一化输出
    reverbed = reverbed / np.max(np.abs(reverbed))
    
    # 保存处理后的音频
    sf.write(save_path, reverbed, fs)
    
    return reverbed, fs

def update_plot(frame):
    # 没有文件上传的情况下 不绘图
    if input_wf is None or output_wf is None:
        return line_input_time, line_input_freq, line_output_time, line_output_freq
    
    input_bytes = input_wf.readframes(OPBLOCKLEN)
    # 实现循环播放音频
    if len(input_bytes) < OPBLOCKLEN*OPWIDTH:
        # 说明已到文件尾部，重新回到文件开头
        input_wf.rewind()
        input_bytes = input_wf.readframes(OPBLOCKLEN)
    input_signal_block = struct.unpack('h' * OPBLOCKLEN, input_bytes)
    # 更新时域图
    line_input_time.set_data(range(OPBLOCKLEN), input_signal_block)
    # 更新频域图
    input_freq_data = np.abs(np.fft.rfft(input_signal_block)) / OPBLOCKLEN
    input_freqs = np.fft.rfftfreq(OPBLOCKLEN, d=1.0 / OPRATE)
    line_input_freq.set_data(input_freqs, input_freq_data)

    output_bytes = output_wf.readframes(OPBLOCKLEN)

    if len(output_bytes) < OPBLOCKLEN*OPWIDTH:
        # 说明已到文件尾部，重新回到文件开头
        output_wf.rewind()
        output_bytes = output_wf.readframes(OPBLOCKLEN)
    output_signal_block = struct.unpack('h' * OPBLOCKLEN, output_bytes)
    # 更新时域
    line_output_time.set_data(range(OPBLOCKLEN), output_signal_block)
    # 更新频域
    output_freq_data = np.abs(np.fft.rfft(output_signal_block)) / OPBLOCKLEN
    output_freqs = np.fft.rfftfreq(OPBLOCKLEN, d=1.0 / OPRATE)
    line_output_freq.set_data(output_freqs, output_freq_data)

    stream.write(output_bytes, OPBLOCKLEN)
    return line_input_time, line_input_freq, line_output_time, line_output_freq

# GUI update func
def update_cutoff(val):
    global cutoff
    cutoff=float(val)

def update_rt60(val):
    global rt60
    rt60 = float(val)

def update_room_dim_x(val):
    global room_dim
    room_dim[0] = float(val)
    x_label.config(text=f"X: {room_dim[0]:.2f}")

def update_room_dim_y(val):
    global room_dim
    room_dim[1] = float(val)
    y_label.config(text=f"Y: {room_dim[1]:.2f}")

def update_room_dim_z(val):
    global room_dim
    room_dim[2] = float(val)
    z_label.config(text=f"Z: {room_dim[2]:.2f}")

def update_source_pos_x(val):
    global source_pos
    source_pos[0] = float(val)
    sx_label.config(text=f"X: {source_pos[0]:.2f}")

def update_source_pos_y(val):
    global source_pos
    source_pos[1] = float(val)
    sy_label.config(text=f"Y: {source_pos[1]:.2f}")

def update_source_pos_z(val):
    global source_pos
    source_pos[2] = float(val)
    sz_label.config(text=f"Z: {source_pos[2]:.2f}")

def update_receiver_pos_x(val):
    global receiver_pos
    receiver_pos[0] = float(val)
    rx_label.config(text=f"X: {receiver_pos[0]:.2f}")

def update_receiver_pos_y(val):
    global receiver_pos
    receiver_pos[1] = float(val)
    ry_label.config(text=f"Y: {receiver_pos[1]:.2f}")

def update_receiver_pos_z(val):
    global receiver_pos
    receiver_pos[2] = float(val)
    rz_label.config(text=f"Z: {receiver_pos[2]:.2f}")

def toggle_signal_source():
    if use_real_time.get():
        use_local_file.set(False)
        print("Using real-time microphone input.")
    elif use_local_file.get():
        use_real_time.set(False)
        print("Using local file input.")
    elif not use_real_time.get() and not use_local_file.get():
        use_real_time.set(False)
        use_local_file.set(False)

def upload_file():
    global file_path
    file_path = askopenfilename(filetypes=[("WAV files", "*.wav")])
    if file_path:
        print(f"File selected: {file_path}")
        use_local_file.set(False)
        use_real_time.set(False)
        process(file_path,output_file_path)

def process(file_path,output_file_path):
    global OPRATE ,OPWIDTH, OPLEN, OPCHANNELS, OPBLOCKLEN
    global input_wf,output_wf
    global p,stream
    input_wf = wave.open(file_path, 'rb')
    add_room_reverb_file(file_path,output_file_path,room_dim, source_pos, receiver_pos, rt60)
    output_wf = wave.open(output_file_path, 'rb')
    OPRATE        = output_wf.getframerate()     # Frame rate (frames/second)
    OPWIDTH       = output_wf.getsampwidth()     # Number of bytes per sample
    OPLEN         = output_wf.getnframes()       # Signal length
    OPCHANNELS    = output_wf.getnchannels()     # Number of channels
    p = pyaudio.PyAudio()

    stream = p.open(
        format=p.get_format_from_width(OPWIDTH),
        channels=OPCHANNELS,
        rate=OPRATE,
        input=True,
        output=True,
        frames_per_buffer=OPBLOCKLEN
    )


'''
def get_signal():
    global file_path,output_file_path,output_wf
    global OPRATE ,OPWIDTH, OPLEN, OPCHANNELS, OPBLOCKLEN
    if use_real_time.get():
        # 实时麦克风输入
        input_bytes = stream.read(FPB, exception_on_overflow=False)
        signal = np.frombuffer(input_bytes, dtype=np.int16)
        
        # 如果是立体声,只保留一个声道
        if CHANNELS > 1:
            signal = signal[::CHANNELS]
            
        # 确保长度为FPB
        if len(signal) > FPB:
            signal = signal[:FPB]
        elif len(signal) < FPB:
            signal = np.pad(signal, (0, FPB - len(signal)))
            
        return signal.astype(np.float32) / 32768.0
    
    elif use_local_file.get() and file_path:
        
        add_room_reverb_file(file_path,output_file_path,room_dim, source_pos, receiver_pos, rt60)
        output_wf = wave.open(output_file_path, 'rb')
        OPRATE        = output_wf.getframerate()     # Frame rate (frames/second)
        OPWIDTH       = output_wf.getsampwidth()     # Number of bytes per sample
        OPLEN         = output_wf.getnframes()       # Signal length
        OPCHANNELS    = output_wf.getnchannels()     # Number of channels
        OPBLOCKLEN = 456
        # # 读取本地文件
        # with wave.open(file_path, 'rb') as wav:
        #     # 获取音频参数
        #     n_channels = wav.getnchannels()
        #     n_frames = wav.getnframes()
            
        #     # 读取音频数据并转换为numpy数组
        #     signal = wav.readframes(n_frames)
        #     signal = np.frombuffer(signal, dtype=np.int16)

        #     # 重塑数组并只返回第一个声道
        #     if n_channels > 1:
        #         signal = signal.reshape(-1, n_channels)[:, 0]

        #     # 重塑数组为(n_samples, n_channels)
        #     # signal = signal.reshape(-1, n_channels)

        #     # 转换为float类型进行处理
        #     signal = signal.astype(np.float32) / 32768.0

        #     # 读取完文件后重置 use_local_file 为 False
        #     # use_local_file.set(False) 

        #     return signal

    else:
        return np.zeros(FPB, dtype=np.float32)
'''

def reset():
    print("Reset button pressed.")

def quit_application():
    root.destroy()
    ...


if __name__ == "__main__":

    OPRATE = 8000
    OPBLOCKLEN=600

    output_file_path="output.wav"
    
    room_dim = [5.0, 4.0, 6.0]
    source_pos = [2.0, 3.5, 2.0]
    receiver_pos = [2.0, 1.5, 2.0]
    rt60 = 0.4
    cutoff=10

    global use_real_time, use_local_file, file_path
    
    root = tk.Tk()
    root.title("Real-Time Reverb with Visualization")
    
    # 设置窗口的固定大小，例如宽度为1200，高度为800
    root.geometry("1100x700")
    use_real_time = tk.BooleanVar(value=False)  # 默认选择实时信号
    use_local_file = tk.BooleanVar(value=False)
    local_file_path = None

    # 滑块和参数输入框
    control_frame = tk.Frame(root, highlightthickness=1, highlightbackground="black")
    control_frame.pack(side=tk.LEFT, fill=tk.Y)
    
    # RT60 参数
    cutoff_label = tk.Label(control_frame, text="Cut off frequencey")
    cutoff_label.pack()
    cutoff_value_lable=tk.Label(control_frame,text=f"cutoff: {cutoff:.2f}")
    cutoff_slider = ttk.Scale(control_frame, from_=100, to=8000, orient="horizontal", command=update_cutoff)
    cutoff_slider.set(cutoff)
    cutoff_slider.pack()
    print(" alread")

    # RT60 参数
    rt60_label = tk.Label(control_frame, text="Reverb Time (RT60)")
    rt60_label.pack()
    rt60_value_lable=tk.Label(control_frame,text=f"rt60: {rt60:.2f}")
    rt60_slider = ttk.Scale(control_frame, from_=0.4, to=2.0, orient="horizontal", command=update_rt60)
    rt60_slider.set(rt60)
    rt60_slider.pack()

    # 房间参数
    room_frame = tk.Frame(control_frame)  # Create a frame for the sliders
    room_frame.pack(fill=tk.X)
    room_label = tk.Label(room_frame, text="Room Dimensions")
    room_label.pack()

    # X dimension
    x_frame = tk.Frame(room_frame)
    x_frame.pack(side=tk.LEFT, padx=5)  # Arrange horizontally in the room_frame
    x_label = tk.Label(x_frame, text=f"X: {room_dim[0]:.2f}")
    x_label.pack()
    x_slider = ttk.Scale(x_frame, from_=1.0, to=10.0, orient="vertical", command=update_room_dim_x)
    x_slider.set(room_dim[0])
    x_slider.pack()

    # Y dimension
    y_frame = tk.Frame(room_frame)
    y_frame.pack(side=tk.LEFT, padx=5)  # Arrange horizontally in the room_frame
    y_label = tk.Label(y_frame, text=f"Y: {room_dim[1]:.2f}")
    y_label.pack()
    y_slider = ttk.Scale(y_frame, from_=2.0, to=10.0, orient="vertical", command=update_room_dim_y)
    y_slider.set(room_dim[1])
    y_slider.pack()

    # Z dimension
    z_frame = tk.Frame(room_frame)
    z_frame.pack(side=tk.LEFT, padx=5)  # Arrange horizontally in the room_frame
    z_label = tk.Label(z_frame, text=f"Z: {room_dim[2]:.2f}")
    z_label.pack()
    z_slider = ttk.Scale(z_frame, from_=1.0, to=10.0, orient="vertical", command=update_room_dim_z)
    z_slider.set(room_dim[2])
    z_slider.pack()

    # 声音源
    source_frame = tk.Frame(control_frame)
    source_frame.pack()

    source_label = tk.Label(source_frame, text="Source Position")
    source_label.pack()

    sx_frame=tk.Frame(source_frame)
    sx_frame.pack(side=tk.LEFT, padx=5)  # Arrange horizontally in the room_frame
    sx_label = tk.Label(sx_frame, text=f"X: {source_pos[0]:.2f}")
    sx_label.pack()
    sx_slider = ttk.Scale(sx_frame, from_=1.0, to=10.0, orient="vertical", command=update_source_pos_x)
    sx_slider.set(source_pos[0])
    sx_slider.pack()

    sy_frame=tk.Frame(source_frame)
    sy_frame.pack(side=tk.LEFT, padx=5)  # Arrange horizontally in the room_frame
    sy_label = tk.Label(sy_frame, text=f"Y: {source_pos[1]:.2f}")
    sy_label.pack()
    sy_slider = ttk.Scale(sy_frame, from_=1.0, to=10.0, orient="vertical", command=update_source_pos_y)
    sy_slider.set(source_pos[1])
    sy_slider.pack()

    sz_frame=tk.Frame(source_frame)
    sz_frame.pack(side=tk.LEFT, padx=5)  # Arrange horizontally in the room_frame
    sz_label = tk.Label(sz_frame, text=f"Z: {source_pos[2]:.2f}")
    sz_label.pack()
    sz_slider = ttk.Scale(sz_frame, from_=1.0, to=10.0, orient="vertical", command=update_source_pos_z)
    sz_slider.set(source_pos[2])
    sz_slider.pack()

    # 接收源
    receiver_frame = tk.Frame(control_frame)
    receiver_frame.pack()

    receiver_label = tk.Label(receiver_frame, text="Receiver Position")
    receiver_label.pack()

    rx_frame=tk.Frame(receiver_frame)
    rx_frame.pack(side=tk.LEFT, padx=5)  # Arrange horizontally in the room_frame
    rx_label = tk.Label(rx_frame, text=f"X: {receiver_pos[0]:.2f}")
    rx_label.pack()
    rx_slider = ttk.Scale(rx_frame, from_=1.0, to=10.0, orient="vertical", command=update_receiver_pos_x)
    rx_slider.set(receiver_pos[0])
    rx_slider.pack()

    ry_frame=tk.Frame(receiver_frame)
    ry_frame.pack(side=tk.LEFT, padx=5)  # Arrange horizontally in the room_frame
    ry_label = tk.Label(ry_frame, text=f"Y: {receiver_pos[1]:.2f}")
    ry_label.pack()
    ry_slider = ttk.Scale(ry_frame, from_=1.0, to=10.0, orient="vertical", command=update_receiver_pos_y)
    ry_slider.set(receiver_pos[1])
    ry_slider.pack()

    rz_frame=tk.Frame(receiver_frame)
    rz_frame.pack(side=tk.LEFT, padx=5)  # Arrange horizontally in the room_frame
    rz_label = tk.Label(rz_frame, text=f"Z: {receiver_pos[2]:.2f}")
    rz_label.pack()
    rz_slider = ttk.Scale(rz_frame, from_=1.0, to=10.0, orient="vertical", command=update_receiver_pos_z)
    rz_slider.set(receiver_pos[2])
    rz_slider.pack()

     # 在 control_frame 下面添加按钮子框架
    button_frame = tk.Frame(control_frame)
    button_frame.pack()
    print("Button frame added to control_frame.")

    # 添加 check_box
    # 添加勾选框到 UI
    signal_frame = tk.Frame(control_frame, highlightthickness=1, highlightbackground="black")
    signal_frame.pack(pady=10)

    real_time_check = tk.Checkbutton(
        signal_frame,
        text="Real Time",
        variable=use_real_time,
        command=toggle_signal_source
    )
    real_time_check.pack(side=tk.LEFT)

    local_file_check = tk.Checkbutton(
        signal_frame,
        text="Local File",
        variable=use_local_file,
        command=toggle_signal_source
    )
    local_file_check.pack(side=tk.LEFT)


    # 添加 "Upload File" 按钮
    upload_button = tk.Button(button_frame,text="upload file",command=upload_file)
    upload_button.pack(side=tk.LEFT, padx=5, pady=5)

    # 添加 "Reset" 按钮
    reset_button = tk.Button(button_frame, text="Reset", command=reset)
    reset_button.pack(side=tk.LEFT, padx=5, pady=5)

    # 添加 "Quit" 按钮
    quit_button = tk.Button(button_frame, text="Quit", command=quit_application)
    quit_button.pack(side=tk.LEFT, padx=5, pady=5)

    chart_frame = tk.Frame(root,width=200)
    chart_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)  # 新建一个 Frame 
    # 图表区域
    fig, axs = plt.subplots(2, 2, figsize=(2,1),dpi=40)
    fig.tight_layout(pad=2.0)

    # 输入信号时域
    ax_input_time = axs[0, 0]
    line_input_time, = ax_input_time.plot([], [], lw=2, label="Input Signal (Time)")
    ax_input_time.set_xlim(0, OPBLOCKLEN)
    ax_input_time.set_ylim(-20000, 20000)
    ax_input_time.set_title("Input Signal (Time Domain)")
    ax_input_time.set_xlabel("Sample Index")
    ax_input_time.set_ylabel("Amplitude")
    ax_input_time.legend()

    # 输入信号频域
    ax_input_freq = axs[0, 1]
    line_input_freq, = ax_input_freq.plot([], [], lw=2, label="Input Signal (Freq)")
    ax_input_freq.set_xlim(0, OPRATE // 2)
    ax_input_freq.set_ylim(0, 2000)
    ax_input_freq.set_title("Input Signal (Frequency Domain)")
    ax_input_freq.set_xlabel("Frequency (Hz)")
    ax_input_freq.set_ylabel("Magnitude")
    ax_input_freq.legend()

    # 输出信号时域
    ax_output_time = axs[1, 0]
    line_output_time, = ax_output_time.plot([], [], lw=2, label="Reverb Signal (Time)")
    ax_output_time.set_xlim(0, OPBLOCKLEN)
    ax_output_time.set_ylim(-20000, 20000)
    ax_output_time.set_title("Reverb Signal (Time Domain)")
    ax_output_time.set_xlabel("Sample Index")
    ax_output_time.set_ylabel("Amplitude")
    ax_output_time.legend()

    # 输出信号频域
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

    ani = FuncAnimation(fig, update_plot,interval=15, blit=True)
    canvas.draw()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Real-time reverb stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
