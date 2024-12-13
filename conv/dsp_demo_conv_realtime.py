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

def update_plot(frame):
    global audio_data, signal_output
    audio_data = get_signal()
    # audio_data = highpass_filter(audio_data,cutoff)
    signal_output = add_room_reverb(audio_data, RATE, room_dim, source_pos, receiver_pos, rt60)
    # convert to int16
    output_bytes = (signal_output * 32767).astype(np.int16).tobytes()
    # output_bytes = np.int16(signal_output * 32767).tobytes()
    stream.write(output_bytes)

    # 更新输入信号时域
    line_input_time.set_data(range(BLOCKLEN), audio_data)
    # 更新输入信号频域
    input_freq_data = np.abs(np.fft.rfft(audio_data)) / BLOCKLEN
    input_freqs = np.fft.rfftfreq(BLOCKLEN, d=1.0 / RATE)
    line_input_freq.set_data(input_freqs, input_freq_data)

    # 更新输出信号时域
    line_output_time.set_data(range(BLOCKLEN), signal_output[:BLOCKLEN])
    # 更新输出信号频域
    output_freq_data = np.abs(np.fft.rfft(signal_output[:BLOCKLEN])) / BLOCKLEN
    output_freqs = np.fft.rfftfreq(BLOCKLEN, d=1.0 / RATE)
    line_output_freq.set_data(output_freqs, output_freq_data)

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
        use_local_file.set(True)
        use_real_time.set(False)

def get_signal():
    global file_path
    if use_real_time.get():
        input_bytes = stream.read(FPB, exception_on_overflow=False)
        return np.frombuffer(input_bytes, dtype=np.int16).astype(np.float32) / 32768
    
    elif use_local_file.get() and file_path:
        # 读取本地文件
        with wave.open(file_path, 'rb') as wav:
            # 获取音频参数
            n_channels = wav.getnchannels()
            n_frames = wav.getnframes()
            
            # 读取音频数据并转换为numpy数组
            signal = wav.readframes(n_frames)
            signal = np.frombuffer(signal, dtype=np.int16)

            # 重塑数组并只返回第一个声道
            if n_channels > 1:
                signal = signal.reshape(-1, n_channels)[:, 0]

            # 重塑数组为(n_samples, n_channels)
            # signal = signal.reshape(-1, n_channels)

            # 转换为float类型进行处理
            signal = signal.astype(np.float32) / 32768.0
            return signal

    else:
        return np.zeros(FPB, dtype=np.float32)


def reset():
    print("Reset button pressed.")

def quit_application():
    root.destroy()
    ...



if __name__ == "__main__":
    WIDTH = 2
    CHANNELS = 1
    RATE = 8000
    FPB = 2048
    BLOCKLEN = FPB

    p = pyaudio.PyAudio()

    stream = p.open(
        format=p.get_format_from_width(WIDTH),
        channels=CHANNELS,
        rate=RATE,
        input=True,
        output=True,
        frames_per_buffer=FPB
    )

    room_dim = [5.0, 4.0, 6.0]
    source_pos = [2.0, 3.5, 2.0]
    receiver_pos = [2.0, 1.5, 2.0]
    rt60 = 0.4
    cutoff=10

    audio_data = np.zeros(FPB, dtype=np.float32)
    signal_output = np.zeros(FPB, dtype=np.float32)

    global use_real_time, use_local_file, file_path
    

    root = tk.Tk()
    root.title("Real-Time Reverb with Visualization")
    
    # 设置窗口的固定大小，例如宽度为1200，高度为800
    root.geometry("800x700")
    use_real_time = tk.BooleanVar(value=False)  # 默认选择实时信号
    use_local_file = tk.BooleanVar(value=False)
    file_path = None


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
    fig, axs = plt.subplots(2, 2, figsize=(5, 3),dpi=100)
    fig.tight_layout(pad=2.0)

    # 输入信号时域
    ax_input_time = axs[0, 0]
    line_input_time, = ax_input_time.plot([], [], lw=2, label="Input Signal (Time)")
    ax_input_time.set_xlim(0, BLOCKLEN)
    ax_input_time.set_ylim(-0.025, 0.025)
    ax_input_time.set_title("Input Signal (Time Domain)")
    ax_input_time.set_xlabel("Sample Index")
    ax_input_time.set_ylabel("Amplitude")
    ax_input_time.legend()

    # 输入信号频域
    ax_input_freq = axs[0, 1]
    line_input_freq, = ax_input_freq.plot([], [], lw=2, label="Input Signal (Freq)")
    ax_input_freq.set_xlim(0, RATE // 2)
    ax_input_freq.set_ylim(0, 0.025)
    ax_input_freq.set_title("Input Signal (Frequency Domain)")
    ax_input_freq.set_xlabel("Frequency (Hz)")
    ax_input_freq.set_ylabel("Magnitude")
    ax_input_freq.legend()

    # 输出信号时域
    ax_output_time = axs[1, 0]
    line_output_time, = ax_output_time.plot([], [], lw=2, label="Reverb Signal (Time)")
    ax_output_time.set_xlim(0, BLOCKLEN)
    ax_output_time.set_ylim(-1, 1)
    ax_output_time.set_title("Reverb Signal (Time Domain)")
    ax_output_time.set_xlabel("Sample Index")
    ax_output_time.set_ylabel("Amplitude")
    ax_output_time.legend()

    # 输出信号频域
    ax_output_freq = axs[1, 1]
    line_output_freq, = ax_output_freq.plot([], [], lw=2, label="Reverb Signal (Freq)")
    ax_output_freq.set_xlim(0, RATE // 2)
    ax_output_freq.set_ylim(0, 0.1)
    ax_output_freq.set_title("Reverb Signal (Frequency Domain)")
    ax_output_freq.set_xlabel("Frequency (Hz)")
    ax_output_freq.set_ylabel("Magnitude")
    ax_output_freq.legend()

    canvas = FigureCanvasTkAgg(fig, master=chart_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    ani = FuncAnimation(fig, update_plot, interval=50, blit=True)
    canvas.draw()

    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("Real-time reverb stopped.")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
