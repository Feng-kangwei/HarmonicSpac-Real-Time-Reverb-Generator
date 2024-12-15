import numpy as np
import pyaudio
import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
from pedalboard import Pedalboard, Reverb, HighpassFilter, LowpassFilter
import wave
import rir_generator as rir
import scipy.signal as ss

class ReverbProcessor:
    def __init__(self):
        # 音频参数
        self.CHUNK = 8192
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.running = False

        # 数据缓冲
        self.input_data = np.zeros(self.CHUNK)
        self.output_data = np.zeros(self.CHUNK)
        
        # 修改效果器参数，增加混响效果
        self.room_size = 0.8  # 增大房间大小
        self.wet_level = 0.7  # 增加湿信号比例
        self.damping = 0.5    # 添加阻尼
        self.width = 0.5      # 增加立体声宽度
        
        # 创建效果器实例
        self.board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=100.0),    # 移除100 Hz以下低频噪声
            LowpassFilter(cutoff_frequency_hz=5000.0),    # 限制高频到5 kHz以下
            Reverb(
                room_size=self.room_size,
                wet_level=self.wet_level,
                damping=self.damping,
                width=self.width
            )
        ])

        # RIR Generator 参数
        self.room_dim = [5, 4, 6]
        self.source_pos = [2, 3.5, 2]
        self.receiver_pos = [2, 1.5, 2]
        self.rt60 = 0.4

        # rir generator
        self.h = rir.generate(
                c=340,
                fs=self.RATE,
                L=self.room_dim,
                s=self.source_pos,
                r=self.receiver_pos,
                reverberation_time=self.rt60,
                nsample=1024)
        self.h = self.h.reshape(-1,1)

        self.is_recording = False
        self.recorded_input = []
        self.recorded_output = []

    
        try:
            # 初始化PyAudio
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(
                format=self.FORMAT,
                channels=self.CHANNELS,
                rate=self.RATE,
                input=True,
                output=True,
                frames_per_buffer=self.CHUNK,
                stream_callback=self.audio_callback
            )
            self.setup_gui()
        except Exception as e:
            messagebox.showerror("错误", f"音频设备初始化失败: {str(e)}")
            raise
    

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            if not self.running:
                # 如果未开始播放，返回静音数据
                return (np.zeros(frame_count * 4).tobytes(), pyaudio.paContinue)
                
            # 处理输入数据
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            

            if (self.use_pedalboard.get()):
                # 将一维数据转换为二维数据 (1, samples)
                audio_data = np.expand_dims(audio_data, axis=0)
                    
                # 使用 Pedalboard 处理
                processed = self.board(audio_data, self.RATE)
            elif (self.use_rir.get()): 
                # 使用rir generator处理
                processed = ss.convolve(audio_data.flatten(), self.h.flatten(), mode='same')
                processed = processed.astype(np.float32)
            else :
                # print("No algorithm selected")
                processed = audio_data
            
            # 更新数据缓冲
            self.input_data = audio_data
            self.output_data = processed
            
            # 录制音频
            if self.is_recording:
                self.recorded_input.extend(self.input_data)
                self.recorded_output.extend(self.output_data)
                
            return (processed.tobytes(), pyaudio.paContinue)
            
        except Exception as e:
            print(f"音频处理错误: {str(e)}")
            return (in_data, pyaudio.paContinue)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Reverb")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.use_pedalboard = tk.BooleanVar(value=False)
        self.use_rir = tk.BooleanVar(value=False)
        

        # 控制面板
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=10)

        # 使用自定义样式的ALGORITHM选择框架
        algorithm_frame = ttk.LabelFrame(control_frame, text="Algorithm Choose")
        algorithm_frame.pack(pady=10, padx=15, fill="x")
        algorithm_frame.config(width=200, height=100)


        ttk.Checkbutton(
            algorithm_frame, 
            text="Schroeder",
            variable=self.use_pedalboard,
            command= lambda : self.on_algorithm_change('pedalboard')
        ).pack(padx=5, pady=2)

        ttk.Checkbutton(
            algorithm_frame, 
            text="Convolution",
            variable=self.use_rir,
            command=lambda : self.on_algorithm_change('rir')
        ).pack(padx=5, pady=2)
        
        pedalboard_frame = ttk.LabelFrame(control_frame, text="Pedalboard Parameters")
        pedalboard_frame.pack(pady=10, padx=5, fill="x")

        # Room Size滑块
        ttk.Label(pedalboard_frame, text="Room Size").pack()
        self.room_size_slider = ttk.Scale(
            pedalboard_frame, from_=0, to=1,
            value=self.room_size,
            command=self.update_room_size
        )
        self.room_size_slider.pack()
        
        # Wet Level滑块
        ttk.Label(pedalboard_frame, text="Wet Level").pack()
        self.wet_level_slider = ttk.Scale(
            pedalboard_frame, from_=0, to=1,
            value=self.wet_level,
            command=self.update_wet_level
        )
        self.wet_level_slider.pack()

        # damping滑块
        ttk.Label(pedalboard_frame, text="Damping").pack()
        self.damping_slider = ttk.Scale(
            pedalboard_frame, from_=0, to=1,
            value=self.damping,
            command=self.update_damping
        )
        self.damping_slider.pack()

        # Width滑块
        ttk.Label(pedalboard_frame, text="Width").pack()
        self.width_slider = ttk.Scale(
            pedalboard_frame, from_=0, to=1,
            value=self.width,
            command=self.update_width
        )   
        self.width_slider.pack()

        # frame for rir generator
        rir_frame = ttk.LabelFrame(control_frame, text="RIR Generator Parameters")
        rir_frame.pack(pady=10, padx=5, fill="x")

        
        
        # 启动按钮
        self.start_button = ttk.Button(control_frame, text="Start", command=self.toggle_processing)
        self.start_button.pack(pady=10)
        
        # 添加录制按钮
        self.record_button = ttk.Button(control_frame, text="Record Audio", command=self.toggle_recording)
        self.record_button.pack(pady=5)
        
        # 添加保存按钮
        self.save_button = ttk.Button(control_frame, text="Save Audio", command=self.save_audio)
        self.save_button.pack(pady=5)
        
        self.setup_plots()
    
    def setup_plots(self):
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)
        
        # 初始化所有图表
        time = np.arange(self.CHUNK)
        freq = np.fft.rfftfreq(self.CHUNK, 1/self.RATE)
        
        # 输入时域 - 调整y轴范围
        self.input_time_line, = self.axs[0,0].plot(time, np.zeros(self.CHUNK))
        self.axs[0,0].set_title("Input (Time Domain)")
        self.axs[0,0].title.set_size(8)
        self.axs[0,0].set_ylim(-1, 1)  
        self.axs[0,0].tick_params(axis='both', labelsize=6)
        self.axs[0,0].grid(True)
        
        # 输入频域
        self.input_freq_line, = self.axs[0,1].plot(freq, np.zeros(self.CHUNK//2 + 1))
        self.axs[0,1].set_title("Input (Frequency Domain)")
        self.axs[0,1].title.set_size(8)
        self.axs[0,1].set_ylim(0, 200)  
        self.axs[0,1].tick_params(axis='both', labelsize=6)
        self.axs[0,1].grid(True)
        
        # 输出时域和频域
        self.output_time_line, = self.axs[1,0].plot(time, np.zeros(self.CHUNK))
        self.axs[1,0].set_title("Output (Time Domain)")
        self.axs[1,0].title.set_size(8)
        self.axs[1,0].set_ylim(-1, 1)  
        self.axs[1,0].tick_params(axis='both', labelsize=6)
        self.axs[1,0].grid(True)
        
        self.output_freq_line, = self.axs[1,1].plot(freq, np.zeros(self.CHUNK//2 + 1))
        self.axs[1,1].set_title("Output (Frequency Domain)")
        self.axs[1,1].title.set_size(8)
        self.axs[1,1].set_ylim(0, 200)  
        self.axs[1,1].tick_params(axis='both', labelsize=6)
        self.axs[1,1].grid(True)
        
        # 添加标签
        for ax in self.axs.flat:
            ax.set_xlabel('Samples' if 'Time' in ax.get_title() else 'Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            # set label font size
            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
    
    def update_plots(self, frame):
        if not self.running:
            return self.input_time_line, self.input_freq_line, self.output_time_line, self.output_freq_line

        try:
            desample_factor = 8
            # 采样索引，每隔一个点取一个
            sampled_indices = np.arange(0, self.CHUNK, desample_factor)
            # 对时间轴进行采样
            time = sampled_indices
            freq = np.fft.rfftfreq(self.CHUNK, 1/self.RATE)
            
            # 确保使用一维数据进行绘图
            input_data_1d = self.input_data[0] if self.input_data.ndim > 1 else self.input_data
            output_data_1d = self.output_data[0] if self.output_data.ndim > 1 else self.output_data

            # 更新时域图，采样数据
            self.input_time_line.set_data(time,input_data_1d[sampled_indices])
            self.output_time_line.set_data(time, output_data_1d[sampled_indices])

            # 更新频域图
            input_fft = np.abs(np.fft.rfft(input_data_1d))
            output_fft = np.abs(np.fft.rfft(output_data_1d))

            # 对频域数据进行采样
            freq_sampled = freq[::desample_factor]
            input_fft_sampled = input_fft[::desample_factor]
            output_fft_sampled = output_fft[::desample_factor]

            self.input_freq_line.set_data(freq_sampled, input_fft_sampled)
            self.output_freq_line.set_data(freq_sampled, output_fft_sampled)

        except Exception as e:
            print(f"更新图表错误: {str(e)}")

        return self.input_time_line, self.input_freq_line, self.output_time_line, self.output_freq_line
    
    def toggle_processing(self):
        self.running = not self.running
        self.start_button.config(text="Stop" if self.running else "Start")

    def on_algorithm_change(self, source):
        if not self.use_pedalboard.get() and not self.use_rir.get():
            print("No algorithm selected")
            return

        if (self.use_pedalboard.get() and self.use_rir.get()):
            if source == 'pedalboard':
                self.use_rir.set(False)
                print("Pedalboard is selected") 
            else:
                self.use_pedalboard.set(False)
                print("RIR is selected")
            
        
    def update_room_size(self, value):
        self.room_size = float(value)
        self.update_reverb()
        
    def update_wet_level(self, value):
        self.wet_level = float(value)
        self.update_reverb()
    
    def update_damping(self, value):
        self.damping = float(value)
        self.update_reverb()

    def update_width(self, value):
        self.width = float(value)
        self.update_reverb()
        
    def update_reverb(self):
        if self.use_pedalboard.get():
            # 更新效果器参数
            self.board = Pedalboard([
                HighpassFilter(cutoff_frequency_hz=100.0),   
                LowpassFilter(cutoff_frequency_hz=5000.0),   
                Reverb(
                    room_size=self.room_size,
                    wet_level=self.wet_level,
                    damping=self.damping,
                    width=self.width
                )
            ])
        elif self.use_rir.get():
            self.h = rir.generate(
                        c=340,
                        fs=self.RATE,
                        L=self.room_dim,
                        s=self.source_pos,
                        r=self.receiver_pos,
                        reverberation_time=self.rt60,
                        nsample=1024)
            self.h = self.h.reshape(-1,1)
    
    def toggle_recording(self):
        self.is_recording = not self.is_recording
        self.record_button.config(text="Stop Recording" if self.is_recording else "Record Audio")
        if not self.is_recording and self.recorded_input:
            self.save_button.config(state="normal")
        if self.is_recording:
            self.recorded_input = []
            self.recorded_output = []
            self.save_button.config(state="disabled")

    def save_audio(self):
        if not self.recorded_input or not self.recorded_output:
            messagebox.showwarning("Warning", "There is no audio to save.")
            return
            
        from tkinter import filedialog
        input_file = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")]
        )
        
        if input_file:
            output_file = input_file.replace(".wav", "_processed.wav")
            
            # 保存输入和输出
            self._save_wav(input_file, np.concatenate(self.recorded_input))
            self._save_wav(output_file, np.concatenate(self.recorded_output))
                
            messagebox.showinfo("Success", 
                f"Audio Saved:\nInput Audio: {input_file}\nOutput Audio: {output_file}")

    def _save_wav(self, filename, data):
        """保存WAV文件"""
        
        # 确保数据范围在 [-1, 1] 之间
        data = np.clip(data, -1, 1)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # 使用16位采样
            wf.setframerate(self.RATE)
            
            # 转换为16位整数
            data_int = (data * 32767).astype(np.int16)
            wf.writeframes(data_int.tobytes())

    def on_closing(self):
        self.running = False
        if hasattr(self, 'stream'):
            self.stream.stop_stream()
            self.stream.close()
        if hasattr(self, 'p'):
            self.p.terminate()
        self.root.destroy()
        
    def run(self):
            
        self.ani = FuncAnimation(
            self.fig,
            self.update_plots,
            frames=None,  # 无限帧
            interval=50, # 更新间隔
            blit=True,
            cache_frame_data=False,  # 禁用帧缓存
            save_count=None  # 不保存帧
        )
            
        self.root.mainloop()

if __name__ == "__main__":
    try:
        processor = ReverbProcessor()
        processor.run()
    except Exception as e:
        print(f"程序运行错误: {str(e)}")