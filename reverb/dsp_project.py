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
import queue
import threading

class ReverbProcessor:
    def __init__(self):
        # Audio parameters
        self.CHUNK = 4096*2 
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.running = False

        # Data buffer
        self.input_data = np.zeros(self.CHUNK)
        self.output_data = np.zeros(self.CHUNK)
        
        # Modify effect parameters, enhance reverb effect
        self.room_size = 0.8  # Increase room size
        self.wet_level = 0.7  # Increase wet signal ratio
        self.damping = 0.5    # Add damping
        self.width = 0.5      # Increase stereo width
        
        # Create effect instances
        self.board = Pedalboard([
            HighpassFilter(cutoff_frequency_hz=100.0),    # Remove low frequency noise below 100 Hz
            LowpassFilter(cutoff_frequency_hz=5000.0),    # Limit high frequency below 5 kHz
            Reverb(
                room_size=self.room_size,
                wet_level=self.wet_level,
                damping=self.damping,
                width=self.width
            )
        ])

        # RIR Generator parameters
        self.room_dim = [5, 4, 6]
        self.source_pos = [2, 3.5, 2]
        self.receiver_pos = [2, 1.5, 2]
        self.rt60 = 0.4
        self.conv_gain = 5.0
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
            # Initialize PyAudio
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
            messagebox.showerror("Error", f"Audio device initialization failed: {str(e)}")
            raise
        
        # Start audio processing thread
        self.input_queue = queue.Queue(maxsize=10)
        self.output_queue = queue.Queue(maxsize=10)
        self.processing_thread_running = True
        self.processing_thread = threading.Thread(target=self.process_audio)
        self.processing_thread.start()

    def process_audio(self):
        while self.processing_thread_running:
            try:
                # Get data from input queue
                audio_data = self.input_queue.get(timeout=0.1)
                if self.use_pedalboard.get():
                    # Convert one-dimensional data to two-dimensional data (1, samples)
                    audio_data = np.expand_dims(audio_data, axis=0)
                    # Process using Pedalboard
                    processed = self.board(audio_data, self.RATE)
                    processed = processed.flatten()
                    print("Processing Thread: Processed using schroeder")
                elif self.use_rir.get(): 
                    # Process using rir generator
                    processed = ss.convolve(audio_data.flatten(), self.h.flatten(), mode='same')
                    processed = processed.astype(np.float32)
                    processed = processed * self.conv_gain
                    print("Processing Thread: Processed using Convolution")
                else :
                    print("Processing Thread: No algorithm selected")
                    processed = audio_data
                
                # Put processed data into output queue
                self.output_queue.put(processed)
            except queue.Empty:
                print("Processing Thread: No data in input queue")
                pass
            except Exception as e:
                print(f"Audio processing thread error: {e}")
                break
    
    def stop_processing_thread(self):
        self.processing_thread_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join()

    def audio_callback(self, in_data, frame_count, time_info, status):
        try:
            if not self.running:
                # If not started playing, return silent data
                return (np.zeros(frame_count * 4).tobytes(), pyaudio.paContinue)
                
            # Process input data
            audio_data = np.frombuffer(in_data, dtype=np.float32)
            
            # Put data into input queue
            self.input_queue.put(audio_data)

            # Get processed data from output queue
            if not self.output_queue.empty():
                processed = self.output_queue.get_nowait()
            else:
                print("Main Thread: No data in output queue")
                processed = np.zeros_like(audio_data)

            # Update data buffer
            self.input_data = audio_data
            self.output_data = processed
            
            # Record audio
            if self.is_recording:
                self.recorded_input.extend(self.input_data)
                self.recorded_output.extend(self.output_data)
                
            return (processed.tobytes(), pyaudio.paContinue)
            
        except Exception as e:
            print(f"Audio processing error: {str(e)}")
            return (in_data, pyaudio.paContinue)

    def setup_gui(self):
        self.root = tk.Tk()
        self.root.title("Reverb")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.use_pedalboard = tk.BooleanVar(value=False)
        self.use_rir = tk.BooleanVar(value=False)
        

        # Control panel
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.LEFT, padx=10)

        # Use custom style for ALGORITHM selection frame
        algorithm_frame = ttk.LabelFrame(control_frame, text="Algorithm Choice")
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
        
        pedalboard_frame = ttk.LabelFrame(control_frame, text="Schroeder Parameters")
        pedalboard_frame.pack(pady=10, padx=5, fill="x")

        # Room Size slider
        ttk.Label(pedalboard_frame, text="Room Size").pack()
        self.room_size_slider = ttk.Scale(
            pedalboard_frame, from_=0, to=1,
            value=self.room_size,
            command=self.update_room_size
        )
        self.room_size_slider.pack()
        
        # Wet Level slider
        ttk.Label(pedalboard_frame, text="Wet Level").pack()
        self.wet_level_slider = ttk.Scale(
            pedalboard_frame, from_=0, to=1,
            value=self.wet_level,
            command=self.update_wet_level
        )
        self.wet_level_slider.pack()

        # Damping slider
        ttk.Label(pedalboard_frame, text="Damping").pack()
        self.damping_slider = ttk.Scale(
            pedalboard_frame, from_=0, to=1,
            value=self.damping,
            command=self.update_damping
        )
        self.damping_slider.pack()

        # Width slider
        ttk.Label(pedalboard_frame, text="Width").pack()
        self.width_slider = ttk.Scale(
            pedalboard_frame, from_=0, to=1,
            value=self.width,
            command=self.update_width
        )   
        self.width_slider.pack()

        # Frame for rir generator
        rir_frame = ttk.LabelFrame(control_frame, text="Convolution Parameters")
        rir_frame.pack(pady=10, padx=5, fill="x")

        # Add combo box
        ttk.Label(rir_frame, text="Room Size Choose").pack()
        # self.room_choice_combo = ttk.Combobox(rir_frame, values=["Room", "Large Hall", "Church", "Theater"], state="readonly")
        self.room_choice_combo = ttk.Combobox(rir_frame, values=["Room", "Large Hall"], state="readonly")
        self.room_choice_combo.pack()

        self.room_choice_combo.current(0)
        self.room_choice_combo.bind("<<ComboboxSelected>>", self.update_room_choice)
        
        # Start button
        self.start_button = ttk.Button(control_frame, text="Start", command=self.toggle_processing)
        self.start_button.pack(pady=10)
        
        # Add record button
        self.record_button = ttk.Button(control_frame, text="Record Audio", command=self.toggle_recording)
        self.record_button.pack(pady=5)
        
        # Add save button
        self.save_button = ttk.Button(control_frame, text="Save Audio", command=self.save_audio)
        self.save_button.pack(pady=5)
        
        self.setup_plots()
    
    def setup_plots(self):
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.fig, self.axs = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.tight_layout(pad=3.0)
        
        # Initialize all plots
        time = np.arange(self.CHUNK)
        freq = np.fft.rfftfreq(self.CHUNK, 1/self.RATE)
        
        # Input time domain - adjust y-axis range
        self.input_time_line, = self.axs[0,0].plot(time, np.zeros(self.CHUNK))
        self.axs[0,0].set_title("Input (Time Domain)")
        self.axs[0,0].title.set_size(8)
        self.axs[0,0].set_ylim(-1, 1)  
        self.axs[0,0].tick_params(axis='both', labelsize=6)
        self.axs[0,0].grid(True)
        
        # Input frequency domain
        self.input_freq_line, = self.axs[0,1].plot(freq, np.zeros(self.CHUNK//2 + 1))
        self.axs[0,1].set_title("Input (Frequency Domain)")
        self.axs[0,1].title.set_size(8)
        self.axs[0,1].set_ylim(0, 200)  
        self.axs[0,1].tick_params(axis='both', labelsize=6)
        self.axs[0,1].grid(True)
        
        # Output time domain and frequency domain
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
        
        # Add labels
        for ax in self.axs.flat:
            ax.set_xlabel('Samples' if 'Time' in ax.get_title() else 'Frequency (Hz)')
            ax.set_ylabel('Amplitude')
            # Set label font size
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
            # Sampling index, take one point every interval
            sampled_indices = np.arange(0, self.CHUNK, desample_factor)
            # Sample the time axis
            time = sampled_indices
            freq = np.fft.rfftfreq(self.CHUNK, 1/self.RATE)
            
            # Ensure using one-dimensional data for plotting
            input_data_1d = self.input_data[0] if self.input_data.ndim > 1 else self.input_data
            output_data_1d = self.output_data[0] if self.output_data.ndim > 1 else self.output_data

            # Update time domain plot, sample data
            print(input_data_1d.shape)
            print(input_data_1d[sampled_indices].shape)
            self.input_time_line.set_data(time,input_data_1d[sampled_indices])
            self.output_time_line.set_data(time, output_data_1d[sampled_indices])

            # Update frequency domain plot
            
            input_fft = np.abs(np.fft.rfft(input_data_1d))
            output_fft = np.abs(np.fft.rfft(output_data_1d))

            # Sample frequency domain data
            freq_sampled = freq[::desample_factor]
            input_fft_sampled = input_fft[::desample_factor]
            output_fft_sampled = output_fft[::desample_factor]

            self.input_freq_line.set_data(freq_sampled, input_fft_sampled)
            self.output_freq_line.set_data(freq_sampled, output_fft_sampled)

        except Exception as e:
            print(f"Plot update error: {str(e)}")

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
            # Update effect parameters
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
    
    def update_room_choice(self, event):
        room = self.room_choice_combo.get()
        if room == "Room":
            self.room_dim = [5, 4, 6]
            self.source_pos = [2, 3.5, 2]
            self.receiver_pos = [2, 1.5, 2]
            self.rt60 = 0.4
            self.conv_gain = 5.0
        
        elif room == "Large Hall":
            self.room_dim = [20, 20, 20]
            self.source_pos = [2, 3.5, 2]
            self.receiver_pos = [2, 1.5, 2]
            self.rt60 = 0.9
            self.conv_gain = 5.0

        else:
            print("No room selected")
            return
        
        self.update_reverb()


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
            
            # Save input and output
            self._save_wav(input_file, np.array(self.recorded_input))
            self._save_wav(output_file, np.array(self.recorded_output))
                
            messagebox.showinfo("Success", 
                f"Audio Saved:\nInput Audio: {input_file}\n\nOutput Audio: {output_file}")

    def _save_wav(self, filename, data):
        """Save WAV file"""
        
        # Ensure data range is within [-1, 1]
        data = np.clip(data, -1, 1)
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(2)  # Use 16-bit sampling
            wf.setframerate(self.RATE)
            
            # Convert to 16-bit integer
            data_int = (data * 32767).astype(np.int16)
            wf.writeframes(data_int.tobytes())

    def on_closing(self):
        try:
            # 1. Stop main loop flag
            self.running = False
            
            # 2. Stop animation
            if hasattr(self, 'ani'):
                self.ani.event_source.stop()
            
            # 3. Stop processing thread
            self.stop_processing_thread()
            
            # 4. Clear queues
            while not self.input_queue.empty():
                self.input_queue.get_nowait()
            while not self.output_queue.empty():
                self.output_queue.get_nowait()
                
            # 5. Close audio stream
            if hasattr(self, 'stream'):
                self.stream.stop_stream()
                self.stream.close()
                
            # 6. Terminate PyAudio
            if hasattr(self, 'p'):
                self.p.terminate()
                
            # 7. Close window
            plt.close('all')  # Close all matplotlib windows
            self.root.quit()  # Exit mainloop
            self.root.destroy()  # Destroy window
            
        except Exception as e:
            print(f"Error while closing program: {str(e)}")
        
    def run(self):
            
        self.ani = FuncAnimation(
            self.fig,
            self.update_plots,
            frames=None,  # Infinite frames
            interval=50, # Update interval
            blit=True,
            cache_frame_data=False,  # Disable frame caching
            save_count=None  # Do not save frames
        )
            
        self.root.mainloop()

if __name__ == "__main__":
    try:
        processor = ReverbProcessor()
        processor.run()
    except Exception as e:
        print(f"Program runtime error: {str(e)}")