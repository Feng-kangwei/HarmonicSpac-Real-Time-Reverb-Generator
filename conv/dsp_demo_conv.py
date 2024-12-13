import numpy as np
import scipy.signal as ss
import soundfile as sf
import rir_generator as rir
import matplotlib.pyplot as plt

def add_room_reverb(audio_path, save_path, 
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

def plot_waveform(signal, title="Waveform"):
    plt.figure(figsize=(10, 3))
    plt.plot(signal[:, 0])
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.show()

# 使用示例
if __name__ == "__main__":
    input_file = "conv/example.wav"
    output_file = "conv/output_room_reverb2.wav"
    
    # 自定义房间参数
    room_config = {
        "room_dim": [5, 4, 6],        # 房间尺寸
        "source_pos": [2, 3.5, 2],    # 声源位置
        "receiver_pos": [2, 1.5, 2],  # 接收器位置
        "rt60": 0.6                   # 混响时间
    }
    
    # 添加混响效果
    processed, fs = add_room_reverb(input_file, output_file, **room_config)
    
    # 显示原始波形和处理后的波形
    original, fs = sf.read(input_file, always_2d=True)
    plot_waveform(original, "Original Audio")
    plot_waveform(processed, "Audio with Room Reverb")