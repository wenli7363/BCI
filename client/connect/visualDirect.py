import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from EEGSerialPortManager import EEGSerialPortManager,SERIAL_PORT_NAME
from time import sleep


# 创建EEG数据驱动和串口管理器实例
eeg_serial_port_manager = EEGSerialPortManager()
# eeg_driver = EEGDataDriver()

# 尝试打开串口
if eeg_serial_port_manager.open_serial_port():
    print(f"成功打开串口: {SERIAL_PORT_NAME}")
    
    # 配置串口以开始监听数据
    eeg_serial_port_manager.config_serial_port()
    eeg_serial_port_manager.request_data()
    print("发送命令成功")

# 模拟获取脑电数据的函数
def get_eeg_data():
    # 这里假设获取到的数据是随机的
    return  -50 + (50 - (-50)) *np.random.rand(32, 125)
    # return np.array(eeg_serial_port_manager.eeg_driver.get_eeg_data())



# 获取脑电数据
eeg_data = get_eeg_data()

# 初始化图形和子图
fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)

# 绘制前4个通道的数据
lines = []
channels_to_plot = [10, 11, 12, 13]  # 前4个通道的索引

# 初始化折线图
for idx, channel in enumerate(channels_to_plot):
    line, = axs[idx].plot(eeg_data[channel])
    lines.append(line)
    axs[idx].set_ylabel(f'Channel {channel + 1}')  # 设置y轴标签
    axs[idx].set_ylim(-120, 120)

# 设置x轴标签
plt.xlabel('Sample Point')

# 设置总标题
plt.suptitle('EEG Data (First 4 Channels)')

# 更新函数，用于更新折线图
def update(frame):
    # 获取新数据
    eeg_data = get_eeg_data()
    # 更新前4个通道的折线图数据
    for idx, channel in enumerate(channels_to_plot):
        lines[idx].set_data(np.arange(125), eeg_data[channel])
    return lines

# 创建动画
ani = FuncAnimation(fig, update, frames=120, interval=150, blit=True)

# 显示图形
plt.show()