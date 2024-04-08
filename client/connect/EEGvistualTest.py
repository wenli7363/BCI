import matplotlib.pyplot as plt
import numpy as np
import threading
from EEGSerialPortManager import EEGSerialPortManager
from EEGDataDriver import EEGDataDriver
from EEGPacketDataParser import EEGPacketDataParser
from EEGDataTransformHelper import EEGDataTransformHelper
import time

# 初始化串口管理器
eeg_serial_port_manager = EEGSerialPortManager()
if not eeg_serial_port_manager.open_serial_port():
    print("Failed to open serial port.")
else:
    print(f"Opened serial port: {eeg_serial_port_manager.SERIAL_PORT_NAME}")

# 初始化EEG数据驱动
eeg_data_driver = EEGDataDriver()

# 实时可视化EEG数据的函数
def visualize_eeg_data(data_buffer, channels):
    # 清除之前的图表
    plt.clf()
    # 创建一个新的图表
    fig, ax = plt.subplots()
    # 为每个通道绘制数据
    for channel in channels:
        # 获取当前通道的数据
        channel_data = [data_buffer[i][channel] for i in range(len(data_buffer))]
        # 绘制数据
        ax.plot(channel_data, label=f'Channel {channel+1}')
    # 设置图表标题和标签
    ax.set_title('Real-time EEG Data Visualization')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('EEG Value')
    ax.legend()
    # 显示图表
    plt.pause(0.001)  # 暂停一小段时间，以便Matplotlib能够更新图表

# 开始实时读取和可视化EEG数据
try:
    while True:
        # 读取串口数据
        raw_data = eeg_serial_port_manager.read_data()
        if raw_data:
            # 解析EEG数据
            eeg_data_buffer = eeg_data_driver.get_eeg_data()
            # 可视化EEG数据
            visualize_eeg_data(eeg_data_buffer, range(eeg_data_driver.maxDataChannel))
            # 每隔100毫秒更新一次数据
            time.sleep(0.1)
except KeyboardInterrupt:
    # 用户中断程序时，关闭串口和图表
    eeg_serial_port_manager.close_serial_port()
    plt.close()
    print("Serial port closed and visualization stopped.")