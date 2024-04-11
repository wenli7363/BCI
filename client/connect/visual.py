import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from EEGSerialPortManager import EEGSerialPortManager, SERIAL_PORT_NAME, eeg_driver

# 创建EEG数据驱动和串口管理器实例
eeg_serial_port_manager = EEGSerialPortManager()

# 尝试打开串口
if eeg_serial_port_manager.open_serial_port():
    print(f"成功打开串口: {SERIAL_PORT_NAME}")

    # 配置串口以开始监听数据
    eeg_serial_port_manager.config_serial_port()
    eeg_serial_port_manager.request_data()
    print("发送命令成功")

# 获取脑电数据的函数
def get_eeg_data():
    return np.array(eeg_driver.get_eeg_data(), dtype=np.float64)

# 定义实时可视化代码
def visualize_eeg():
    # 初始化图和子图
    fig, axs = plt.subplots(4, 1, figsize=(10, 6))

    # 设置子图属性
    for ax in axs:
        ax.set_ylim(-100, 100)  # 根据您的数据调整Y轴范围
        ax.set_xlim(0, 125)  # X轴范围是125个样本点
        ax.set_ylabel('Amplitude')
        ax.set_xlabel('Sample Index')
        ax.grid(True)

    # 初始化滑动窗口
    window_size = 125  # 滑动窗口的大小
    eeg_data_window = np.zeros((4, window_size))  # 前4个通道的滑动窗口

    # 定义一个更新函数，用于动画
    def update(frame):
        nonlocal eeg_data_window
        # 获取新的脑电数据
        new_eeg_data = get_eeg_data()[:4]  # 获取前4个通道的数据

        # 将新数据添加到窗口并移除旧数据
        eeg_data_window = np.hstack((eeg_data_window[:, 1:], new_eeg_data))

        # 更新前四个通道的图形
        for i in range(4):
            axs[i].clear()
            axs[i].plot(eeg_data_window[i])
            axs[i].set_ylim(-100, 100)
            axs[i].set_xlim(0, window_size)
            axs[i].set_ylabel(f'Channel {i + 1}')
            axs[i].grid(True)

        # 设置图的标题
        fig.suptitle('EEG Data Visualization')

    # 创建动画
    ani = FuncAnimation(fig, update, interval=100)

    # 显示图形
    plt.tight_layout()
    plt.show()

# 启动实时可视化
visualize_eeg()
