import threading
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from EEGSerialPortManager import EEGSerialPortManager,SERIAL_PORT_NAME,eeg_driver


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
    return np.random.rand(32, 125)
    # return np.array(eeg_driver.get_eeg_data())

# 创建动画
fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)

lines = [ax.plot([], [])[0] for ax in axs]  # 创建多条线
window_size = 125  # 滑动窗口大小
x_data = np.arange(window_size)
y_data = np.zeros((32, window_size))

# 更新函数，用于更新动画
def update(frame):
    global y_data
    eeg_data = get_eeg_data()
    y_data = np.roll(y_data, -1, axis=1)
    y_data[:, -1] = eeg_data[:, 0]  # 更新第一个通道的数据
    for i, line in enumerate(lines):
        line.set_data(x_data, y_data[i])  # 更新每个通道的数据
    return lines

# 设置坐标轴
for ax in axs:
    ax.set_xlim(0, window_size - 1)
    ax.set_ylim(0, 1)  # 假设幅值范围是[0, 1]

# 创建动画
ani = FuncAnimation(fig, update, frames=None, blit=True, interval=50)

plt.tight_layout()
plt.show()