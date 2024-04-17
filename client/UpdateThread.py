from PyQt5.QtCore import QThread, pyqtSignal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import threading
from constVar import DOWNSAMPLE_SIZE

def get_eeg_data():
    return -50 + (50 - (-50)) *np.random.rand(32, DOWNSAMPLE_SIZE)
    # return np.random.rand(32,DOWNSAMPLE_SIZE)

class DataUpdateThread(threading.Thread):
    def __init__(self, canvas, lines, channels_to_plot, shift):
        super().__init__()
        self.canvas = canvas
        self.lines = lines
        self.channels_to_plot = channels_to_plot
        self.shift = shift
        self.daemon = True  # 设置为守护线程，当主线程结束时，线程也会结束
        self.old_eeg_data = np.zeros((32, DOWNSAMPLE_SIZE))
        self.window_data = np.zeros((32, DOWNSAMPLE_SIZE))

    def run(self):
        while True:
            # 更新数据
            self.update()
            # 暂停一段时间，以便模拟数据采集
            time.sleep(0.1)

    def update(self):
        new_eeg_data = get_eeg_data()
        rolled_old_eeg_data = np.roll(self.old_eeg_data, -self.shift, axis=1)
        window_data = np.hstack((rolled_old_eeg_data[:, :-self.shift], new_eeg_data[:,:self.shift]))
        self.old_eeg_data = window_data
        for idx, channel in enumerate(self.channels_to_plot):
            self.lines[idx].set_data(np.arange(DOWNSAMPLE_SIZE), window_data[channel])
            self.canvas.draw_idle() 