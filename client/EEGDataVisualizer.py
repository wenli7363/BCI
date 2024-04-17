import pyqtgraph as pg
from PyQt5.QtWidgets import QWidget, QVBoxLayout,QScrollArea,QSizePolicy
import numpy as np
from constVar import DOWNSAMPLE_SIZE
import pdb


class EEGDataVisualizer(QWidget):
    def __init__(self, channels_to_plot=None, parent=None):
        super().__init__(parent)
        if channels_to_plot is None:
            self.channels_to_plot = list(range(32))
        else:
            self.channels_to_plot = channels_to_plot

        self.initUI()


    def initUI(self):
        layout = QVBoxLayout()
        self.setLayout(layout)

        # 创建 PyQtGraph 的绘图区域
        self.plot_layout = pg.GraphicsLayoutWidget()
        self.plot_layout.setFixedHeight(3000)  # 设置固定高度


        self.plot_scroll_area = QScrollArea()
        self.plot_scroll_area.setWidget(self.plot_layout)
        self.plot_scroll_area.setWidgetResizable(True)
        # self.plot_scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        # self.plot_scroll_area.setFixedHeight(600)  # 设置固定高度

        layout.addWidget(self.plot_scroll_area)

        # 创建多个 PlotItem,每个对应一个通道
        self.plot_items = []
        for i in self.channels_to_plot:
            self.plot_layout.nextRow()
            plot_item = self.plot_layout.addPlot()
            plot_item.setFixedHeight(80)  # 设置每个 PlotItem 的高度为 50
            plot_item.setMouseEnabled(x=False, y=False)  # 禁用鼠标交互
            plot_item.setTitle({f'Channel {i+1}'})
            plot_item.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            # plot_item.showAxis('bottom', False)                         # 是否显示坐标轴
            if i != self.channels_to_plot[-1]:
                plot_item.getAxis('bottom').setStyle(showValues=False)
            plot_item.getAxis('left').setStyle(showValues=False)        # 是否显示刻度
            self.plot_items.append(plot_item)

        # 初始化曲线
        self.plots = []
        for plot_item in self.plot_items:
            plot = plot_item.plot(pen=pg.intColor(len(self.plots), len(self.channels_to_plot)))
            self.plots.append(plot)

        # 设置 x 轴范围
        for plot_item in self.plot_items:
            plot_item.setXRange(0, DOWNSAMPLE_SIZE)

    # def update_eeg_data(self, eeg_data):
    #     """
    #     更新 EEG 数据并刷新图形
    #     """
       
    #     for i, channel in enumerate(self.channels_to_plot):
    #         # if(i==9): print(eeg_data[channel].shape)
    #         self.plots[i].setData(np.arange(DOWNSAMPLE_SIZE), eeg_data[channel])

    #         self.plot_layout.update()

    def update_eeg_data(self, eeg_data):
        """
        更新 EEG 数据并刷新图形
        """
        try:
            for i, channel in enumerate(self.channels_to_plot):
                self.plots[i].setData(np.arange(DOWNSAMPLE_SIZE), eeg_data[channel])

            self.plot_layout.update()
        except Exception as e:
            raise e 
    
    def reset(self):
        self.update_eeg_data(np.zeros((32, DOWNSAMPLE_SIZE)))