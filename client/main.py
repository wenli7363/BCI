import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, 
    QFrame, QTextEdit, QComboBox,
    QCheckBox, QLineEdit,QScrollArea,
)
import pyqtgraph as pg
import numpy as np


class EEGDataCollectionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 创建主布局
        main_layout = QVBoxLayout()

        # 添加标题
        title_label = QLabel("多通道脑电信号可视化区域")
        main_layout.addWidget(title_label)

        # 创建主区域布局
        main_area_layout = QHBoxLayout()

        # 左侧EEG数据可视化区域
        eeg_data_area = QFrame()
        eeg_data_area.setFrameStyle(QFrame.Box)
        eeg_data_layout = QVBoxLayout()
        eeg_data_label = QLabel("EEG数据可视化区域")
        eeg_data_layout.addWidget(eeg_data_label)
        
        # # 添加一个占位符Widget,使可视化区域更大
        # eeg_data_placeholder = QWidget()
        # eeg_data_placeholder.setMinimumSize(800, 600)  # 调整大小以适应32通道显示
        # eeg_data_layout.addWidget(eeg_data_placeholder)
        # eeg_data_area.setLayout(eeg_data_layout)
        # main_area_layout.addWidget(eeg_data_area, stretch=4)  # 设置拉伸因子为4

        # 添加PyQtGraph绘图区域
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('bottom', 'Time (s)')  # 设置x轴标签为时间
        self.plot_item = self.plot_widget.getPlotItem()
        self.plot_item.showGrid(True, True)  # 显示网格
        # self.plot_item.hideAxis('left')  # 隐藏左侧y轴

        # 创建32个绘图曲线对象
        self.curves = []
        for i in range(32):
            pen = pg.mkPen(color=(i, 32 * 1.3))  # 为每个曲线分配不同颜色
            curve = self.plot_item.plot(pen=pen)
            self.curves.append(curve)

        eeg_data_layout.addWidget(self.plot_widget)

        # 添加通道选择框
        channel_selection_layout = QVBoxLayout()
        self.channel_checkboxes = []

        # 使用滚动区域容纳所有复选框
        channel_scroll_area = QScrollArea()
        channel_scroll_area.setWidgetResizable(True)
        channel_widget = QWidget()
        channel_layout = QHBoxLayout()

        for i in range(32):
            checkbox = QCheckBox(f"ch{i+1}")
            checkbox.setChecked(True)  # 默认全部选中
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_channel_visibility(idx, state))
            self.channel_checkboxes.append(checkbox)
            channel_layout.addWidget(checkbox)

        channel_widget.setLayout(channel_layout)
        channel_scroll_area.setWidget(channel_widget)
        channel_selection_layout.addWidget(channel_scroll_area)

        eeg_data_layout.addLayout(channel_selection_layout)

        eeg_data_area.setLayout(eeg_data_layout)
        main_area_layout.addWidget(eeg_data_area, stretch=4)  # 设置拉伸因子为4

        # 右侧控制区域
        control_area = QFrame()
        control_area.setFrameStyle(QFrame.Box)
        control_area_layout = QVBoxLayout()

        # 串口配置区域
        serial_config_label = QLabel("设备连接状态：未连接")
        control_area_layout.addWidget(serial_config_label)

        # serial_config_button = QPushButton("可选串口设备")
        serial_config_combox = QComboBox()
        serial_config_combox.addItem("COM1")
        control_area_layout.addWidget(serial_config_combox)

        connect_button = QPushButton("连接设备")
        control_area_layout.addWidget(connect_button)

        disconnect_button = QPushButton("断开连接")
        control_area_layout.addWidget(disconnect_button)

        # 数据采集控制区域
        data_resolution_label = QLabel("数据采集控制区")
        control_area_layout.addWidget(data_resolution_label)

        resolution_layout = QHBoxLayout()
        two_mins_button = QPushButton("二分类")
        four_mins_button = QPushButton("四分类")
        resolution_layout.addWidget(two_mins_button)
        resolution_layout.addWidget(four_mins_button)
        control_area_layout.addLayout(resolution_layout)

        # 日志区域
        log_area_label = QLabel("日志区域")
        control_area_layout.addWidget(log_area_label)
        log_area = QTextEdit()
        log_area.setReadOnly(True)
        control_area_layout.addWidget(log_area)

        control_area.setLayout(control_area_layout)
        main_area_layout.addWidget(control_area, stretch=1)  # 设置拉伸因子为1

        main_layout.addLayout(main_area_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("EEG数据采集软件_CJY_V0.4")

    def toggle_channel_visibility(self, channel_index, state):
        """
        Toggle visibility of EEG channels based on checkbox state.
        """
        self.curves[channel_index].setVisible(state == 2)  # 2表示选中状态

if __name__ == '__main__':
    app = QApplication(sys.argv)
    eeg_ui = EEGDataCollectionUI()
    eeg_ui.show()
    sys.exit(app.exec_())

