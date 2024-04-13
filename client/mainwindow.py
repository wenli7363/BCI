from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, 
    QFrame, QTextEdit, QComboBox,
    QCheckBox, QLineEdit, QScrollArea,QSizePolicy
)
from PyQt5.QtCore import Qt,QSize
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from logger import Logger
import twoclass, fourclass
import numpy as np
import threading
from connect.EEGSerialPortManager import EEGSerialPortManager,SERIAL_PORT_NAME
from time import sleep

shift = 10
window_size = 250

class EEGDataCollectionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.initConnect()
        self.eeg_serial_port_manager = EEGSerialPortManager()

    def initUI(self):
        # 创建主布局
        self.main_layout = QVBoxLayout()

        # 添加标题
        self.title_label = QLabel("多通道脑电信号可视化区域")
        self.main_layout.addWidget(self.title_label)

        # 创建主区域布局
        self.main_area_layout = QHBoxLayout()

        # ========================================================================== 左侧EEG数据可视化区域
        self.eeg_data_area = QFrame()
        self.eeg_data_area.setFrameStyle(QFrame.Box)
        self.eeg_data_layout = QVBoxLayout()
        self.eeg_data_label = QLabel("EEG数据可视化区域")
        self.eeg_data_layout.addWidget(self.eeg_data_label)

        N = 6
        fig = plt.Figure(figsize=(6, N * 2))  # 6个子图，每个子图的高度是2英寸
        axes = [fig.add_subplot(N, 1, i+1) for i in range(N)]

        # 准备数据和绘制折线图
        for i, ax in enumerate(axes):  # N个子图
            # 生成一些数据
            x = np.linspace(0, 10, 100)
            y = np.sin(x * (i + 1))  # 每个子图使用不同的频率
            
            # 绘制折线图在对应的子图上
            ax.plot(x, y)

        # 调整子图布局，消除间隙
        fig.tight_layout(h_pad=0, w_pad=0.1)  # 调整布局，使得子图之间没有水平间隙
        fig.subplots_adjust(hspace=0, bottom=0.05, top=0.95)  # 调整布局，使得子图之间没有垂直间隙


        # 创建一个FigureCanvas，用于在Qt中显示matplotlib图形
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        self.visual_area = QVBoxLayout()
        self.visual_scroll_area = QScrollArea()
        self.visual_scroll_area.setWidgetResizable(True)
        self.visual_scroll_area.setWidget(self.canvas)
        self.visual_area.addWidget(self.visual_scroll_area)
        self.eeg_data_layout.addLayout(self.visual_area,stretch=8)

        # ============================================================================================================

        # 添加通道选择框
        self.channel_selection_layout = QVBoxLayout()
        self.channel_checkboxes = []

        # 使用滚动区域容纳所有复选框
        self.channel_scroll_area = QScrollArea()
        self.channel_scroll_area.setWidgetResizable(True)
        self.channel_widget = QWidget()
        self.channel_layout = QHBoxLayout()

        for i in range(32):
            checkbox = QCheckBox(f"ch{i+1}")
            checkbox.setChecked(True)  # 默认全部选中
            checkbox.stateChanged.connect(lambda state, idx=i: self.toggle_channel_visibility(idx, state))
            self.channel_checkboxes.append(checkbox)
            self.channel_layout.addWidget(checkbox)

        self.channel_widget.setLayout(self.channel_layout)
        self.channel_scroll_area.setWidget(self.channel_widget)
        self.channel_selection_layout.addWidget(self.channel_scroll_area)

        self.eeg_data_layout.addLayout(self.channel_selection_layout,stretch=1)

        self.eeg_data_area.setLayout(self.eeg_data_layout)
        self.main_area_layout.addWidget(self.eeg_data_area, stretch=3)  # 设置拉伸因子为4

        # 右侧控制区域
        self.control_area = QFrame()
        self.control_area.setFrameStyle(QFrame.Box)
        self.control_area_layout = QVBoxLayout()

        # 串口配置区域
        self.serial_config_label = QLabel("设备连接状态：未连接")
        self.control_area_layout.addWidget(self.serial_config_label)

        # serial_config_button = QPushButton("可选串口设备")
        self.serial_config_combox = QComboBox()
        self.serial_config_combox.addItem("COM1")
        self.serial_config_combox.addItem("COM2")
        self.serial_config_combox.addItem("COM3")
        self.control_area_layout.addWidget(self.serial_config_combox)

        self.connect_button = QPushButton("连接设备")
        self.control_area_layout.addWidget(self.connect_button)

        self.disconnect_button = QPushButton("断开连接")
        self.control_area_layout.addWidget(self.disconnect_button)

        # 数据采集控制区域
        self.data_resolution_label = QLabel("数据采集控制区")
        self.control_area_layout.addWidget(self.data_resolution_label)

        self.resolution_layout = QHBoxLayout()
        self.two_class_button = QPushButton("二分类")
        
        self.four_class_button = QPushButton("四分类")
        
        self.resolution_layout.addWidget(self.two_class_button)
        self.resolution_layout.addWidget(self.four_class_button)
        self.control_area_layout.addLayout(self.resolution_layout)

        # 日志区域
        self.log_area_label = QLabel("日志区域")
        self.control_area_layout.addWidget(self.log_area_label)
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.control_area_layout.addWidget(self.log_area)

        # 日志记录器
        self.logger = Logger()

        self.control_area.setLayout(self.control_area_layout)
        self.main_area_layout.addWidget(self.control_area, stretch=1)  # 设置拉伸因子为1

        self.main_layout.addLayout(self.main_area_layout)

        self.setLayout(self.main_layout)
        self.setWindowTitle("EEG数据采集软件")

    def toggle_channel_visibility(self, channel_index, state):
        """
        Toggle visibility of EEG channels based on checkbox state.
        """
        self.curves[channel_index].setVisible(state == 2)  # 2表示选中状态

    def initConnect(self):
        self.two_class_button.clicked.connect(self.on_2class_button_clicked)        # 二分类按钮点击事件
        self.four_class_button.clicked.connect(self.on_4class_button_clicked)       # 四分类按钮点击事件
        self.logger.log_signal.connect(lambda msg: self.log_area.append(msg))                       # 日志记录器信号连接到日志区域
        self.connect_button.clicked.connect(self.on_connect_button_clicked)                         # 串口连接按钮点击事件 
        self.disconnect_button.clicked.connect(self.on_disconnect_button_clicked)                   # 串口断开按钮点击事件


    "槽函数，用于处理串口连接按钮点击事件"
    def on_connect_button_clicked(self):
        # 尝试打开串口
        if self.eeg_serial_port_manager.open_serial_port():
            print(f"成功打开串口: {SERIAL_PORT_NAME}")
            
            # 配置串口以开始监听数据
            self.eeg_serial_port_manager.config_serial_port()
            self.eeg_serial_port_manager.request_data()
            print("发送命令成功")
            selected_port = self.serial_config_combox.currentText()
            self.logger.log("连接到设备:{}".format(selected_port))
            self.serial_config_label.setText("设备连接状态：已连接")
    
    def on_disconnect_button_clicked(self):
        self.logger.log("断开设备连接")

    def on_2class_button_clicked(self):
        self.logger.log("开始二分类数据采集")
        self.eeg_collection_window = twoclass.TwoClassUI()
        self.eeg_collection_window.show()

    
    def on_4class_button_clicked(self):
        self.logger.log("开始四分类数据采集")
        self.eeg_collection_window = fourclass.FourClassUI()
        self.eeg_collection_window.show()

    # 模拟获取脑电数据的函数
    def get_eeg_data(self):
        # 这里假设获取到的数据是随机的
        # return  -50 + (50 - (-50)) *np.random.rand(32, 125)
        return np.array(self.eeg_serial_port_manager.eeg_driver.get_eeg_data())