from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, 
    QHBoxLayout, QPushButton, QLabel, 
    QFrame, QTextEdit, QComboBox,
    QCheckBox, QScrollArea,QSizePolicy
)
from PyQt5.QtCore import QTimer
from logger import Logger
import twoclass, fourclass
import numpy as np
from connect.EEGSerialPortManager import EEGSerialPortManager,SERIAL_PORT_NAME
from time import sleep
from EEGDataVisualizer import EEGDataVisualizer
from constVar import DOWNSAMPLE_SIZE,CHANEL_NUM
from SaveData import saveData
import time

shift = 40
old_eeg_data = np.zeros((CHANEL_NUM, DOWNSAMPLE_SIZE))
lines = []
channels_to_plot = [i for i in range(CHANEL_NUM)]
eeg_data_per_trial_buffer = []      # 用于存储每次采集的数据

class EEGDataCollectionUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.data_update_timer = QTimer()           # 获取新数据的定时器
        self.data_saver_timer = QTimer()            # 保存数据的定时器
        self.eeg_serial_port_manager = EEGSerialPortManager()
        self.writeBufferNum = 0

        self.initConnect()
        

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

        
        self.eeg_data_visualizer = EEGDataVisualizer(channels_to_plot=channels_to_plot)
        self.eeg_data_visualizer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.eeg_data_layout.addWidget(self.eeg_data_visualizer,stretch=8)

        # ============================================================================================================

        # 添加通道选择框
        self.channel_selection_layout = QVBoxLayout()
        self.channel_checkboxes = []

        # 使用滚动区域容纳所有复选框
        self.channel_scroll_area = QScrollArea()
        self.channel_scroll_area.setWidgetResizable(True)
        self.channel_widget = QWidget()
        self.channel_layout = QHBoxLayout()
        
        # 每个通道对应一个复选框
        for i in range(CHANEL_NUM):
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
        # self.axes[channel_index].set_visible(state == 2)
        self.eeg_data_visualizer.set_visibility(channel_index, state == 2)

    def initConnect(self):
        self.two_class_button.clicked.connect(self.on_2class_button_clicked)        # 二分类按钮点击事件
        self.four_class_button.clicked.connect(self.on_4class_button_clicked)       # 四分类按钮点击事件
        self.logger.log_signal.connect(lambda msg: self.log_area.append(msg))                       # 日志记录器信号连接到日志区域
        self.connect_button.clicked.connect(self.on_connect_button_clicked)                         # 串口连接按钮点击事件 
        self.disconnect_button.clicked.connect(self.on_disconnect_button_clicked)                   # 串口断开按钮点击事件
        self.data_update_timer.timeout.connect(self.update_eeg_data)                               # 定时更新EEG数据
        self.data_saver_timer.timeout.connect(self.eegdata_buffer_add)

    "槽函数，用于处理串口连接按钮点击事件"
    def on_connect_button_clicked(self):        
        # self.data_update_timer.start(100)

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
            # 启动计时器，定时读数据
        self.data_update_timer.start(100)
    
    def on_disconnect_button_clicked(self):
        self.logger.log("断开设备连接")
        self.data_update_timer.stop()
        self.eeg_serial_port_manager.close_serial_port()
        self.serial_config_label.setText("设备连接状态：未连接")
        self.eeg_data_visualizer.reset()

    def on_2class_button_clicked(self):
        self.logger.log("开始二分类数据采集")
        self.eeg_collection_window = twoclass.TwoClassUI()
        self.eeg_collection_window.start_save_eeg_data_signal.connect(self.start_save_timer)
        self.eeg_collection_window.stop_save_eeg_data_signal.connect(self.stop_save_timer)
        self.eeg_collection_window.collect_finished_signal.connect(lambda path : self.logger.log("完成二分类数据采集，保存到："+path))
        self.eeg_collection_window.show()

    
    def on_4class_button_clicked(self):
        self.logger.log("开始四分类数据采集")
        self.eeg_collection_window = fourclass.FourClassUI()
        self.eeg_collection_window.start_save_eeg_data_signal.connect(self.start_save_timer)
        self.eeg_collection_window.stop_save_eeg_data_signal.connect(self.stop_save_timer)
        self.eeg_collection_window.collect_finished_signal.connect(lambda path : self.logger.log("完成四分类数据采集，保存到："+path))
        self.eeg_collection_window.show()

    def update_eeg_data(self):
        """
        处理新数据，传递给eeg可视化对象进行折线图更新
        """
        global old_eeg_data
        new_eeg_data = self.get_eeg_data()
        rolled_old_eeg_data = np.roll(old_eeg_data, -shift, axis=1)
        window_data = np.hstack((rolled_old_eeg_data[:, :-shift], new_eeg_data[:,:shift]))
        old_eeg_data = window_data
        self.eeg_data_visualizer.update_eeg_data(window_data)

    def get_eeg_data(self):
        return -50 + (50 - (-50)) *np.random.rand(CHANEL_NUM, DOWNSAMPLE_SIZE)
        # return np.array(self.eeg_serial_port_manager.eeg_driver.get_eeg_data())

    def start_save_timer(self):
        # print("主窗口：启动保存数据定时器:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        self.writeBufferNum = 0
        if not self.data_saver_timer.isActive():
            self.data_saver_timer.start(1000)    # 每隔100ms读一次数据
            
    
    # 向buffer列表中添加数据
    def eegdata_buffer_add(self):
        global eeg_data_per_trial_buffer
        self.writeBufferNum += 1

        # 根据实验的设置，保证只有4s的数据，不然可能因为时钟的不同步问题，出现5s数据
        if (self.writeBufferNum <= 4):
            eeg_data_per_trial_buffer.append(self.get_eeg_data())

    def stop_save_timer(self,stop_advance,flag,save_path,fileName):
        global eeg_data_per_trial_buffer
        if self.data_saver_timer.isActive():
           self.data_saver_timer.stop()

        buffer_concatenate = np.concatenate(eeg_data_per_trial_buffer,axis=1)   # 将buffer中的数据拼接起来
        eeg_data_per_trial_buffer = []
        # 处理buffer并保存
        print("主窗口：收到停止保存数据信号：",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        print("主窗口：停止保存数据定时器，本次的label为：",flag)
        print("主窗口：本轮写了{}次数据".format(self.writeBufferNum))
        
        # 如果不是提前停止采集，就保存数据
        if stop_advance == False:
            saveData(buffer_concatenate,flag,save_path,fileName)
        else:
            print("提前停止采集，不保存数据")