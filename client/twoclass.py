from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,QLineEdit,QPushButton,
                             QHBoxLayout,QFileDialog,QMessageBox)
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer,pyqtSignal,QEvent
import random
import winsound
from constVar import LABEL_MAP
import time


class TwoClassUI(QMainWindow):
    start_save_eeg_data_signal = pyqtSignal()
    stop_save_eeg_data_signal = pyqtSignal(bool,int,str,str)  # 是否提前停止，本次实验的标签，保存路径
    collect_finished_signal = pyqtSignal(str)


    def __init__(self):
        super().__init__()
        self.initUI()
        self.total_trial = 0    # 已经执行的总共trials数
        self.max_trials = 1   # 每个方向最多出现12次
        self.num_trials = {'left':0,'right':0}      # 统计每个方向的次数
        self.save_path = None
        self.fileName = None
        self.timer = QTimer()      # 控制实验流程的定时器
        self.flag = None            # 标记本次实验的方向
        self.stop_advance = False   # 是否提前停止实验
        self.window_geometry = self.geometry()
    
    def __del__(self):
        print("删除二分类的窗口")
        self.timer.stop()
        self.stopCollect()
        self.timer.deleteLater()
        self.deleteLater()

    def initUI(self):
        # 设置窗口的标题和初始位置
        self.setWindowTitle('二分类')
        self.setGeometry(100, 100, 400, 400) 
        self.centerWindow()
        # 创建一个 QWidget 作为中央部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建 QVBoxLayout 来组织布局
        layout = QVBoxLayout(central_widget)

        """
        提示UI
        """
        tip = "<div align='center'><b>说明</b></div><br />" \
              "<div align='left'>左右运动想象数据采集，每次测试从bee声开始，有2s集中注意，4s想象，2秒休息组成，一组实验总共持续8s</div>"
        font = QFont()
        font.setPointSize(14)

        self.tip_text = QLabel(tip, self)
        self.tip_text.setFont(font)
        self.tip_text.setWordWrap(True)
        self.tip_text.setGeometry(10, 10, 300, 150)

        layout.addWidget(self.tip_text,stretch=5)

        """
        每组试验次数输入框
        """
        # 创建 QLabel 和 QLineEdit 输入框
        num_trials_input_layout = QHBoxLayout()
        self.num_trials_label = QLabel('这组采集多少组数据?(建议每组各10次trials):', self)
        self.num_trails_lineEdit = QLineEdit(self)
        num_trials_input_layout.addWidget(self.num_trials_label)
        num_trials_input_layout.addWidget(self.num_trails_lineEdit)
        layout.addLayout(num_trials_input_layout,stretch=2)
        
        """
        保存路径UI
        """
        # 创建一个 QLabel 用于提示用户选择文件保存路径
        file_save_layout = QHBoxLayout()
        self.label_file_save = QLabel('保存路径:', self)
        file_save_layout.addWidget(self.label_file_save)

        # 创建一个 QLineEdit 用于显示和编辑文件路径
        self.line_edit_file_save = QLineEdit(self)
        self.line_edit_file_save.setReadOnly(True)  # 设置为只读，因为路径是通过按钮选择的
        file_save_layout.addWidget(self.line_edit_file_save)

        # 创建一个 QPushButton 用于打开文件选择对话框
        self.button_file_save = QPushButton('浏览...', self)
        self.button_file_save.clicked.connect(self.selectSavePath)  # 连接按钮点击事件到 selectSavePath 方法
        file_save_layout.addWidget(self.button_file_save)

        # 将新的文件保存布局添加到主布局中
        layout.addLayout(file_save_layout, stretch=2)
        
        """
        文件名
        """
        file_name_input_layout = QHBoxLayout()
        self.file_name_label = QLabel('文 件 名:', self)
        self.file_name_lineEdit = QLineEdit(self)
        self.tip_label = QLabel('(不要有特殊字符)', self)
        file_name_input_layout.addWidget(self.file_name_label)
        file_name_input_layout.addWidget(self.file_name_lineEdit)
        file_name_input_layout.addWidget(self.tip_label)
        layout.addLayout(file_name_input_layout, stretch=1)
        
        """
        按钮 
        """
        # 创建一个按钮
        self.button = QPushButton('开始采集', self)
        layout.addWidget(self.button)

        # 点击按钮的槽函数
        self.button.clicked.connect(self.startTrial)
        # 窗口拖动的事件过滤
        self.installEventFilter(self)

    def showBreakUI(self):
        print("=====================================")
        print("[breakUI]:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if self.stop_advance == True:
            self.stopCollect()
            print("提前结束")

        self.timer.singleShot(3000, self.showFixationCross)
        
        # self.setGeometry(100, 100, 400, 400)        # 窗口尺寸
        self.setGeometry(self.window_geometry)
        self.setWindowTitle('休息时间')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 添加 "休息时间" 文字
        self.label = QLabel("休息时间", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 24))
        self.layout.addWidget(self.label)

    def showFixationCross(self):
        print("[showFixationCross]:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if self.stop_advance == True:
            self.stopCollect()
            print("提前结束")

        self.timer.singleShot(2000, self.showRandomArrow)

        pixmap = QPixmap(self.label.width(), self.label.height())
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        pen = QPen(Qt.black, 5)
        painter.setPen(pen)
        
        # Draw the fixation cross
        center_x = pixmap.width() / 2
        center_y = pixmap.height() / 2
        painter.drawLine(center_x, center_y - 30, center_x, center_y + 30)
        painter.drawLine(center_x - 30, center_y, center_x + 30, center_y)
        painter.end()
        
        self.label.setPixmap(pixmap)
        winsound.Beep(500,500)

        
    def drawArrow(self, painter, direction):
        # 在标题中显示这是第几组实验
        self.setWindowTitle('Trial {}/{}'.format(self.total_trial+1, self.max_trials * 2))
        # Define the path for the arrow
        arrow_path = QPainterPath()
        center_x = painter.device().width() / 2
        center_y = painter.device().height() / 2
        
        painter.translate(center_x, center_y)  # 平移坐标系到中心点
        
        if direction == 'left':
            arrow_path.moveTo(-150, 0)
            arrow_path.lineTo(-50, -50)
            arrow_path.lineTo(-50, 50)
        elif direction == 'right':
            arrow_path.moveTo(150, 0)
            arrow_path.lineTo(50, -50)
            arrow_path.lineTo(50, 50)
        arrow_path.closeSubpath()
        
        # Draw the arrow
        painter.drawPath(arrow_path)
        
    def showRandomArrow(self):
        print("[cue]:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if self.stop_advance == True:
            self.stopCollect()
            print("提前结束")

        # cue阶段读数据
        self.startCollect()
        # 1秒后显示想象
        self.timer.singleShot(1000, self.showMotorImagery)

        pixmap = QPixmap(self.label.width(), self.label.height())
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        pen = QPen(Qt.black, 5)
        painter.setPen(pen)
        
        # Draw the fixation cross
        center_x = pixmap.width() / 2
        center_y = pixmap.height() / 2
        painter.drawLine(center_x, center_y - 30, center_x, center_y + 30)
        painter.drawLine(center_x - 30, center_y, center_x + 30, center_y)
        
        # Randomly choose a direction for the arrow
        
        directions = [ 'left', 'right']
        direction = random.choice(directions)       # 随机选择一个方向
        
        # 如果某个方向的次数已经达到最大次数，则重新选择方向
        while(self.num_trials[direction] == self.max_trials):
            if(self.total_trial == self.max_trials * 2) :
                print("已经采集了{}次trails".format(self.total_trial))
                return
            direction = random.choice(directions)
        
        self.flag = LABEL_MAP[direction]       # 标记一下本次实验的方向
        self.num_trials[direction] +=1
        
        # Draw the arrow
        self.drawArrow(painter, direction)
        painter.end()
        
        self.label.setPixmap(pixmap)
        
    
    # 将窗口放到显示器中间
    def centerWindow(self):
        # 获取显示器的可用空间
        screen = QApplication.primaryScreen()
        screen_geometry = screen.availableGeometry()
        
        # 获取窗口的尺寸
        window_geometry = self.geometry()
        window_width = window_geometry.width()
        window_height = window_geometry.height()
        
        # 计算窗口居中的坐标
        center_x = screen_geometry.width() // 2 - window_width // 2
        center_y = screen_geometry.height() // 2 - window_height // 2
        
        # 设置窗口位置
        self.setGeometry(center_x, center_y, window_width, window_height)

    def runExperiment(self):
        if(self.total_trial != 0) : self.stopCollect()
        if self.stop_advance == True:
            self.stopCollect()
            print("提前结束")
            return
        if self.total_trial >= self.max_trials * 2:
            self.collect_finished_signal.emit(self.save_path)
            self.allTrialsEnd(True)
            return
        self.showBreakUI()
        

    
    def selectSavePath(self):
        # 这里可以使用 QFileDialog 来打开文件选择对话框
        save_path = QFileDialog.getExistingDirectory(self, "选择保存路径", "~")  # "~" 代表用户的主目录
        if save_path:
            self.line_edit_file_save.setText(save_path)  # 将选择的路径显示在 QLineEdit 中

    # 开始实验
    def startTrial(self):
        # 如果没有保存文件路径，弹出消息窗提示用户选择路径
        if self.line_edit_file_save.text()=="":
            msg_box = QMessageBox()
            msg_box.setText("请先选择保存路径！！")
            msg_box.setWindowTitle("提示")
            msg_box.setIcon(QMessageBox.Information)
            msg_box.exec_()
            return

        if self.num_trails_lineEdit.text() != "":
            self.max_trials = int(self.num_trails_lineEdit.text())
        else: 
            self.max_trials = 12
        self.save_path = self.line_edit_file_save.text()

        if self.file_name_lineEdit.text() == "":
            self.fileName = "twoclass_data"
        else:
            self.fileName = self.file_name_lineEdit.text()
        
        self.runExperiment()
    
    def showMotorImagery(self):
        print("[showMotorImagery]:",time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
        if self.stop_advance == True:
            self.stopCollect()
            print("提前结束")

        self.timer.singleShot(4000, self.runExperiment)

        self.setGeometry(self.window_geometry)
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        # 添加 "休息时间" 文字
        self.label = QLabel("请想象...", self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setFont(QFont("Arial", 24))
        self.layout.addWidget(self.label)

        # 进行下一组实验
        
        self.total_trial += 1
        

    
    # 关闭窗口时，停止定时器
    def closeEvent(self, event):
        # 在这里停止定时器并执行其他清理任务
        print("关闭二分类的窗口")
        if self.total_trial >= self.max_trials * 2:
            self.stop_advance = False
        else:
            self.stop_advance = True
        self.timer.stop()
        
        self.window_geometry = self.geometry()
        # self.stopCollect()
        # 执行其他清理任务...
        event.accept()

    # 开始采集
    def startCollect(self):
        print("子窗口：发送一次开始start_save_eeg_data_signal")
        self.start_save_eeg_data_signal.emit()
    
    # 停止采集
    def stopCollect(self):
        print("子窗口：发送一次停止stop_save_eeg_data_signal")
        self.stop_save_eeg_data_signal.emit(self.stop_advance,self.flag,self.save_path,self.fileName)     # 发送本次实验的标签
    
 
    def allTrialsEnd(self,flag):
        task_end_msg = QMessageBox()
        task_end_msg.buttonClicked.connect(lambda: self.close())
        if flag:
            task_end_msg.setText("实验成功结束,可以关闭当前窗口了")
        else:
            task_end_msg.setText("实验提前结束,出错,数据不保存")
        task_end_msg.setWindowTitle("提示")
        task_end_msg.setIcon(QMessageBox.Information)
        task_end_msg.exec_()
        return
    
    def eventFilter(self, obj, event):
        if event.type() == QEvent.WindowStateChange or event.type() == QEvent.Move:
            self.window_geometry = self.geometry()  # 更新窗口位置和大小
        return super().eventFilter(obj, event)