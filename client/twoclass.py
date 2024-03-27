from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QApplication
from PyQt5.QtCore import QTimer

class TwoClassUI(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.prepareTime = 5  # 准备阶段时间（秒）
        self.imageryTime = 5  # 运动想象时间（秒）
        self.restTime = 5     # 休息时间（秒）
        self.currentPhase = "准备"  # 当前阶段：准备、运动想象、休息
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.updateState)

    def initUI(self):
        self.layout = QVBoxLayout()
        self.instructionLabel = QLabel("按下开始按钮，准备运动想象")
        self.layout.addWidget(self.instructionLabel)
        self.startButton = QPushButton("开始")
        self.startButton.clicked.connect(self.startImageryProcess)
        self.layout.addWidget(self.startButton)
        self.setLayout(self.layout)
        self.setWindowTitle("脑电数据采集")

    def startImageryProcess(self):
        self.startButton.setDisabled(True)  # 开始后禁用开始按钮
        self.timer.start(1000)  # 设置定时器1000ms
        self.countdown = self.prepareTime
        self.instructionLabel.setText("准备阶段...")

    def updateState(self):
        self.countdown -= 1
        if self.countdown <= 0:
            if self.currentPhase == "准备":
                self.currentPhase = "运动想象"
                self.countdown = self.imageryTime
                self.instructionLabel.setText("开始运动想象！")
            elif self.currentPhase == "运动想象":
                self.currentPhase = "休息"
                self.countdown = self.restTime
                self.instructionLabel.setText("休息阶段...")
            elif self.currentPhase == "休息":
                self.currentPhase = "准备"
                self.countdown = self.prepareTime
                self.instructionLabel.setText("按下开始按钮，准备运动想象")
                self.timer.stop()  # 停止定时器
                self.startButton.setEnabled(True)  # 重新启用开始按钮
                return  # 退出函数，等待用户再次启动
        # 更新状态显示（可选）
        self.updateDisplay()

    def updateDisplay(self):
        # 此函数可以根据需要更新UI显示，例如显示倒计时
        pass
