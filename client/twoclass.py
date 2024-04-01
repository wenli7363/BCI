import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,QLineEdit,QPushButton,QTextEdit,QHBoxLayout,QFileDialog
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QPixmap, QFont
from PyQt5.QtCore import Qt, QTimer, QRectF
import random
import winsound

class TwoClassUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        # self.showStart()
        self.total_trial = 0
        self.max_trials = 12
        self.num_trials = {'left':0,'right':0}
    
    # 初始化界面
    def initUI(self):
        # 设置窗口的标题和初始位置
        self.setWindowTitle('左右手二分类')
        self.setGeometry(100, 100, 400, 400) 

        # 创建一个 QWidget 作为中央部件
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        # 创建 QVBoxLayout 来组织布局
        layout = QVBoxLayout(central_widget)

        tip = "<div align='center'><b>说明</b></div><br />" \
              "<div align='left'>左右运动想象数据采集，每次测试从bee声开始，有2s集中注意，4s想象，2秒休息组成，一组实验总共持续8s</div>"
        font = QFont()
        font.setPointSize(15)

        self.tip_text = QLabel(tip, self)
        self.tip_text.setFont(font)
        self.tip_text.setWordWrap(True)
        self.tip_text.setGeometry(10, 10, 380, 180)

        layout.addWidget(self.tip_text)
        # layout.setSpacing(40)
        # 创建 QLabel 和 QLineEdit 输入框
        input_layout = QHBoxLayout()
        # input_layout.setSpacing(40)
        self.label1 = QLabel('这组采集多少组数据？（建议左右各10次trials）:', self)
        self.lineEdit1 = QLineEdit(self)
        input_layout.addWidget(self.label1)
        input_layout.addWidget(self.lineEdit1)
        layout.addLayout(input_layout)

        file_save_layout = QHBoxLayout()

        # 创建一个 QLabel 用于提示用户选择文件保存路径
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
        layout.addLayout(file_save_layout)
        

        # 创建一个按钮
        self.button = QPushButton('开始采集', self)
        layout.addWidget(self.button)

        self.button.clicked.connect(self.showInputs)

    def showInputs(self):
        # 当按钮被点击时，获取输入框的内容，并显示在标签上
        input1 = self.lineEdit1.text()
        # input2 = self.lineEdit2.text()

        self.label1.setText(f'You entered: {input1}')
        # self.label2.setText(f'You entered: {input2}')
        
    # def initUI(self):
    #     self.setGeometry(100, 100, 400, 400)
    #     self.setWindowTitle('Fixation Cross with Random Arrows')
    #     self.central_widget = QWidget()
    #     self.setCentralWidget(self.central_widget)
    #     self.layout = QVBoxLayout(self.central_widget)
        
    #     self.label = QLabel(self)
    #     self.label.setAlignment(Qt.AlignCenter)
    #     self.layout.addWidget(self.label)
        
    #     # Timer setup
    #     self.timer = QTimer(self)
    #     self.timer.timeout.connect(self.showRandomArrow)
    #     self.timer.start(8000)  # 8 seconds interval
        
    #     # self.showFixationCross()
        
    def showFixationCross(self):
        pixmap = QPixmap(400, 400)
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
        winsound.Beep(500,1000)
        
        
    def drawArrow(self, painter, direction):
        # Define the path for the arrow
        arrow_path = QPainterPath()
        center_x = painter.device().width() / 2
        center_y = painter.device().height() / 2
        
        painter.translate(center_x, center_y)
        
        if  direction == 'left':
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
        pixmap = QPixmap(400, 400)
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
        
        directions = ['left', 'right']
        direction = random.choice(directions)
        
        while(self.num_trials[direction] == self.max_trials):
            if(self.total_trial == self.max_trials * 2) :
                print("已经采集了{}次trails".format(self.total_trial))
                return
            direction = random.choice(directions)
            
        self.num_trials[direction] +=1
        self.total_trial += 1
        # Draw the arrow
        self.drawArrow(painter, direction)
        painter.end()
        
        self.label.setPixmap(pixmap)

    def drawText(self, painter, text):
        # Set the font and draw the text
        font = painter.font()
        font.setPointSize(24)  # Set the font size
        painter.setFont(font)
        painter.drawText(QRectF(0, 0, 400, 400), Qt.AlignCenter, text)

    def showImaginationPrompt(self):
        pixmap = QPixmap(400, 400)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setFont(QFont("Arial", 24))  # Set the font and size

        # Draw the text "开始想象"
        self.drawText(painter, "开始想象")
        painter.end()

        self.label.setPixmap(pixmap)

    def showRestMessage(self):
        pixmap = QPixmap(400, 400)
        pixmap.fill(Qt.white)
        painter = QPainter(pixmap)
        painter.setFont(QFont("Arial", 24))  # Set the font and size

        # Draw the text "休息"
        self.drawText(painter, "休息...")
        painter.end()

        self.label.setPixmap(pixmap)

    def dataCollect(self):
        self.initUI()

        # 您还需要实现 selectSavePath 方法来处理路径选择
    def selectSavePath(self):
        # 这里可以使用 QFileDialog 来打开文件选择对话框
        save_path = QFileDialog.getExistingDirectory(self, "选择保存路径", "~")  # "~" 代表用户的主目录
        if save_path:
            self.line_edit_file_save.setText(save_path)  # 将选择的路径显示在 QLineEdit 中