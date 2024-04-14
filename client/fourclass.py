import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QPixmap
from PyQt5.QtCore import Qt, QTimer
import random
import winsound

class FourClassUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.total_trial = 0    # 总共的trials
        self.max_trials = 12    # 每个方向最多出现12次
        self.num_trials = {'up':0,'down':0,'left':0,'right':0}      # 统计每个方向的次数
        
    def initUI(self):
        self.setGeometry(100,100,400, 400)        # 窗口尺寸
        self.centerWindow()
        self.setWindowTitle('Fixation Cross with Random Arrows')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)
        
        # Timer setup
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showRandomArrow)    # 每次定时器超时，显示一个随机方向的箭头
        self.timer.start(3000)  # 8 seconds interval
        
        self.showFixationCross()
        
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
        
    def drawArrow(self, painter, direction):
        # 在标题中显示这是第几组实验
        self.setWindowTitle('Trial {}/{}'.format(self.total_trial, self.max_trials * 4))
        # Define the path for the arrow
        arrow_path = QPainterPath()
        center_x = painter.device().width() / 2
        center_y = painter.device().height() / 2
        
        painter.translate(center_x, center_y)  # 平移坐标系到中心点
        
        if direction == 'up':
            arrow_path.moveTo(0, -150)
            arrow_path.lineTo(-50, -50)
            arrow_path.lineTo(50, -50)
        elif direction == 'down':
            arrow_path.moveTo(0, 150)
            arrow_path.lineTo(-50, 50)
            arrow_path.lineTo(50, 50)
        elif direction == 'left':
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
        
        directions = ['up', 'down', 'left', 'right']
        direction = random.choice(directions)       # 随机选择一个方向
        
        # 如果某个方向的次数已经达到最大次数，则重新选择方向
        while(self.num_trials[direction] == self.max_trials):
            if(self.total_trial == self.max_trials * 4) :
                print("已经采集了{}次trails".format(self.total_trial))
                return
            direction = random.choice(directions)
            
        self.num_trials[direction] +=1
        self.total_trial += 1
        # Draw the arrow
        self.drawArrow(painter, direction)
        painter.end()
        
        self.label.setPixmap(pixmap)
    
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