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
        self.total_trial = 0
        self.max_trials = 12
        self.num_trials = {'up':0,'down':0,'left':0,'right':0}
        
    def initUI(self):
        self.setGeometry(100, 100, 400, 400)
        self.setWindowTitle('Fixation Cross with Random Arrows')
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)
        
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)
        
        # Timer setup
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.showRandomArrow)
        self.timer.start(8000)  # 8 seconds interval
        
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
        # Define the path for the arrow
        arrow_path = QPainterPath()
        center_x = painter.device().width() / 2
        center_y = painter.device().height() / 2
        
        painter.translate(center_x, center_y)
        
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
        direction = random.choice(directions)
        
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