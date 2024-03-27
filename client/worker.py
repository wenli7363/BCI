from PyQt5.QtCore import QThread, pyqtSignal

# 创建一个新的线程类
class WorkerThread(QThread):
    finished = pyqtSignal()
    
    def run(self):
        # 发射信号
        self.finished.emit()