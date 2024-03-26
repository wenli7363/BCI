from PyQt5.QtCore import QObject, pyqtSignal
import logging

class Logger(QObject):
    log_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.logger = logging.getLogger('GUI_Logger')
        self.logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter('【%(asctime)s】 %(message)s', datefmt='%Y/%m/%d %H:%M:%S')

        # 创建一个Handler来发射日志信号
        handler = LogHandler(self)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log(self, message):
        self.logger.info(message)

class LogHandler(logging.Handler):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent

    def emit(self, record):
        msg = self.format(record)
        self.parent.log_signal.emit(msg)
