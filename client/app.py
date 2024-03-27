from PyQt5.QtWidgets import QApplication
from mainwindow import EEGDataCollectionUI
import sys

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mw = EEGDataCollectionUI()
    mw.show()
    sys.exit(app.exec_())
