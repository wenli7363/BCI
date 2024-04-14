from PyQt5.QtWidgets import QApplication
from mainwindow import EEGDataCollectionUI
import sys
import PyQt5.QtCore as QtCore

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setAttribute(QtCore.Qt.AA_Use96Dpi)
    mw = EEGDataCollectionUI()
    mw.show()
    sys.exit(app.exec_())
