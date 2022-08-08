from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from InfantUI.Infant import Ui_MainWindow
import sys
import cv2
import os
import time
from InfantUI.inference_0 import model_load, Dection


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.cap_video = None
        self.image = None
        self.device = None
        self.model = None
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screen_height = self.screenRect.height()
        self.screen_width = self.screenRect.width()
        self.setupUi(self)
        self.qssFileName = "qss/mainWin.qss"
        self.qss = QSSLoad()
        self.qssFile = self.qss.readQssFile(self.qssFileName)
        self.setStyleSheet(self.qssFile)
        self.setWindowIcon(QIcon("./qt_ui/a2x1a-0p5yz-001.ico"))

        # 按钮事件绑定
        self.btn_quit.clicked.connect(QApplication.quit)
        self.btn_input.clicked.connect(self.show_video)
        self.btn_pause.clicked.connect(self.pause_video)
        self.btn_pause.setEnabled(False)
        self.init_work()

        self.timer_camera1 = QtCore.QTimer()
        self.timer_camera2 = QtCore.QTimer()
        self.timer_camera3 = QtCore.QTimer()
        self.timer_video = QtCore.QTimer()

        # self.timer_video.timeout.connect(self.show_video)

    def init_work(self):
        self.model, self.device = model_load('3DResnet')

    #   视频导入功能
    # def load_path(self):
    #     if not self.timer_video.isActive():
    #         # imgName, imgType = QFileDialog.getOpenFileName(self, "打开视频", "", "*.mp4;;*.AVI;;*.rmvb;;All Files(*)")
    #         # self.cap_video = cv2.VideoCapture(imgName)
    #         # flag = self.cap_video.isOpened()
    #         # if not flag:
    #         #     msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
    #         #                                         buttons=QtWidgets.QMessageBox.Ok,
    #         #                                         defaultButton=QtWidgets.QMessageBox.Ok)
    #         # else:
    #             # self.timer_camera3.start(30)
    #             print('1')
    #             # self.timer_video.stop()
    #             # self.cap_video.release()
    #             self.show_video()
    #             # self.label_video.clear()
    #             # self.open_video.setText(u'关闭视频')

    def show_video(self):
        print('2')
        # flag, self.image = self.cap_video.read()
        # if not flag:
        #     msg = QtWidgets.QMessageBox.warning(self, u"Warning", u"请检测相机与电脑是否连接正确",
        #                                         buttons=QtWidgets.QMessageBox.Ok,
        #                                         defaultButton=QtWidgets.QMessageBox.Ok)
        #
        # else:
        video = 'E:\HDU\Project\InfantC3D\data\demo_video.mp4'
        cap = cv2.VideoCapture(video)
        # cap = cv2.VideoCapture(self.cap_video)
        clip = []
        retaining = True
        flag = True
        while flag:
            print('3')
            frame = Dection(self.model, cap, retaining, self.device, clip)
            frame = cv2.resize(frame, (760, 428), cv2.INTER_LINEAR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
            self.label_video.setPixmap(QtGui.QPixmap.fromImage(showImage))
            time.sleep(0.05)
        print('4')

    #   暂停/播放功能
    def pause_video(self):
        if self.btn_pause.text() == "暂停":
            self.btn_pause.setText("播放")
            self.playwork.playFlag = 0
        else:
            self.btn_pause.setText("暂停")
            self.playwork.playFlag = 1

    def closeEvent(self, event):
        print("关闭线程")
        # # Qt需要先退出循环才能关闭线程
        # if self.decodework.isRunning():
        #     self.decodework.threadFlag = 0
        #     self.decodework.quit()
        # # if self.playwork.isRunning():
        # #     self.playwork.threadFlag = 0
        # #     self.playwork.quit()
        # # self.playwork.threadFlag = 0
        # # self.pool.clear()
        #
        # if self.play_thread.isRunning():
        #     self.playwork.threadFlag = 0
        #     self.play_thread.quit()


class QSSLoad:
    @staticmethod
    def readQssFile(qssFileName):
        with open(qssFileName, 'r', encoding='UTF-8') as file:
            return file.read()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
