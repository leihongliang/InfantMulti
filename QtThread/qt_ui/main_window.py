# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.10.1
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 596)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.frame)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.btn_input = QtWidgets.QPushButton(self.frame)
        self.btn_input.setMinimumSize(QtCore.QSize(80, 30))
        self.btn_input.setMaximumSize(QtCore.QSize(80, 30))
        self.btn_input.setObjectName("btn_input")
        self.gridLayout_2.addWidget(self.btn_input, 1, 0, 1, 1)
        self.label_video = QtWidgets.QLabel(self.frame)
        self.label_video.setMinimumSize(QtCore.QSize(760, 480))
        self.label_video.setMaximumSize(QtCore.QSize(760, 480))
        self.label_video.setObjectName("label_video")
        self.gridLayout_2.addWidget(self.label_video, 0, 0, 1, 3)
        self.btn_pause = QtWidgets.QPushButton(self.frame)
        self.btn_pause.setMinimumSize(QtCore.QSize(80, 30))
        self.btn_pause.setMaximumSize(QtCore.QSize(80, 30))
        self.btn_pause.setObjectName("btn_pause")
        self.gridLayout_2.addWidget(self.btn_pause, 1, 1, 1, 1)
        self.gridLayout.addWidget(self.frame, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.btn_input.setText(_translate("MainWindow", "导入视频文件"))
        self.label_video.setText(_translate("MainWindow", "TextLabel"))
        self.btn_pause.setText(_translate("MainWindow", "暂停"))

