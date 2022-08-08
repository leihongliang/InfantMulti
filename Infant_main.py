#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
import time

import cv2
import numpy as np

import re
import os
from PIL import Image
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog, QGraphicsPixmapItem, QGraphicsScene, QMainWindow

from Infant_liudk import Ui_MainWindow
from QtThread.inference_gui import InfantDetection_Thread


class infantMain(QMainWindow, Ui_MainWindow):

    def __init__(self):
        super(infantMain, self).__init__()

        self.get_screen_width_and_height()
        self.setupUi(self)

        self.real_time_camera.clicked.connect(self.open_camera)
        self.load_video.clicked.connect(self.load_infant_video)
        self.export_log.clicked.connect(self.export_detection_log)
        self.quit_system.clicked.connect(QCoreApplication.quit)

    def QLabel_Show_Image(self, Control, imgPath=None, image=None):  # control为显示该内容的控件名称
        img = None
        # 预处理图像,对小于QLable控件宽高的图像进行填充,从而保证其自适应缩放过程中不会发生形变
        if imgPath is not None and image is None:
            img = cv2.imread(imgPath)
        elif imgPath is None and image is not None:
            img = image

        img_h, img_w, img_channel = img.shape[0], img.shape[1], img.shape[2]

        top, bottom, left, right = (0, 0, 0, 0)         # 对于图像上下左右四条边的填充像素
        if img_h < img_w * (Control.height() / Control.width()):
            # 对称式填充图像的高,使其与控件的高相同
            fill_pix_h = int((img_w * (Control.height() / Control.width()) - img_h) / 2)
            top = bottom = fill_pix_h
        if img_w < img_h * (Control.width() / Control.height()):
            # 对称式填充图像的宽,使其与控件的宽相同
            fill_pix_w = int((img_h * (Control.width() / Control.height()) - img_w) / 2)
            left = right = fill_pix_w

        res = cv2.cvtColor(cv2.copyMakeBorder(img, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(255, 255, 255)), cv2.COLOR_BGR2RGB)
        R, G, B = cv2.split(res)

        Alpha = np.ones(R.shape, dtype=R.dtype) * 255

        Alpha[:top, :] = Alpha[top:, :left] =\
            Alpha[res.shape[0] - bottom:, left:] = Alpha[top:res.shape[0] - bottom, res.shape[1] - right:] = 0

        res_ = cv2.merge((R, G, B, Alpha))

        pix = QtGui.QPixmap(QtGui.QImage(res_.data, res_.shape[1], res_.shape[0], QtGui.QImage.Format_RGBA8888))
        Control.setPixmap(pix)
        Control.setScaledContents(True)

    def open_camera(self):
        # TODO 打开摄像头,将画面内容实时传递到主界面(其实回传的应该是模型推理输出的画面)
        print("1")

    def load_infant_video(self):
        # TODO 加载拍摄的画面内容,将画面内容实时传递到主界面(其实回传的应该是模型推理输出的画面)
        self.videoName, imgType = QFileDialog.getOpenFileNames(self.centralwidget,
                                                               "打开文件",
                                                               "*.mp4;; *.avi;; *.mpeg;; All Files(*)")

        self.cap = cv2.VideoCapture(self.videoName[0])
        retaining = True
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

        # while retaining:
        # retaining, frame = cap.read()
        # if not retaining and frame is None:
        #     continue

        print(self.cap.isOpened())
        # while cap.isOpened():
        self.retaining, frame = self.cap.read()
        self.frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        print(self.frame)
        pix = QtGui.QPixmap(QtGui.QImage(self.frame.data, self.frame.shape[1], self.frame.shape[0], QtGui.QImage.Format_RGB888))
        self.video_show.setPixmap(pix)
        self.video_show.setScaledContents(True)

    #     self.InfantDetectionModel = InfantDetection_Thread(self.frame)
    #     self.InfantDetectionModel.finish_Single.connect(self.InfantDetectionResultsShow)
    #
    # def InfantDetectionResultsShow(self, class_result, prob_result):
    #     # if not self.retaining and self.frame is None:
    #     #     self.InfantDetectionModel.terminate()
    #     self.QLabel_Show_Image(self.video_show, image=Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)))
    #     cv2.putText(self.frame, class_result, (20, 100),  cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)
    #     cv2.putText(self.frame, "prob: %.4f" % prob_result, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)
    #     # self.QLabel_Show_Image(self.video_show, image=Image.fromarray(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)))


    def export_detection_log(self):
        # TODO 导出检测结果日志
        print("没做")


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    ui = infantMain()
    ui.show()
    sys.exit(app.exec_())
