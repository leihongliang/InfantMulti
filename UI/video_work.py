import datetime
from queue import Queue
import PIL.Image as Image
import cv2
import numpy as np
import pylab as pl
import time
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from matplotlib.backends.backend_agg import FigureCanvasAgg

from UI.inference_gui import infantDetection, moderl_load, center_crop

Decode2Play = Queue()


class cvDecode(QThread):
    def __init__(self):
        super(cvDecode, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.video_path = ""  # 视频文件路径
        self.changeFlag = 0  # 判断视频文件路径是否更改
        self.cap = cv2.VideoCapture()

    def run(self):
        while self.threadFlag:
            if self.changeFlag == 1 and self.video_path != "":
                self.changeFlag = 0
                self.cap = cv2.VideoCapture(r"" + self.video_path)

            if self.video_path != "":
                if self.cap.isOpened():
                    ret, frame = self.cap.read()
                    time.sleep(0.05)  # 控制读取录像的时间，连实时视频的时候改成time.sleep(0.001)，多线程的情况下最好加上，否则不同线程间容易抢占资源

                    # 下面两行码用来控制循环播放的，如果不需要可以删除
                    if frame is None:
                        self.cap = cv2.VideoCapture(r"" + self.video_path)

                    if ret:
                        Decode2Play.put(frame)  # 解码后的数据放到队列中
                    del frame  # 释放资源
                else:
                    #   控制重连
                    self.cap = cv2.VideoCapture(r"" + self.video_path)
                    time.sleep(0.01)


class play_Work(QObject):
    def __init__(self):
        super(play_Work, self).__init__()
        self.threadFlag = 0  # 控制线程退出
        self.playFlag = 0  # 控制播放/暂停
        self.playLabel = QLabel()  # 初始化QLabel对象
        self.speech_signal = QLabel()
        self.recognition_result = QListWidget()
        self.data_statistic_show = QLabel()
        self.infant_detection_model = moderl_load()
        self.clip = []

    #   不需要重写run方法
    def play(self):
        last_result = ' '
        this_result = ' '
        self.prob_result = np.array([[0,0,0,0,0]])
        model=self.infant_detection_model

        while self.threadFlag:
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                tmp_ = center_crop(cv2.resize(frame, (171, 128)))
                tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                self.clip.append(tmp)
                if len(self.clip) > 16:
                    self.clip = []
                if len(self.clip) == 16 and self.playFlag == 1:
                    frame, class_result, prob_result = infantDetection(model, clip=self.clip, frame=frame)
                    self.prob_result = prob_result
                    class_result = str(class_result)
                    print(class_result,prob_result)
                    this_result = class_result
                    self.clip = []
                if self.playFlag == 1:
                    frame = cv2.resize(frame, (760, 428), cv2.INTER_LINEAR)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    qimg = QImage(frame.data, frame.shape[1], frame.shape[0],
                                  QImage.Format_RGB888)  # 在这里可以对每帧图像进行处理，
                    self.playLabel.setPixmap(QPixmap.fromImage(qimg))  # 图像在QLabel上展示
                    if (last_result != this_result):
                        now = datetime.datetime.now()
                        # print(self.show_data(prob_result))
                        self.recognition_result.addItem(now.strftime("%m-%d %H:%M:%S") + '识别结果：' + self.show_data(prob_result))
                        # self.data_statistic(this_result)  # 统计结果自增1
                        piximage_data_statistic = self.draw_bar().toqpixmap()
                        self.data_statistic_show.setPixmap(piximage_data_statistic)
                        self.data_statistic_show.setScaledContents(True)
                        # self.speech_signal.setPixmap(self.image)
                    last_result = this_result
                # cv2.putText(frame, class_result, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 5)
            # time.sleep(0.001)


    def show_data(self, prob_result):
        probs = []
        res = str
        name_list = ['头部', '左手', '右手', '左腿', '右腿']
        for i in range(len(prob_result)):
            if prob_result[i] > 0.5:
                prob_result[i] = prob_result[i]
                probs.append(name_list[i] + '(' + str("%.2f" % (prob_result[i] * 100)) + '%)')
        if len(probs):
            res = ('+'.join(probs))
        else:
            res = "正常"
        return res


    # 显示概率
    def draw_bar(self):
        pl.rcParams['font.sans-serif'] = ['SimHei']
        name_list = ['头部', '左手', '左腿', '右手', '右腿']
        num_list = self.prob_result
        fig = pl.figure(figsize=(6.5, 4))
        pl.ylim(0, 1)
        pl.bar(range(len(num_list)), num_list, tick_label=name_list)
        canvas = FigureCanvasAgg(pl.gcf())
        fig.canvas.draw()
        w, h = canvas.get_width_height()
        buf = np.fromstring(canvas.tostring_argb(), dtype=np.uint8)
        buf.shape = (w, h, 4)
        buf = np.roll(buf, 3, axis=2)
        image = Image.frombytes("RGBA", (w, h), buf.tostring())
        return image
