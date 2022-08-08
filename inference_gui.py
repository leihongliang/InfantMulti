import torch
import numpy as np
from network import C3D_model
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

torch.backends.cudnn.benchmark = True


def CenterCrop(frame, size):
    h, w = np.shape(frame)[0:2]
    th, tw = size
    x1 = int(round((w - tw) / 2.))
    y1 = int(round((h - th) / 2.))

    frame = frame[y1:y1 + th, x1:x1 + tw, :]
    return np.array(frame).astype(np.uint8)


def center_crop(frame):
    def play(self):
        last_result = ' '
        this_result = ' '
        while self.threadFlag:
            if not Decode2Play.empty():
                frame = Decode2Play.get()
                tmp_ = center_crop(cv2.resize(frame, (171, 128)))
                tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
                self.clip.append(tmp)
                if len(self.clip) > 16:
                    self.clip = []
                if len(self.clip) == 16 and self.playFlag == 1:
                    frame, class_result, prob_result = InfantDetection(model=self.infant_detection_model,
                                                                       clip=self.clip, frame=frame)
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
                        prob = '%.2f' % (prob_result.item() * 100)
                        self.recognition_result.addItem(
                            now.strftime("%m-%d %H:%M:%S") + '识别结果：' + class_result + '  概率：' + str(prob) + '%')
                        self.data_statistic(this_result)  # 统计结果自增1
                        piximage_data_statistic = self.draw_bar().toqpixmap()
                        self.data_statistic_show.setPixmap(piximage_data_statistic)
                        self.speech_signal.setPixmap(self.image)
                    last_result = this_result
            time.sleep(0.001)
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def moderl_load():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    model = C3D_model.C3D(num_classes=5)
    checkpoint = torch.load(
        'E:\HDU\Project\InfantMuti\\run\C3D-infant_multi_labels_epoch-199.pth.tar',
        map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    return model


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)


def label_conversion(probs_01):
    prob = ','.join(str(i) for i in probs_01)
    prob = prob.replace(' ', ',')
    dic = {}
    keys = []
    fr = open('/dataloaders/infant_multi_labels_2.txt', 'r')
    for line in fr.readlines():
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dic[k] = v
    fr.close()
    return dic[prob]


def infantDetection(model, clip, frame):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('E:\HDU\Project\InfantC3D\dataloaders\\ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    class_result = ""
    prob_result = float(0.0)

    if len(clip) == 16:
        inputs = np.array(clip).astype(np.float32)
        inputs = np.expand_dims(inputs, axis=0)
        inputs = np.transpose(inputs, (0, 4, 1, 2, 3))
        inputs = torch.from_numpy(inputs)
        inputs = torch.autograd.Variable(inputs, requires_grad=False).to(device)
        with torch.no_grad():
            outputs = model.forward(inputs)

        # probs = torch.nn.Softmax(dim=1)(outputs)
        # label = torch.max(probs, 1)[1].detach().cpu().numpy()[0]
        #
        # class_result = class_names[label].split(' ')[-1].strip()
        # prob_result = probs[0][label]
        probs = torch.sigmoid(outputs)
        probs_np = probs.cpu().detach().numpy()
        probs_01 = np.int64(probs_np > 0.7)
        class_result = label_conversion(probs_01)
        cv2.putText(frame, class_result, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)
        clip.pop(0)

    # return frame, class_result, prob_result
    return frame, class_result, prob_result


class infantDetection_Thread(QThread):
    finish_Single = pyqtSignal(list, float)
    def __init__(self, frame, parent=None):
        super(infantDetection_Thread, self).__init__(parent)
        self.frame = frame

    def run(self):
        # class_result, prob_result = InfantDetection(self.frame)
        class_result, prob_result = 1, 0.5
        self.finish_Single.emit(list(class_result), float(prob_result))


if __name__ == '__main__':
    videoName = ""
    infantDetection(videoName)
