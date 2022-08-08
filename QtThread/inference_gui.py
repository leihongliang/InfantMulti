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
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

def moderl_load():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    # init model
    model = C3D_model.C3D(num_classes=5)
    checkpoint = torch.load('E:\HDU\Project\InfantMuti\\run\C3D-infant_multi_labels_epoch-199.pth.tar',
                            map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

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

def InfantDetection(model, clip, frame):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print("Device being used:", device)
    # # init model
    # model = C3D_model.C3D(num_classes=101)
    # checkpoint = torch.load('/home/yuhj/Documents/Infant Pose/video-recognition-master/run/run_1/models/C3D-ucf101_epoch-99.pth.tar', map_location=lambda storage, loc: storage)
    # model.load_state_dict(checkpoint['state_dict'])
    # model.to(device)
    # model.eval()

    with open('/dataloaders/org_ucf_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    # clip = []
    class_result = ""
    prob_result = float(0.0)

    # tmp_ = center_crop(cv2.resize(frame, (171, 128)))
    # tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
    # clip.append(tmp)
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
        probs = torch.sigmoid(outputs)
        probs_np = probs.cpu().detach().numpy()
        probs_01 = np.int64(probs_np > 0.7)
        label = label_conversion(probs_01)


        class_result = class_names[label].split(' ')[-1].strip()
        prob_result = probs[0][label]
        cv2.putText(frame, label, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 1)

        print(class_result, prob_result)

        # cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 100),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5,
        #             (0, 0, 255), 1)
        #
        # cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 140),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.5,
        #             (0, 0, 255), 1)
        clip.pop(0)

    return frame


    # cap.release()
    # cv2.destroyAllWindows()


class InfantDetection_Thread(QThread):
    finish_Single = pyqtSignal(list, float)

    def __init__(self, frame, parent=None):
        super(InfantDetection_Thread, self).__init__(parent)
        self.frame = frame

    def run(self):
        # class_result, prob_result = InfantDetection(self.frame)
        class_result, prob_result = 1, 0.5
        self.finish_Single.emit(list(class_result), float(prob_result))


if __name__ == '__main__':
    videoName = ""
    InfantDetection(videoName)









