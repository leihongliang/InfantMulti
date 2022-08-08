import torch
import numpy as np
from network import C3D_model,Resnet
import cv2

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

def label_conversion(probs_01):
    prob = ','.join(str(i) for i in probs_01)
    prob = prob.replace(' ', ',')
    dic = {}
    keys = []
    fr = open('/\dataloaders\infant_multi_labels_2.txt', 'r')
    for line in fr.readlines():
        k = line.split(' ')[0]
        v = line.split(' ')[1]
        dic[k] = v
    fr.close()
    return dic[prob]

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    with open('dataloaders/infant_multi_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()
    # init model
    # model = C3D_model.C3D(num_classes=5)
    # checkpoint = torch.load('E:\HDU\Project\InfantMulti\\run\C3D-infant_multi_labels_epoch-199.pth.tar',map_location=lambda storage, loc: storage)
    model = Resnet_3D.generate_model(50)
    checkpoint = torch.load('E:\HDU\Project\InfantMuti/run/run_2\models/3DResnet-ucf101_epoch-999.pth.tar',map_location='cpu')

    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()

    # read video
    video = 'E:\HDU\Project\InfantC3D\data\demo_video.mp4'

    cap = cv2.VideoCapture(video)
    retaining = True
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

    # create VideoWriter for saving
    outVideo = cv2.VideoWriter('save_test_video.avi', fourcc, fps, (width, height))

    clip = []
    while retaining:
        retaining, frame = cap.read()
        if not retaining and frame is None:
            continue
        tmp_ = center_crop(cv2.resize(frame, (171, 128)))
        tmp = tmp_ - np.array([[[90.0, 98.0, 102.0]]])
        clip.append(tmp)
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
            probs_np = probs.cpu().detach().numpy()# 返回一个new Tensor，只不过不再有梯度
            probs_01 = np.int64(probs_np > 0.7)
            label = label_conversion(probs_01)
            print(probs_np)
            print(probs_01)
            # cv2.putText(frame, class_names[label].split(' ')[-1].strip(), (20, 100),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5,
            #             (0, 0, 255), 1)
            cv2.putText(frame, label, (20, 100),cv2.FONT_HERSHEY_SIMPLEX, 1.5,(0, 0, 255), 2)
            # a = class_names[label].split(' ')[-1].strip()
            # b = probs[0][label]
            # cv2.putText(frame, "prob: %.4f" % probs[0][label], (20, 140),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1.5,
            #             (0, 0, 255), 1)
            clip.pop(0)
        # outVideo.write(frame)
        frame = cv2.resize(frame,(960,540))
        cv2.imshow('result', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
