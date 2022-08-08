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


def show_data(prob_result):
    probs = []
    res = str
    name_list = ['Head', 'LeftHand', 'LeftLeg', 'RightHand', 'RightLeg']
    for i in range(len(prob_result)):
        if prob_result[i] > 0.5:
            prob_result[i] = prob_result[i]
            probs.append(name_list[i] + '(' + str("%.2f" % (prob_result[i] * 100)) + '%)')
    if len(probs):
        res = ('+'.join(probs))
    else:
        res = "Normal"
    return res


def center_crop(frame):
    frame = frame[8:120, 30:142, :]
    return np.array(frame).astype(np.uint8)

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

    cap = cv2.VideoCapture(0)
    retaining = True
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')

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

            probs = torch.sigmoid(outputs)
            prob_result = probs.cpu().detach().numpy()
            prob_result = prob_result.reshape(prob_result.shape[0] * prob_result.shape[1], )
            res = show_data(prob_result)

            cv2.putText(frame, res, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

            clip.pop(0)

        frame = cv2.resize(frame, (800, 600))
        cv2.imshow('InfantDetection', frame)
        cv2.waitKey(30)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
