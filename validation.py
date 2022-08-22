import timeit
from datetime import datetime
import socket
import os
import glob
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from tensorboardX import SummaryWriter
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model, Resnet

"""
旧版本，没啥用了
新的train文件，换了评价标准，而且有函数会保留每轮训练结果
"""
def Accuracy(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(np.logical_or(y_true[i], y_pred[i]))
        count += p / q
    return count / y_true.shape[0]

def Precision(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_pred[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_pred[i])
    return count / y_true.shape[0]

def Recall(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if sum(y_true[i]) == 0:
            continue
        count += sum(np.logical_and(y_true[i], y_pred[i])) / sum(y_true[i])
    return count / y_true.shape[0]

def F1Measure(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]

def Failure_Analysis( y_true, y_pred ):
    count = 0
    for i in range(y_true.shape[0]):
        if ( sum(y_true[i] - y_pred[i] ) != 0 ):
            print( "y_true:", y_true[i], "y_pred:", y_pred[i] )
    return count / y_true.shape[0]
        # else:
        #     true = true + 1
        #     print( true, "y_true:", y_true[i], "y_pred:", y_pred[i] )


def validation():
    modelName = '3DResnet'
    label = 'multi'
    dataset = 'infant_multi_labels'

    running_loss = 0.0
    running_corrects = 0.0
    running_acc = 0.0
    running_acc2 = 0.0
    count = 0
    true = 0
    false = 0

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used:", device)
    with open('dataloaders/infant_multi_labels.txt', 'r') as f:
        class_names = f.readlines()
        f.close()

    if dataset == 'hmdb51':
        num_classes = 51
    elif dataset == 'infant_multi_labels':
        num_classes = 5

    if modelName == '3DResnet':
        model = Resnet_3D.generate_model(50)
        checkpoint = torch.load(os.path.join(os.path.abspath("."), r'run\run_2\models\3DResnet-ucf101_epoch-999.pth.tar'),map_location='cpu')
    elif modelName =="C3D":
        model = C3D_model.C3D(num_classes=5)
        checkpoint = torch.load(os.path.join(os.path.abspath("."), r'run\C3D-infant_multi_labels_epoch-199.pth.tar'),map_location=lambda storage, loc: storage)
    # optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=4,shuffle=True,num_workers=1)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=4, num_workers=1)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=4, num_workers=1)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['val']}
    test_size = len(test_dataloader.dataset)


    for inputs, labels in tqdm(test_dataloader):
        inputs = Variable(inputs, requires_grad=True).to(device)
        labels = Variable(labels).to(device)

        count += 1
        true += true
        false += false

        with torch.no_grad():
            outputs = model.forward(inputs)

        if label == 'singel':
            criterion = nn.CrossEntropyLoss()
            probs = nn.Softmax(dim=1)(outputs)
            preds = torch.max(probs, 1)[1]
            loss = criterion(outputs, labels)
            running_corrects += torch.sum(preds == labels.data)

        elif label == 'multi':
            criterion_multi = nn.BCELoss()
            probs = torch.sigmoid(outputs)
            probs_np = probs.cpu().detach().numpy()
            probs_01 = np.int64(probs_np > 0.5)
            loss = criterion_multi(torch.sigmoid(outputs.float()), labels.float())
            # running_corrects += torch.sum(probs_01 == labels.data)


        running_loss += loss.item() * inputs.size(0)
        running_acc += F1Measure(labels.cpu().detach().numpy(), probs_01)
        # running_acc2 += Failure_Analysis( labels.cpu().detach().numpy(), probs_01)
        running_acc += running_corrects

    epoch_loss = running_loss / test_size
    epoch_acc = running_acc / count
    print('data/test_loss_epoch: ', epoch_loss, 'data/test_acc_epoch', epoch_acc, running_acc2)

if __name__ == "__main__":
    validation()
