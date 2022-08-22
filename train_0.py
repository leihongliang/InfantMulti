# -*- coding: UTF-8 -*-
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
from network import C3D_model, R2Plus1D_model, R3D_model, Resnet, Res2Net3D

"""
原始版本
没啥用了
"""

def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(lr)
    return lr


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


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 200  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 3  # Run on test set every nTestInterval epochs
snapshot = 50  # Store a model every snapshot epochs
lr = 4e-3  # Learning rate
# lr = 1e-3   # 原

dataset = 'infant_multi_labels'

if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'infant_multi_labels':
    # 身体五个部位
    num_classes = 5
else:
    print('datasets wrong！')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
with open(save_dir + '/trainloss.txt', 'w') as f:
    f.close()
with open(save_dir + '/trainacc.txt', 'w') as f:
    f.close()
with open(save_dir + '/valloss.txt', 'w') as f:
    f.close()
with open(save_dir + '/valacc.txt', 'w') as f:
    f.close()

# 模型选择
# modelName = 'C3D'
modelName = 'Res2Net3D'
# modelName = '3DResnet'
saveName = modelName + '-' + dataset


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """

    if modelName == 'C3D':
        model = C3D_model.C3D(num_classes=num_classes, pretrained=True)
        train_params = [{'params': C3D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif modelName == '3DResnet':
        # resnet50
        model = Resnet_3D.generate_model(50)
        train_params = model.parameters()
        # 加载预训练模型
        # pretext_model = torch.load('pretrained/r3d50_S_200ep.pth')
        # model2_dict = model.state_dict()
        # state_dict = {k: v for k, v in pretext_model.items() if k in model2_dict.keys()}
        # model2_dict.update(state_dict)
        # model.load_state_dict(model2_dict)
    elif modelName == 'Res2Net3D':
        model = Res2Net3D.res2net50(pretrained=False)
        train_params = model.parameters()
    else:
        print('No more model')
        raise NotImplementedError

    # 单标签二分类损失函数
    criterion = nn.CrossEntropyLoss()  # standard crossentropy loss for classification
    # 多标签二分类问题函数
    criterion_multi = nn.BCELoss()
    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    # 原参数 step_size=10
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50,gamma=0.8)  # the scheduler divides the lr by 10 every 10 epochs
    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage)  # Load all tensors onto the CPU
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
    model.to(device)

    # 单标签
    # criterion.to(device)
    criterion_multi.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=20, shuffle=True,num_workers=4)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=20, num_workers=4)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=20, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0

            running_acc = 0.0
            count = 0
            # set model to train() or eval() mode depending on whether it is trainFed
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                # 自适应学习率
                # w_lr=adjust_learning_rate_poly(optimizer,epoch,num_epochs,lr,4)
                model.train()
            else:
                model.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                # move inputs and labels to the device the training is taking place on
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()
                count += 1
                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                # print(outputs)
                loss = criterion_multi(torch.sigmoid(outputs.float()), labels.float())

                probs = torch.sigmoid(outputs)
                probs_np = probs.cpu().detach().numpy()
                probs_01 = np.int64(probs_np > 0.5)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_acc += F1Measure(labels.cpu().detach().numpy(), probs_01)
                # running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_acc / count
            # epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                # writer.add_scalar('lr/eppch',w_lr,epoch)
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
                with open(save_dir + '/trainloss.txt', 'a') as f:
                    f.write('%.5f\n' % (epoch_loss))
                    f.close()
                with open(save_dir + '/trainacc.txt', 'a') as f:
                    f.write('%.5f\n' % (epoch_acc))
                    f.close()


            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
                with open(save_dir + '/valloss.txt', 'a') as f:
                    f.write('%.5f\n' % (epoch_loss))
                    f.close()
                with open(save_dir + '/valacc.txt', 'a') as f:
                    f.write('%.5f\n' % (epoch_acc))
                    f.close()


            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0
            running_acc = 0.0
            count = 0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)
                count += 1
                with torch.no_grad():
                    outputs = model(inputs)
                # probs = nn.Softmax(dim=1)(outputs)
                # preds = torch.max(probs, 1)[1]
                probs = torch.sigmoid(outputs)
                probs_np = probs.cpu().detach().numpy()
                probs_01 = np.int64(probs_np > 0.5)

                # loss = criterion(outputs, labels)
                loss = criterion_multi(torch.sigmoid(outputs.float()), labels.float())

                running_loss += loss.item() * inputs.size(0)
                # running_corrects += torch.sum(preds == labels.data)
                running_acc += F1Measure(labels.cpu().detach().numpy(), probs_01)

            epoch_loss = running_loss / test_size
            epoch_acc = running_acc / count

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch + 1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()
