# -*- coding: UTF-8 -*-
import timeit
import warnings
from datetime import datetime
import socket
import os
import glob
from sklearn import metrics
import numpy as np
import random
from tensorboardX import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable

from dataloaders.dataset import VideoDataset
from network import C3D_model, R2Plus1D_model, R3D_model, Resnet, Res2Net3D, resnext, C3D_base
from utils import lossFunction

warnings.filterwarnings("ignore")

"""主 改进版本"""
"""增加了一些参数"""

def compute_mAP(y_true, y_pred):
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(metrics.average_precision_score(y_true[:, i], y_pred[:, i], average='macro', pos_label=1))
    return np.mean(AP)


def adjust_learning_rate_poly(optimizer, epoch, num_epochs, base_lr, power):
    lr = base_lr * (1 - epoch / num_epochs) ** power
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print(lr)
    return lr


def F1Measure(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        if (sum(y_true[i]) == 0) and (sum(y_pred[i]) == 0):
            continue
        p = sum(np.logical_and(y_true[i], y_pred[i]))
        q = sum(y_true[i]) + sum(y_pred[i])
        count += (2 * p) / q
    return count / y_true.shape[0]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False


setup_seed(2)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 400  # Number of epochs for training
resume_epoch = 0  # Default is 0, change if want to resume
useTest = True  # See evolution of the test set when training
nTestInterval = 1  # Run on test set every nTestInterval epochs
snapshot = 1000  # Store a model every snapshot epochs
lr = 8e-3 # Learning rate

dataset = 'infant_multi_labels'

if dataset == 'infant_multi_labels':
    num_classes = 5
else:
    print('datasets wrong！')
    raise NotImplementedError

# save_dir_root= '/student1/hl_lei/InfantMulti'
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
# exp_name = 'InfantMulti'
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    list = []
    for run in runs:
        list.append(run.split('run_')[1])
    list = [int(x) for x in list]
    list = sorted(list)
    run_id = int(list[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    list = []
    for run in runs:
        list.append(run.split('run_')[1])
    list = [int(x) for x in list]
    list = sorted(list)
    run_id = int(list[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

# 模型选择
modelName = 'C3D'
# modelName = 'Resnet'
saveName = modelName


def train_model(dataset = dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):
    global acc_epoch
    if modelName == 'C3D':
        model = C3D_base.C3D(num_classes=num_classes , pretrained=False)
        train_params = [{'params': C3D_base.get_1x_lr_params(model), 'lr': lr},
                        {'params': C3D_base.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R2Plus1D':
        model = R2Plus1D_model.R2Plus1DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = [{'params': R2Plus1D_model.get_1x_lr_params(model), 'lr'
                                                                           '': lr},
                        {'params': R2Plus1D_model.get_10x_lr_params(model), 'lr': lr * 10}]
    elif modelName == 'R3D':
        model = R3D_model.R3DClassifier(num_classes=num_classes, layer_sizes=(2, 2, 2, 2))
        train_params = model.parameters()
    elif modelName == 'Resnet':
        # resnet50
        model = Resnet.generate_model(50)
        train_params = model.parameters()
        # 加载预训练模型
        pretext_model = torch.load(save_dir_root + '/pretrained/r3d50_S_200ep.pth')
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in pretext_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    elif modelName == 'ResneXt':
        model = resnext.generate_model(50)
        train_params = model.parameters()
    elif modelName == 'Res2Net3D':
        model = Res2Net3D.res2net50(pretrained=False)
        train_params = model.parameters()
    else:
        print('No more model')
        raise NotImplementedError


    # 多标签二分类问题函数
    criterion_multi = nn.BCELoss()
    criterion_multi2 = lossFunction.wlsep()
    # criterion_multi = lossFunction.bce()

    optimizer = optim.SGD(train_params, lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
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
    criterion_multi.to(device)

    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16), batch_size=16, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(VideoDataset(dataset=dataset, split='val', clip_len=16), batch_size=16, num_workers=8)
    test_dataloader = DataLoader(VideoDataset(dataset=dataset, split='test', clip_len=16), batch_size=16, num_workers=8)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)
    best_acc = 0.0
    best_of1 = 0.0
    best_cf1 = 0.0
    best_map = 0.0

    with open(save_dir + '/train.txt', 'w') as f:
        f.close()
    with open(save_dir + '/val.txt', 'w') as f:
        f.close()
    with open(save_dir + '/test.txt', 'w') as f:
        f.close()

    for epoch in range(resume_epoch, num_epochs):
        # each epoch has a training and validation step
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0.0
            AP_prob = []
            AP_target = []
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
                # inputs(batchsize, 3, 16, 112, 112)
                # labels(batchsize, 5)
                inputs = Variable(inputs, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()
                # count += 1
                if phase == 'train':
                    outputs = model(inputs)
                else:
                    with torch.no_grad():
                        outputs = model(inputs)
                # print(outputs)

                # bce需要预先经过sigmoid
                # loss = criterion_multi(torch.sigmoid(outputs.float()), labels.float())
                loss1 = criterion_multi(torch.sigmoid(outputs.float()), labels.float())
                loss2 = criterion_multi2(torch.sigmoid(outputs.float()), labels.float())
                loss = loss1 + loss2

                probs = torch.sigmoid(outputs)
                # 这个放到后面整个矩阵一起处理
                # probs = probs.cpu().detach().numpy()
                # probs = np.int64(probs > 0.5)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                # running_acc += F1Measure(labels.cpu().detach().numpy(), probs_01)
                # running_corrects += torch.sum(preds == labels.data)
                AP_prob.append(probs)
                AP_target.append(labels)

            epoch_loss = running_loss / trainval_sizes[phase]
            # epoch_acc = running_corrects.double() / trainval_sizes[phase]
            # 标签 （452,5）
            sum_target = torch.cat(AP_target).cpu().detach().numpy()
            sum_prob = torch.cat(AP_prob).cpu().detach().numpy()
            sum_probs = np.int64(sum_prob >= 0.5)

            epoch_acc = F1Measure(sum_target, sum_probs)

            epoch_cp = metrics.precision_score(sum_target, sum_probs, average='macro')
            epoch_cr = metrics.recall_score(sum_target, sum_probs, average='macro')
            epoch_cf1 = metrics.f1_score(sum_target, sum_probs, average="macro")

            epoch_op = metrics.precision_score(sum_target, sum_probs, average='micro')
            epoch_or = metrics.recall_score(sum_target, sum_probs, average='micro')
            epoch_of1 = metrics.f1_score(sum_target, sum_probs, average="micro")

            epoch_map = compute_mAP(sum_target, sum_prob)

            if phase == 'train':
                # writer.add_scalar('lr/eppch',w_lr,epoch)
                writer.add_scalar('data/train_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/train_acc_epoch', epoch_acc, epoch)
                with open(save_dir + '/train.txt', 'a') as f:
                    f.write("[{}] Epoch: {}/{} Loss: {} Acc: {} CP: {} CR: {} CF1: {} OP: {} OR: {} OF1: {} MAP: {} "
                            .format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc, epoch_cp, epoch_cr, epoch_cf1,
                                    epoch_op, epoch_or, epoch_of1, epoch_map)+"\n")
                    f.close()

            else:
                writer.add_scalar('data/val_loss_epoch', epoch_loss, epoch)
                writer.add_scalar('data/val_acc_epoch', epoch_acc, epoch)
                with open(save_dir + '/val.txt', 'a') as f:
                    f.write("[{}] Epoch: {}/{} Loss: {} Acc: {} CP: {} CR: {} CF1: {} OP: {} OR: {} OF1: {} MAP: {} "
                            .format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc, epoch_cp, epoch_cr, epoch_cf1,
                                    epoch_op, epoch_or, epoch_of1, epoch_map) + "\n")
                    f.close()

            print("[{}] Epoch: {}/{} Loss: {} Acc: {} CP: {} CR: {} CF1: {} OP: {} OR: {} OF1: {} MAP: {} "
                  .format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc, epoch_cp, epoch_cr, epoch_cf1, epoch_op,
                          epoch_or, epoch_of1, epoch_map))

            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        # 保存模型
        if epoch % save_epoch == (save_epoch - 1):
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(
                os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        # 测试集
        if useTest and epoch % test_interval == (test_interval - 1):
            model.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0
            running_acc = 0.0
            count = 0

            AP_prob = []
            AP_target = []
            running_loss = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs = model(inputs)

                # probs = nn.Softmax(dim=1)(outputs)
                # preds = torch.max(probs, 1)[1]

                # loss = criterion_multi(torch.sigmoid(outputs.float()), labels.float())
                loss1 = criterion_multi(torch.sigmoid(outputs.float()), labels.float())
                loss2 = criterion_multi2(torch.sigmoid(outputs.float()), labels.float())
                loss = loss1 + loss2

                probs = torch.sigmoid(outputs)
                # probs_np = probs.cpu().detach().numpy()
                # probs_01 = np.int64(probs_np > 0.5)

                AP_prob.append(probs)
                AP_target.append(labels)

                running_loss += loss.item() * inputs.size(0)
                AP_prob.append(probs)
                AP_target.append(labels)
                # running_corrects += torch.sum(preds == labels.data)
                # running_acc += F1Measure(labels.cpu().detach().numpy(), probs_01)

            epoch_loss = running_loss / test_size
            # epoch_acc = running_acc / count

            sum_target = torch.cat(AP_target).cpu().detach().numpy()
            sum_prob = torch.cat(AP_prob).cpu().detach().numpy()
            sum_probs = np.int64(sum_prob >= 0.5)

            epoch_acc = F1Measure(sum_target, sum_probs)

            epoch_cp = metrics.precision_score(sum_target, sum_probs, average='macro')
            epoch_cr = metrics.recall_score(sum_target, sum_probs, average='macro')
            epoch_cf1 = metrics.f1_score(sum_target, sum_probs, average="macro")

            epoch_op = metrics.precision_score(sum_target, sum_probs, average='micro')
            epoch_or = metrics.recall_score(sum_target, sum_probs, average='micro')
            epoch_of1 = metrics.f1_score(sum_target, sum_probs, average="micro")

            epoch_map = compute_mAP(sum_target, sum_prob)

            if epoch > 50:
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    acc_epoch = epoch
                    acc_of1 = epoch_of1
                    acc_cf1 = epoch_cf1
                    acc_map = epoch_map
                if epoch_cf1 > best_cf1:
                    cf1_acc = epoch_acc
                    cf1_epoch = epoch
                    cf1_of1 = epoch_of1
                    best_cf1 = epoch_cf1
                    cf1_map = epoch_map
                if epoch_of1 > best_of1:
                    of1_acc = epoch_acc
                    of1_epoch = epoch
                    best_of1 = epoch_of1
                    of1_cf1 = epoch_cf1
                    of1_map = epoch_map
                if epoch_map > best_map:
                    map_acc = epoch_acc
                    map_epoch = epoch
                    map_of1 = epoch_of1
                    map_cf1 = epoch_cf1
                    best_map = epoch_map
                print("acc_Epoch:{} best_acc:{} CF1: {} OF1: {} MAP: {} ".format(acc_epoch, best_acc, acc_cf1, acc_of1, acc_map))
                print("cf1_Epoch:{} acc:{} bestCF1: {} OF1: {} MAP: {} ".format(cf1_epoch, cf1_acc, best_cf1, cf1_of1, cf1_map))
                print("of1_Epoch:{} acc:{} CF1: {} bestOF1: {} MAP: {} ".format(of1_epoch, of1_acc, of1_cf1, best_of1, of1_map))
                print("map_Epoch:{} acc:{} CF1: {} OF1: {} bestMAP: {} ".format(map_epoch, map_acc, map_cf1, map_of1, best_map))

            writer.add_scalar('data/test_loss_epoch', epoch_loss, epoch)
            writer.add_scalar('data/test_acc_epoch', epoch_acc, epoch)


            print("[test] Epoch: {}/{} Loss: {} Acc: {} CP: {} CR: {} CF1: {} OP: {} OR: {} OF1: {} MAP: {} "
                  .format(epoch + 1, nEpochs, epoch_loss, epoch_acc, epoch_cp, epoch_cr, epoch_cf1, epoch_op, epoch_or
                          , epoch_of1, epoch_map))
            with open(save_dir + '/test.txt', 'a') as f:
                f.write("[{}] Epoch: {}/{} Loss: {} Acc: {} CP: {} CR: {} CF1: {} OP: {} OR: {} OF1: {} MAP: {} "
                        .format(phase, epoch + 1, nEpochs, epoch_loss, epoch_acc, epoch_cp, epoch_cr, epoch_cf1,
                                epoch_op, epoch_or, epoch_of1, epoch_map) + "\n")
                f.close()
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    writer.close()


if __name__ == "__main__":
    train_model()
