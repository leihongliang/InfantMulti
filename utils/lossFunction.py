import torch
import torch.nn as nn
from torch.nn import functional as F
# from utils.lossFunction import bce_lsep, lsep, wlsep, bce

class bce(nn.Module):

    def __init__(self) -> None:

        super(bce, self).__init__()

    def forward(self, output, target):
        """

        Args:
            output: 已经做了sigmoid，不需要额外进行
            target: ground-true

        Returns:

        """
        output_sigmoid = torch.sigmoid(output.float())
        batch_size = output_sigmoid.shape[0]  # 有几条样本的输出结果
        loss = 0
        for i in range(batch_size):  # 对每条样本
            for j in range(len(output_sigmoid[i])):  # 对每条样本所有类别
                if target[i][j] == 0:
                    temp_loss = torch.log(1 - output_sigmoid[i][j])
                else:
                    temp_loss = torch.log(output_sigmoid[i][j])
                loss = loss - temp_loss
        return loss / (batch_size * output_sigmoid.shape[1])


class lsep(nn.Module):
    def __init__(self) -> None:
        super(lsep, self).__init__()

    def forward(self, input, target, average=True):
        differences = input.unsqueeze(1) - input.unsqueeze(2)
        where_different = (target.unsqueeze(1) < target.unsqueeze(2)).float()

        exps = differences.exp() * where_different
        lsep = torch.log(1 + exps.sum(2).sum(1))

        if average:
            return lsep.mean()
        else:
            return lsep

class wlsep(nn.Module):
    def __init__(self) -> None:
        super(wlsep, self).__init__()

    def forward(self, scores, labels, weights=None):
        mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
                 labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
        diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
                 scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
        if weights is not None:
            return F.pad(diffs.add(-(1 - mask) * 1e10),
                         pad=(0, 0, 0, 1)).logsumexp(dim=1).mul(weights).masked_select(labels.bool()).mean()
        else:
            return F.pad(diffs.add(-(1 - mask) * 1e10),
                         pad=(0, 0, 0, 1)).logsumexp(dim=1).masked_select(labels.bool()).mean()

class bce_wlsep(nn.Module):
    def __init__(self) -> None:
        super(bce_wlsep, self).__init__()

    def forward(self, output, target):
        """

        Args:
            output: 输入不需要预先sigmoid，函数里面再做
            target: ground-true

        Returns: bce + wlsep (without sigmoid)

        """
        bceF = bce()
        bceloss = bceF(output, target)
        wlsepF = wlsep()
        wlseploss = wlsepF(output, target)
        # wlseploss = torch.tanh(torch.log(wlseploss))

        fuss = bceloss + wlseploss

        return fuss