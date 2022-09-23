import torch
import torch.nn as nn
from torch import tensor
from torch.nn import functional as F
from utils import lossFunction


def lsep(input, target, average=True):
    # 给出了相减之后的一个矩阵
    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_different = (target.unsqueeze(1) < target.unsqueeze(2)).float()
    exps = differences.exp() * where_different
    lsep = torch.log(1 + exps.sum(2).sum(1))
    if average:
        return lsep.mean()
    else:
        return lsep


def wlsep(input, target, weights=None):
    mask = ((target.unsqueeze(1).expand(target.size(0), target.size(1), target.size(1)) -
             target.unsqueeze(2).expand(target.size(0), target.size(1), target.size(1))) > 0).float()
    diffs = (input.unsqueeze(2).expand(target.size(0), target.size(1), target.size(1)) -
             input.unsqueeze(1).expand(target.size(0), target.size(1), target.size(1)))
    if weights is not None:
        return F.pad(diffs.add(-(1 - mask) * 1e10),
                     pad=(0, 0, 0, 1)).logsumexp(dim=1).mul(weights).masked_select(target.bool()).mean()
    else:
        return F.pad(diffs.add(-(1 - mask) * 1e10),
                     pad=(0, 0, 0, 1)).logsumexp(dim=1).masked_select(target.bool()).mean()


labels = tensor([[0, 1, 1, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 1, 1, 1, 0]])
outputs = tensor([[0, 1, 1, 1, 1],
                  [1, 0, 1, 1, 1],
                  [1, 1, 0, 1, 1]])

# res = lsep(outputs, labels)
label = tensor([[0, 0, 1, 1, 1]])
# label = tensor([[0, 1, 1, 1, 1]])
# label = tensor([[0, 1, 1, 1, 1]])
# output = tensor([[0, 1, 3]])
output = tensor([[-1, -1, 1, 1, 1]])

label2 = tensor([2])
output2 = tensor([[-1, 2, -1]],  dtype=torch.float)

outputsigmod = torch.sigmoid(output.float())
# print(outputsigmod)

bceloss = nn.BCELoss()
criterion = nn.CrossEntropyLoss()

bce = bceloss(outputsigmod, label.float())
wlsep = wlsep(output.float(), label)
lsep = lsep(output.float(), label)


ce = criterion(output2, label2)
print(bce, wlsep, lsep, ce)
