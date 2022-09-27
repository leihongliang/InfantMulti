import torch
import torch.nn as nn
from torch import tensor
from torch.nn import functional as F
from utils.lossFunction import bce_lsep, lsep, wlsep, bce


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

# print(outputsigmod)

bceloss = bce()
wlseploss = wlsep()
lseploss = lsep()
fussionloss = bce_lsep()

bce = bceloss(output.float(), label)
wlsep = wlseploss(output.float(), label)
lsep = lseploss(output.float(), label)
fussion = fussionloss(output.float(), label)

print(bce, wlsep, lsep, fussion)
