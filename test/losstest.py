import torch
import torch.nn as nn
from torch import tensor
from torch.nn import functional as F

# batchsize为20
labels = tensor([[0, 1, 1, 0, 0],
                 [0, 0, 1, 0, 0],
                 [0, 1, 1, 1, 0],
                 [1, 0, 0, 0, 1]])
outputs = tensor([[-0.4876, 2.527, 3.669, -0.2799, -0.2925],
                  [-4.276, -2.95, 5.306, -1.4016, -0.301],
                  [-1.5691, -0.3117, 3.266, 3.947, -0.4907],
                  [5.271, -2.870, -1.2821, -3.5109, 4.7]])

# outputs2 = tensor([[-0.4876, -0.2527, 3.669, -0.2799, -0.2925],
#                   [-0.4276, -0.1295, -0.5306, -0.4016, -0.0301],
#                   [-1.5691, -0.3117, -0.3266, -0.3947, -0.4907],
#                   [-0.5271, -0.2870, -0.2821, -0.5109, -0.0047]])
outputs2 = tensor([[-4.876, -0.2527, 3.669, -2.799, -2.925],
                  [-0.4276, -0.1295, -0.5306, -0.4016, -0.0301],
                  [-1.5691, -0.3117, -0.3266, -0.3947, -0.4907],
                  [-0.5271, -0.2870, -0.2821, -0.5109, -0.0047]])

label = tensor([[1, 1, 1, 0, 0]])
output0 = tensor([[1, 0, 0, 0, 0]])
output1 = tensor([[1, 0, 0, 1, 0]]) # 差不多对
output2 = tensor([[1, 0, 1, 0, 0]]) # 漏检
output3 = tensor([[0.9, 5, 2, 5, 0]]) # 多检出来
output4 = tensor([[0.9, 0.3, 2, 0, 4]]) # 一对一错



def lsep(scores, labels, weights=None):
    mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
             labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
    diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
             scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
    return diffs.exp().mul(mask).sum().add(1).log().mean()

def lsep2(input, target, average=True):
    a = input.unsqueeze(1)
    b = input.unsqueeze(2)
    differences = input.unsqueeze(1) - input.unsqueeze(2)
    where_different = (target.unsqueeze(1) < target.unsqueeze(2)).float()
    exps = differences.exp() * where_different
    lsep = torch.log(1 + exps.sum(2).sum(1))
    if average:
        return lsep.mean()
    else:
        return lsep

def wlsep(scores, labels, weights=None):
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

bce = nn.BCELoss()
loss_bce0 = bce(torch.sigmoid(output0.float()), label.float())
loss_bce1 = bce(torch.sigmoid(output1.float()), label.float())
loss_bce2 = bce(torch.sigmoid(output2.float()), label.float())
loss_bce3 = bce(torch.sigmoid(output3.float()), label.float())
loss_bce4 = bce(torch.sigmoid(output4.float()), label.float())
loss_bce5 = bce(torch.sigmoid(outputs.float()), labels.float())
loss_bce6 = bce(torch.sigmoid(outputs2.float()), labels.float())
print(loss_bce0, loss_bce1, loss_bce2, loss_bce3, loss_bce4, loss_bce5, loss_bce6)
# tensor(0.4652) tensor(0.3053) tensor(0.6330) tensor(1.2349) tensor(1.1467)


loss_lsep0 = lsep(output0.float(), label.float())
loss_lsep1 = lsep(output1.float(), label.float())
loss_lsep2 = lsep(output2.float(), label.float())
loss_lsep3 = lsep(output3.float(), label.float())
loss_lsep4 = lsep(output4.float(), label.float())
loss_lsep5 = lsep(outputs.float(), labels.float())
loss_lsep6 = lsep(outputs2.float(), labels.float())
# print(loss_lsep0, loss_lsep1, loss_lsep2, loss_lsep3, loss_lsep4, loss_lsep5, loss_lsep6)
# tensor(1.7513) tensor(1.5705) tensor(1.9007) tensor(1.8793) tensor(1.9937)

loss_lsep0 = lsep2(output0.float(), label.float())
loss_lsep1 = lsep2(output1.float(), label.float())
loss_lsep2 = lsep2(output2.float(), label.float())
loss_lsep3 = lsep2(output3.float(), label.float())
loss_lsep4 = lsep2(output4.float(), label.float())
loss_lsep5 = lsep2(outputs.float(), label.float())
loss_lsep6 = lsep2(outputs2.float(), label.float())
print(loss_lsep0, loss_lsep1, loss_lsep2, loss_lsep3, loss_lsep4, loss_lsep5, loss_lsep6)

loss_wlsep0 = wlsep(output0.float(), label.float())
loss_wlsep1 = wlsep(output1.float(), label.float())
loss_wlsep2 = wlsep(output2.float(), label.float())
loss_wlsep3 = wlsep(output3.float(), label.float())
loss_wlsep4 = wlsep(output4.float(), label.float())
loss_wlsep5 = wlsep(outputs.float(), labels.float())
loss_wlsep6 = wlsep(outputs2.float(), labels.float())
# print(loss_wlsep0, loss_wlsep1, loss_wlsep2, loss_wlsep3, loss_wlsep4, loss_wlsep5, loss_wlsep6)
# tensor(0.9506) tensor(0.8192) tensor(1.0616) tensor(1.0443) tensor(1.1324)