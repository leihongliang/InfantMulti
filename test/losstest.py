import torch
import torch.nn as nn
from torch import tensor



# labels = tensor([[0, 0, 1, 0, 0],
#                  [0, 0, 1, 0, 0],
#                  [0, 1, 0, 0, 0],
#                  [0, 0, 0, 0, 1],
#                  [0, 0, 0, 0, 0],
#                  [0, 0, 1, 0, 0],
#                  [0, 0, 0, 1, 0],
#                  [0, 1, 0, 0, 0],
#                  [0, 1, 0, 0, 0],
#                  [0, 0, 0, 1, 1],
#                  [0, 0, 0, 1, 0],
#                  [0, 1, 0, 1, 1],
#                  [0, 0, 0, 1, 0],
#                  [0, 0, 1, 0, 0],
#                  [0, 0, 0, 1, 0],
#                  [0, 0, 0, 1, 0],
#                  [0, 0, 1, 0, 0],
#                  [0, 0, 0, 1, 1],
#                  [0, 0, 0, 1, 0],
#                  [0, 0, 1, 0, 0]])
# outputs = tensor([[-0.4876, -0.2527, -0.3669, -0.2799, -0.2925],
#                   [-0.4276, -0.1295, -0.5306, -0.4016, -0.0301],
#                   [-1.5691, -0.3117, -0.3266, -0.3947, -0.4907],
#                   [-0.5271, -0.2870, -0.2821, -0.5109, -0.0047],
#                   [-0.3758, -0.3472, -0.4557, -0.4918, -0.2659],
#                   [-0.2579, -0.3059, -0.1345, -0.4152, -0.0352],
#                   [-0.6828, -0.1146, -0.1137, -0.8153, -0.3100],
#                   [-0.9924, -0.2871, -0.2073, -0.4455, -0.4459],
#                   [-0.7734, -0.2190, -0.5503, -0.4954, -0.4202],
#                   [-0.7389, -0.2793, -0.2932, -0.4532, -0.0665],
#                   [-1.0988, -0.0056, -0.1935, -0.3867, -0.4171],
#                   [-1.3567, -0.2116, -0.4937, -0.5897, -0.5537],
#                   [-0.6358, -0.1169, -0.2563, -0.5143, -0.1612],
#                   [-0.5724, -0.1514, 0.0200, -0.2883, 0.0735],
#                   [-0.5514, -0.2125, -0.5116, -0.5768, -0.0931],
#                   [-1.0246, -0.0345, -0.3830, -0.4542, 0.2740],
#                   [-0.5496, 0.0290, -0.4044, -0.5198, -0.1594],
#                   [-0.4939, -0.1391, -0.0868, -0.4468, -0.1608],
#                   [-0.2371, -0.2796, -0.1947, -0.4110, -0.2085],
#                   [-0.5368, -0.2321, -0.2800, -0.6769, -0.2264]], )
# outputs = torch.sigmoid(outputs.float())

label2 = tensor([[0, 0, 1, 0, 0]])
output2 = tensor([[0.5, 1, 2, 0, 0]])
output2 = torch.sigmoid(output2.float())


# print(outputs)

def bce(output_sigmoid, target):
    batch_size = output_sigmoid.shape[0]  # 有几条样本的输出结果
    loss = 0
    for i in range(batch_size):  # 对每条样本
        for j in range(len(output_sigmoid[i])):  # 对每条样本所有类别
            if target[i][j] == 0:
                temp_loss = torch.log(1-output_sigmoid[i][j])
            else:
                temp_loss = torch.log(output_sigmoid[i][j])
            loss = loss - temp_loss
    return loss/(batch_size * output_sigmoid.shape[1])

def lsep(scores, labels, weights=None):
    mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
             labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
    diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
             scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
    return diffs.exp().mul(mask).sum().add(1).log().mean()

# loss = loss(outputs, labels.float())
loss_bce = bce(output2, label2.float())
loss_lsep = lsep(output2, label2.float())
print(loss_lsep)
