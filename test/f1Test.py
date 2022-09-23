import numpy as np
import torch
from sklearn import metrics
from torch import tensor


def compute_mAP(y_true, y_pred):
    AP = []
    for i in range(y_true.shape[1]):
        AP.append(metrics.average_precision_score(y_true[:, i], y_pred[:, i], average='macro', pos_label=1))
    return np.mean(AP)


sum_target = np.array([[0, 1, 1, 0, 1],
                       [0, 1, 1, 0, 1],
                       [0, 1, 1, 0, 1]])

sum_probs = np.array([[0, 1, 1, 0, 1],
                      [1, 1, 1, 0, 1],
                      [0, 1, 1, 0, 1]])

# sum_target = torch.cat(sum_target).cpu().detach().numpy()
# sum_probs = torch.cat(sum_probs).cpu().detach().numpy()

a = metrics.f1_score(sum_target, sum_probs, average='macro')
recall = metrics.recall_score(sum_target, sum_probs, average='macro')
macroAP = metrics.average_precision_score(sum_target, sum_probs, average='macro')
microAP = metrics.average_precision_score(sum_target, sum_probs, average='micro')
macroAP2 = compute_mAP(sum_target, sum_probs)
metrics.average_precision_score(sum_target, sum_probs)
# metrics.f1_score(sum_target, sum_probs)
# metrics.f1_score(sum_target, sum_probs)
# metrics.f1_score(sum_target,X sum_probs)
