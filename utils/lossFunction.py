def lsep(scores, labels, weights=None):
    mask = ((labels.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)) -
             labels.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1))) > 0).float()
    diffs = (scores.unsqueeze(2).expand(labels.size(0), labels.size(1), labels.size(1)) -
             scores.unsqueeze(1).expand(labels.size(0), labels.size(1), labels.size(1)))
    return diffs.exp().mul(mask).sum().add(1).log().mean()