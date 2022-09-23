import torch
import torch.nn as nn
import numpy as np

from abc import ABCMeta, abstractmethod

def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class BaseHead(nn.Module, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - Methods:``init_weights``, initializing weights in some modules.
    - Methods:``forward``, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss', loss_weight=1.0).
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Default: 0.
        topk (int | tuple): Top-k accuracy. Default: (1, 5).
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class=False,
                 label_smooth_eps=0.0,
                 topk=(1, 5)):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        ##
        # loss build
        #self.loss_cls = build_loss(loss_cls)
        self.loss_cls = BinaryLogisticRegressionLoss()
        ##
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

    @abstractmethod
    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""

    @abstractmethod
    def forward(self, x):
        """Defines the computation performed at every call."""

    def loss(self, cls_score, labels, **kwargs):
        """Calculate the loss given output ``cls_score``, target ``labels``.
        Args:
            cls_score (torch.Tensor): The output of the model.
            labels (torch.Tensor): The target output of the model.
        Returns:
            dict: A dict containing field 'loss_cls'(mandatory) and 'topk_acc'(optional).
        """
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_score.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_socre` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if not self.multi_class and cls_score.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_score.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_score.device)

        elif self.multi_class and self.label_smooth_eps != 0:
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_score, labels, **kwargs)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls

        return losses


def binary_logistic_regression_loss(reg_score, label, threshold=0.5,
                                    ratio_range=(1.05, 21), eps=1e-5):
    """Binary Logistic Regression Loss."""
    label = label.view(-1).to(reg_score.device)
    reg_score = reg_score.contiguous().view(-1)

    pmask = (label > threshold).float().to(reg_score.device)
    num_positive = max(torch.sum(pmask), 1)
    num_entries = len(label)
    ratio = num_entries / num_positive
    # clip ratio value between ratio_range
    ratio = min(max(ratio, ratio_range[0]), ratio_range[1])

    coef_0 = 0.5 * ratio / (ratio - 1)
    coef_1 = 0.5 * ratio
    loss = coef_1 * pmask * torch.log(reg_score + eps) + coef_0 * (
        1.0 - pmask) * torch.log(1.0 - reg_score + eps)
    loss = -torch.mean(loss)
    return loss


class BinaryLogisticRegressionLoss(nn.Module):
    """Binary Logistic Regression Loss.
    It will calculate binary logistic regression loss given reg_score and label.
    """
    def forward(self, reg_score, label, threshold=0.5,
                ratio_range=(1.05, 21), eps=1e-5):
        """Calculate Binary Logistic Regression Loss.
        Args:
                reg_score (torch.Tensor): Predicted score by model.
                label (torch.Tensor): Groundtruth labels.
                threshold (float): Threshold for positive instances.
                    Default: 0.5.
                ratio_range (tuple): Lower bound and upper bound for ratio.
                    Default: (1.05, 21)
                eps (float): Epsilon for small value. Default: 1e-5.

        Returns:
                torch.Tensor: Returned binary logistic loss.
        """

        return binary_logistic_regression_loss(reg_score, label, threshold,
                                               ratio_range, eps)


def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.

    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).

    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = []
    labels = np.array(labels)[:, np.newaxis]
    for k in topk:
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res.append(topk_acc_score)

    return res


class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Default: 1.
    """

    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class TSNHead(BaseHead):
    """Class head for TSN.
    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 spatial_type='avg',
                 consensus=dict(type='AvgConsensus', dim=1),
                 dropout_ratio=0.4,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std

        consensus_ = consensus.copy()

        consensus_type = consensus_.pop('type')
        if consensus_type == 'AvgConsensus':
            self.consensus = AvgConsensus(**consensus_)
        else:
            self.consensus = None

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool2d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.avg_pool = None

        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x, num_segs):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N * num_segs, in_channels, 7, 7]
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
            # [N * num_segs, in_channels, 1, 1]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        # [N, num_segs, in_channels, 1, 1]
        x = self.consensus(x)
        # [N, 1, in_channels, 1, 1]
        x = x.squeeze(1)
        # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


class TPNHead(TSNHead):
    """Class head for TPN.
    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Default: False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: https://arxiv.org/abs/1906.02629. Default: 0.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool3d = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool3d = None

        self.avg_pool2d = None
        self.new_cls = None

    def _init_new_cls(self):
        self.new_cls = nn.Conv3d(self.in_channels, self.num_classes, 1, 1, 0)
        if next(self.fc_cls.parameters()).is_cuda:
            self.new_cls = self.new_cls.cuda()
        self.new_cls.weight.copy_(self.fc_cls.weight[..., None, None, None])
        self.new_cls.bias.copy_(self.fc_cls.bias)

    def forward(self, x, num_segs=None, fcn_test=False):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
            num_segs (int | None): Number of segments into which a video
                is divided. Default: None.
            fcn_test (bool): Whether to apply full convolution (fcn) testing.
                Default: False.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        if fcn_test:
            if self.avg_pool3d:
                x = self.avg_pool3d(x)
            if self.new_cls is None:
                self._init_new_cls()
            cls_score_feat_map = self.new_cls(x)
            return cls_score_feat_map

        if self.avg_pool2d is None:
            kernel_size = (1, x.shape[-2], x.shape[-1])
            self.avg_pool2d = nn.AvgPool3d(kernel_size, stride=1, padding=0)

        if num_segs is None:
            # [N, in_channels, 3, 7, 7]
            x = self.avg_pool3d(x)
        else:
            # [N * num_segs, in_channels, 7, 7]
            x = self.avg_pool2d(x)
            # [N * num_segs, in_channels, 1, 1]
            x = x.reshape((-1, num_segs) + x.shape[1:])
            # [N, num_segs, in_channels, 1, 1]
            x = self.consensus(x)
            # [N, 1, in_channels, 1, 1]
            x = x.squeeze(1)
            # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
            # [N, in_channels, 1, 1]
        x = x.view(x.size(0), -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score


if __name__ == "__main__":

    tensor = torch.randn([12, 32, 8, 112, 112])

    head = TPNHead(num_classes=5, in_channels=32,
                   loss_cls=dict(type='BinaryCrossEntropyLoss'),
                   spatial_type='avg', multi_class=True)

    res = head(tensor)
    print(res.shape)

    reg_Score = torch.rand([12, 5], dtype=torch.float32)
    reg_Score2 = torch.randn([12, 5], dtype=torch.float32)
    loss = BinaryLogisticRegressionLoss()(reg_Score, reg_Score2)
    print(loss)