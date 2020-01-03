import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, proba, target):
        proba = proba.view(-1)
        target = target.view(-1)
        proba = target * proba + (1. - target) * (1. - proba)
        logproba = proba.log()
        loss = -1 * (1-proba)**self.gamma * logproba
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
