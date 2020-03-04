import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=0.5, size_average=True, device='cpu'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha]).to(device)
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha).to(device)
        self.size_average = size_average

    def forward(self, proba, target):
        proba = proba.view(-1)
        target = target.view(-1)
        alpha = self.alpha[target.long()]
        proba = target * proba + (1. - target) * (1. - proba)
        logproba = proba.log()
        loss = - alpha * (1-proba)**self.gamma * logproba
        if self.size_average:
            return loss.sum() / alpha.sum()
        else:
            return loss.sum()
