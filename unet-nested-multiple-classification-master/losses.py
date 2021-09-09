# -*- coding: utf-8 -*-
# @Time    : 2020-02-26 17:46
# @Author  : Zonas
# @Email   : zonas.wang@gmail.com
# @File    : losses.py
"""

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from LovaszSoftmax.pytorch import lovasz_losses as L


class LovaszLossSoftmax(nn.Module):
    def __init__(self):
        super(LovaszLossSoftmax, self).__init__()

    def forward(self, input, target):
        out = F.softmax(input, dim=1)
        loss = L.lovasz_softmax(out, target)
        return loss


class LovaszLossHinge(nn.Module):
    def __init__(self):
        super(LovaszLossHinge, self).__init__()

    def forward(self, input, target):
        loss = L.lovasz_hinge(input, target)
        return loss

# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch?scriptVersionId=68471013&cellId=4
# the next 2 dice loss have problems
class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class DiceCoeff(Function):
    """Dice coeff for individual examples"""
    @staticmethod
    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

