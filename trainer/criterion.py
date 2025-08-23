import torch
import torch.nn as nn
import torch.nn.functional as F


class binary_entropy(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()

        self.NAME = 'binary_entropy'
        self.TYPE = 'prediction'
        self.reduction = reduction

    def forward(self, inputs, targets, weights=None):
        return F.binary_cross_entropy_with_logits(inputs, targets, weight=weights, reduction=self.reduction)


class cross_entropy(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()

        self.NAME = 'cross_entropy'
        self.TYPE = 'prediction'
        self.reduction = reduction

    def forward(self, inputs, targets):
        targets = targets.squeeze(-1)
        return F.cross_entropy(inputs, targets, reduction=self.reduction)


CRITERIONS = {
    'binary_entropy': binary_entropy,
    'cross_entropy': cross_entropy
}
