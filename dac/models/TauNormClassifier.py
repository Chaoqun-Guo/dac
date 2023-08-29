# -*- coding: utf-8 -*-
"""
@File: TauNormClassifier.py
@Time: 2023/08/25 18:29:09
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: TauNorm classifier.
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class TauNormClassifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048, *args):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        self.scales = Parameter(torch.ones(num_classes))
        for param_name, param in self.fc.named_parameters():
            param.requires_grad = False

    def forward(self, x, *args):
        x = self.fc(x)
        x *= self.scales
        return x, None
