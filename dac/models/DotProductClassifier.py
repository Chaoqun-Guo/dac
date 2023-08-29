# -*- coding: utf-8 -*-
"""
@File: DotProductClassifier.py
@Time: 2023/08/25 18:31:56
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: Dot Product classifier.
"""
import torch
import torch.nn as nn


class DotProductClassifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048):
        super().__init__()
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        x = self.fc(x)
        return x
