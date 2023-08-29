# -*- coding: utf-8 -*-
"""
@File: CosNormClassifier.py
@Time: 2023/08/25 18:23:24
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: 
"""
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math


class CosNormClassifier(nn.Module):
    def __init__(self, feat_dim, num_class, scale=16, margin=0.5, *args, **kwargs):
        super().__init__()
        self.in_dims = feat_dim
        self.out_dims = num_class
        self.scale = scale
        self.margin = margin
        self.weight = Parameter(torch.Tensor(
            self.out_dims, self.in_dims).cuda())
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, inputs, *args):
        norm_x = torch.norm(inputs.clone(), dim=1, keepdim=True)
        ex = (norm_x / (1 + norm_x)) * (inputs / norm_x)
        ew = self.weight / torch.norm(self.weight, dim=1, keepdim=True)
        return torch.mm(self.scale * ex, ew.t())
