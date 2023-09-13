# -*- coding: utf-8 -*-
"""
@File: ClusterAidedClassifier.py
@Time: 2023/08/25 18:18:38
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: ClusterAidedClassifier
"""

from collections import OrderedDict
import torch
import torch.nn as nn
from dac.models.CosNormClassifier import CosNormClassifier

from dac.models.NetBlocks import MLP
from dac.models.TauNormClassifier import TauNormClassifier
from dac.utils import state_dict_strip_prefix


class ClusterAidedClassifier(nn.Module):
    def __init__(self, model_type, feat_dim, num_class):
        super().__init__()
        assert model_type in ["XZP", "XZ"], f"model_type {model_type} error!"
        self.model_type = model_type
        self.out_dim = num_class
        self.zp_dim = {
            "XZ": feat_dim,
            "XZP": feat_dim+num_class,
        }[model_type]

        def unImpl(x):
            raise NotImplementedError()

        self.cls_x = unImpl
        self.cls_zp = unImpl

    def forward(self, x, z, p=None):
        if p is None:
            zp = z
        else:
            zp = torch.cat([z, p], 1)

        logit_x = self.cls_x(x)
        logit_zp = self.cls_zp(zp)
        logit = logit_x + logit_zp
        return logit


class LinearClusterAidedClassifier(ClusterAidedClassifier):
    def __init__(self, model_type, feat_dim, num_class, *args, **kwargs):
        super().__init__(model_type, feat_dim, num_class)
        self.cls_x = MLP(feat_dim, num_class, *args, **kwargs)
        self.cls_zp = MLP(self.zp_dim, num_class, *args, **kwargs)


class CosineClusterAidedClassifier(ClusterAidedClassifier):
    def __init__(self, model_type, feat_dim, num_class, *args, **kwargs):
        super().__init__(model_type, feat_dim, num_class)
        self.cls_x = CosNormClassifier(feat_dim, num_class, *args, **kwargs)
        self.cls_zp = CosNormClassifier(
            self.zp_dim, num_class, *args, **kwargs)


class TauClusterAidedClassifier(ClusterAidedClassifier):
    def __init__(self, model_type, feat_dim, num_class, *args, **kwargs):
        super().__init__(model_type, feat_dim, num_class)
        self.cls_x = TauNormClassifier(feat_dim, num_class)
        self.cls_zp = TauNormClassifier(self.zp_dim, num_class,)
