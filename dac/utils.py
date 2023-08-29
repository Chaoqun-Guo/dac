# -*- coding: utf-8 -*-
"""
@File: utils.py
@Time: 2023/08/25 13:46:19
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: utils to training.
"""
from collections import OrderedDict
import importlib
import numpy as np
import matplotlib.pyplot as plt
import yaml
from typing import Any, Optional
import argparse
import os.path as osp


def source_import(file_path):
    """Import python module directly from source code using importlib.
    @Time: 2023/08/25 13:47:13
    @Params: 
        param: desc
    @Return: 

    """
    spec = importlib.util.spec_from_file_location('', file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def batch_show(inp, title=None):
    """Imshow for tensor.
    @Time: 2023/08/25 13:48:26
    @Params: 
        param: desc
    @Return: 

    """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.figure(figsize=(20, 20))
    plt.imshow(inp)
    if title is not None:
        plt.title(title)


def get_value(oldv, newv):
    if newv is not None:
        return newv
    else:
        return oldv


class ConfigManager(object):
    """Experiment configuration container, whose values can be access as property or by indexing.
    * Case insensitive

    # Initialize
    ConfigManager():  initalize an empty config
    ConfigManager.init_with_dict(dict)
    ConfigManager.init_with_yaml(yaml path)
    ConfigManager.init_with_namespace(namespace)

    # Subset of configs
    cfg.subset(*keys)
    cfg.except(*keys)

    # Update
    cfg.override_with_dict(dict)
    cfg.override_with_yaml(yaml path)
    cfg.override_with_namespace(namespace)
    """

    def __init__(self):
        self.__cfg = {}

    @classmethod
    def init_with_dict(cls, dict_cfg: dict):
        assert isinstance(dict_cfg, dict)
        instance = cls()
        for key, value in dict_cfg.items():
            key = key.lower()
            if isinstance(value, dict):
                value = cls.init_with_dict(dict_cfg=value)
            instance.__cfg[key] = value
        return instance

    @classmethod
    def init_with_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as fp:
            a_dict = yaml.load(fp, yaml.FullLoader)
        return cls.init_with_dict(dict_cfg=a_dict)

    def subset(self, *keys: str):
        include_keys = {x.lower() for x in keys}
        cfg = ConfigManager()
        cfg.__cfg = {k: v for k, v in self.__cfg.items() if k in include_keys}
        return cfg

    def excepts(self, *keys: str):
        exclude_keys = {x.lower() for x in keys}
        cfg = ConfigManager()
        cfg.__cfg = {k: v for k, v in self.__cfg.items()
                     if k not in exclude_keys}
        return cfg

    def override_with_dict(self, cfg: dict, strict: bool = True):
        """override configs with a `dict`"""
        assert isinstance(cfg, dict)

        for key, value in cfg.items():
            key = key.lower()
            if key not in self.__cfg:
                if strict:
                    raise KeyError(key)
                else:
                    self.__cfg[key] = ConfigManager()

            if isinstance(value, dict):
                self.__cfg[key].override_with_dict(value, strict)
            else:
                self.__cfg[key] = value

    def override_with_yaml(self, yaml_path: str, strict: bool = True):
        """override configs with a yaml file"""
        with open(yaml_path, "r") as fp:
            a_dict = yaml.load(fp, yaml.FullLoader)
            self.override_with_dict(a_dict, strict)

    def update(self, key: str, new_val: Any):
        if "." in key:
            key_lst = key.split(".")
            key_this, key_rest = key_lst[0], ".".join(key_lst[1:])
            self.__cfg[key_this].update(key_rest, new_val)
        else:
            self.__cfg[key] = new_val

    def to_dict(self) -> dict:
        return self.__cfg

    def __index__(self, key: str):
        key = key.lower()
        return self.__cfg[key]

    def get(self, name: str, default: Optional[Any] = None):
        name = name.lower()
        if name in self.__cfg:
            return self.__cfg[name]
        else:
            return default

    def __getattr__(self, name: str):
        name = name.lower()
        if name in self.__cfg:
            return self.__cfg[name]
        else:
            raise AttributeError(f"config term {name} does not exist.")

    def __contains__(self, key: str):
        key = key.lower()
        return key in self.__cfg


def parse_args(cmd_str=None):
    """parser args.
    @Time: 2023/08/25 15:59:05
    @Params: 
        param: desc
    @Return: 

    """
    parser = argparse.ArgumentParser(description="Divide and Conquer.")
    # basic args
    # parser.add_argument("--name", type=str)
    parser.add_argument("--cfg", default=None, type=str)
    parser.add_argument("--force", default=False, action="store_true")
    parser.add_argument("--only_cluster", default=False, action="store_true")

    # resume args
    parser.add_argument("--ft_model_weight", type=str)
    parser.add_argument("--cls_weight", type=str)

    # parallel args
    parser.add_argument("--local_rank", type=int, default=False)
    parser.add_argument("--distributed", default=False, action="store_true")
    parser.add_argument("--dist_url", type=str)

    if cmd_str:
        cmd_args = parser.parse_args(cmd_str.split())
    else:
        cmd_args = parser.parse_args()

    config = ConfigManager.init_with_yaml("./conf/BaseConfig.yaml")
    config.override_with_yaml(cmd_args.cfg)
    cmd_args = vars(cmd_args)
    config.override_with_dict(cmd_args, strict=False)

    if config.name is None:
        config.update("name", osp.splitext(osp.basename(config.cfg))[0])

    if config.ft_model_weight is None:
        config.update("ft_model_weight", [])
    else:
        config.update(
            "ft_mpdel_weight", config.ft_model_weight.split(","))

    if config.cls_weight is None:
        config.update("cls_weight", [])
    else:
        config.update("cls_weight", config.cls_weight.split(","))

    return config


def state_dict_strip_prefix(state_dict, prefix="module."):
    state_dict = OrderedDict({
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    })
    return state_dict
