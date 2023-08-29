# -*- coding: utf-8 -*-
"""
@File: main.py
@Time: 2023/08/25 13:36:47
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: main run file
"""

from logging import handlers
import os
import argparse
import pprint
import random
import warnings
import logging
import os.path as osp
import dac.logger
import torch
import torch.nn as nn
import numpy as np
import yaml
from dac.logger import prepare_log_dir
from dac.utils import parse_args
from dac.data.data_utils import read_h5_data
from dac.solver import *


def run():
    """run
    @Time: 2023/08/25 14:39:28
    @Params: 
        param: desc
    @Return: 

    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info("Initializiing...")
    cfg = parse_args()
    features, labels = read_h5_data(cfg)
    save_dir = prepare_log_dir(cfg)
    file_logger = logging.getLogger('file')
    file_logger_fh = handlers.RotatingFileHandler(
        filename=osp.join(save_dir, "run_logs.txt"), encoding='utf-8')
    file_logger_sh = logging.StreamHandler()
    file_logger.addHandler(file_logger_fh)
    file_logger.addHandler(file_logger_sh)
    logging.getLogger('file').info(
        f"num_max: {cfg.num_max}; imbalance_ratio: {cfg.imbalance_ratio}")
    divide_and_conquer(features, features, labels, cfg, save_dir, global_mask=None,
                       rec_depth=cfg.rec_depth, max_depth=cfg.max_depth)


if __name__ == "__main__":
    run()
