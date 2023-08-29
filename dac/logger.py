# -*- coding: utf-8 -*-
"""
@File: logger.py
@Time: 2023/08/25 13:27:19
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: define logger
"""

import glob
import os
import shutil
import yaml
import csv
import h5py
import logging
import os.path as osp
from datetime import datetime

logging.basicConfig(format='[%(asctime)s] %(filename)s %(lineno)d %(levelname)s %(message)s',
                    level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


class Logger():
    """Define logger to log results.
    @Time: 2023/08/25 13:50:51
    @Version : 0.0.1
    Attributes :
        Attributes: desc
    """

    def __init__(self, logdir):
        self.logdir = logdir
        if not os.path.isdir(logdir):
            os.makedirs(logdir)
        self.cfg_file = os.path.join(self.logdir, 'cfg.yaml')
        self.acc_file = os.path.join(self.logdir, 'acc.csv')
        self.loss_file = os.path.join(self.logdir, 'loss.csv')
        self.ws_file = os.path.join(self.logdir, 'ws.h5')
        self.acc_keys = None
        self.loss_keys = None
        self.logging_ws = False

    def log_cfg(self, cfg):
        print('===> Saving cfg parameters to: ', self.cfg_file)
        with open(self.cfg_file, 'w', encoding='utf-8') as f:
            yaml.dump(cfg, f)

    def log_acc(self, accs):
        if self.acc_keys is None:
            self.acc_keys = [k for k in accs.keys()]
            with open(self.acc_file, 'w', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                writer.writeheader()
                writer.writerow(accs)
        else:
            with open(self.acc_file, 'a', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.acc_keys)
                writer.writerow(accs)

    def log_loss(self, losses):
        # valid_losses = {k: v for k, v in losses.items() if v is not None}
        valid_losses = losses
        if self.loss_keys is None:
            self.loss_keys = [k for k in valid_losses.keys()]
            with open(self.loss_file, 'w', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writeheader()
                writer.writerow(valid_losses)
        else:
            with open(self.loss_file, 'a', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.loss_keys)
                writer.writerow(valid_losses)

    def log_ws(self, epoch, ws):
        mode = 'a' if self.logging_ws else 'w'
        self.logging_ws = True
        key = 'Epoch {:02d}'.format(epoch)
        with h5py.File(self.ws_file, mode) as f:
            g = f.create_group(key)
            for k, v in ws.items():
                g.create_dataset(k, data=v)


def print_write(print_str, log_file):
    """Print and write to log file.
    @Time: 2023/08/25 13:50:23
    @Params: 
        param: desc
    @Return: 

    """
    logging.info(" ".join(map(str, print_str)))
    if log_file is None:
        return
    with open(log_file, 'a', encoding='utf-8') as f:
        print(*print_str, file=f)


def prepare_log_dir(cfg):
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = osp.join(
        cfg.root_dir, f"{cfg.name}_{now}_Nmax{cfg.num_max}_Imi{cfg.imbalance_ratio}")
    if osp.exists(save_dir):
        if cfg.force:
            for f in glob.glob(save_dir+"/*"):
                logging.warning(f"Removing {f}")
                os.remove(f)
        else:
            raise FileExistsError(f"{save_dir} exists")
    else:
        os.makedirs(save_dir)
    logging.info(f"Save to {save_dir}")
    shutil.copy(cfg.cfg, osp.join(save_dir, "config.yaml"))
    return save_dir
