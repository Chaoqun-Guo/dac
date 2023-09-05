# -*- coding: utf-8 -*-
"""
@File: data_utils.py
@Time: 2023/08/25 16:29:15
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: data utils
"""

import numpy as np
import torch
import torch.utils.data as tdata
from collections import defaultdict
import h5py
import pandas as pd
import os.path as osp
import logging
from easydict import EasyDict
from torch.utils.data import DataLoader, TensorDataset
from dac.data import ClassAwareSampler
import dac.logger
from dac.metrics import numpy2torch


class ContrastiveDataset(tdata.Dataset):
    """Simple dataset"""

    def __init__(self, inputs, labels, weight_q):
        super().__init__()
        logging.info(f"Using simplified sampler with q = {weight_q}")
        self.inputs = inputs.cpu()
        self.labels = labels.cpu()

        self.ids_per_class = defaultdict(list)
        for i, x in enumerate(labels.cpu().numpy().tolist()):
            self.ids_per_class[x].append(i)
        self.ids_per_class = {i: v for i,
                              v in enumerate(self.ids_per_class.values())}

        self.num_classes = len(self.ids_per_class)
        self.num_per_class = [len(self.ids_per_class[i])
                              for i in range(self.num_classes)]
        logging.info(f"num per class: {self.num_per_class}")
        max_class_size = np.max(self.num_per_class)*1.0

        self.reverse_freq = np.array(
            [(max_class_size/x)**(weight_q) for x in self.num_per_class])
        # self.reverse_freq = np.minimum(self.reverse_freq, 64)
        self.reverse_freq = self.reverse_freq / np.sum(self.reverse_freq)
        logging.info(
            f"Average weight:  {np.sum(self.reverse_freq * np.array(self.num_per_class)) / np.sum(self.num_per_class)}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        pos_y, neg_y = np.random.choice(
            np.arange(self.num_classes), size=2, replace=False).tolist()
        pos_idx = np.random.choice(self.ids_per_class[pos_y])
        neg_idx = np.random.choice(self.ids_per_class[neg_y])
        pos_x = self.inputs[pos_idx]
        neg_x = self.inputs[neg_idx]

        return (pos_x, neg_x, self.reverse_freq[pos_y])


def read_h5_data(args):
    """Read h5 format data.
    @Time: 2023/08/25 16:39:56
    @Params: 
        param: desc
    @Return: 

    """
    split_name = {"train": "train", "test": "test"}
    features = EasyDict()
    labels = EasyDict()
    for split in ["train", "test"]:
        with h5py.File(osp.join(args.feat_dir, f"feature_{split_name[split]}.h5"), "r") as fp:
            for key in fp.keys():
                print(key, fp.get(key).shape)
            features[split] = torch.from_numpy(
                fp.get("features")[...]).float().to(args.device)
            if "label" in fp:  # compatible
                labels[split] = torch.from_numpy(
                    fp.get("label")[...]).float().to(args.device)
            else:
                labels[split] = torch.from_numpy(
                    fp.get("labels")[...]).float().to(args.device)
    return features, labels


def make_tensor_dataloader(np_array_list, mask=None, key=str, batch_size=int, shuffle=bool, sampler=None, num_workers=0):
    """make tensor dataloader from given np array list with mask.
    @Time: 2023/08/25 16:41:18
    @Params: 
        param: desc
    @Return: 

    """
    if mask is not None:
        mask = mask[key]
        tensors = [dt[key][mask, ...] for dt in np_array_list]
    else:
        tensors = [dt[key] for dt in np_array_list]
    tensors = numpy2torch(tensors)
    return tdata.DataLoader(tdata.TensorDataset(*tensors), batch_size, shuffle=shuffle, sampler=sampler, num_workers=num_workers)


def get_data_sampler(sampler_arg, train_labels):
    if sampler_arg.name is None or sampler_arg.name == "":
        return None
    if sampler_arg.name == "CBS":
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.cpu().numpy()
        return ClassAwareSampler(train_labels, sampler_arg.num_samples_cls)


def make_imb_data(num_max, num_classes, imb_ratio):
    mu = np.power(1 / imb_ratio, 1 / (num_classes - 1))
    class_num_list = []
    for i in range(num_classes):
        if i == (num_classes - 1):
            class_num_list.append(int(num_max / imb_ratio))
        else:
            class_num_list.append(int(num_max * np.power(mu, i)))
    num_per_class = class_num_list
    return list(num_per_class)


def get_data_idx(labels, num_per_class):

    labels = np.array(labels)
    label_idx = []
    need_label = []
    final_label = []

    for i in range(len(num_per_class)):
        idx = np.where(labels == i)[0]
        logging.info(f"Class {i} with {len(idx)} examples.")
        if num_per_class[i] > len(idx):
            logging.warning("Sampled number is bigger than total to train for class: {}, max is {}".format(
                i, len(idx)))
            label_idx.extend(idx[-num_per_class[i]:])
            final_label.append(len(idx[-num_per_class[i]:]))
        else:
            np.random.shuffle(idx)
            label_idx.extend(idx[:num_per_class[i]])
            final_label.append(len(idx[:num_per_class[i]]))
        need_label.append(num_per_class[i])
    logging.info(
        f"\nlabel_idx: {len(label_idx)}\nneed_label: {need_label}\nfinal_label: {final_label}")
    return label_idx, need_label, final_label


def make_data_loader(datas: pd.DataFrame,  idx: list = None, train=True, batch_size=64, num_workers=0, return_df=False, keep_all=False):
    """make data laoder.
    @Time: 2023/08/25 22:07:10
    @Params: 
        param: desc
    @Return: 

    """
    if idx is None and not keep_all:
        raise ValueError(f"idx: {idx}")

    num_col = list(set(datas.columns)-set(['label']))
    data_ = datas[num_col].to_numpy()
    label_ = datas['label'].to_numpy()

    if keep_all:
        dataset = TensorDataset(torch.from_numpy(
            data_), torch.from_numpy(label_))

    if idx is not None:
        dataset = TensorDataset(torch.from_numpy(
            data_[idx]), torch.from_numpy(label_[idx]))

        data_dict = {f"f_{i+1}": data_[idx][:, i]
                     for i in range(data_[idx].shape[1])}
        data_dict['label'] = label_[idx]
    else:
        data_dict = {f"f_{i+1}": data_[:, i]
                     for i in range(data_.shape[1])}
        data_dict['label'] = label_

    df = pd.DataFrame(data_dict)

    if train:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        dataloader = DataLoader(dataset,
                                batch_size=batch_size, shuffle=False, num_workers=num_workers)

    if return_df:
        return df, dataloader
    return None, dataloader


def write2h5py(filename, features, labels):
    with h5py.File(filename, 'w')as f:
        f.create_dataset('features', data=features)
        f.create_dataset('labels', data=labels)


def build_dataset(dataset_name="dataset", num_max=3000, num_classes=10, imb_ratio=20):

    multi_train = pd.read_csv("./unsw_nb15/train_set_multi_classification.csv")
    multi_test = pd.read_csv("./unsw_nb15/test_set_multi_classification.csv")

    num_per_class = make_imb_data(num_max, num_classes, imb_ratio)

    label_idx, need_label, final_label = get_data_idx(
        multi_train['label'].to_numpy(), num_per_class)

    train_df, train_dataloader = make_data_loader(
        multi_train, label_idx, True, return_df=True, keep_all=False)
    test_df, test_dataloader = make_data_loader(
        multi_test, None, False, return_df=True, keep_all=True)

    logging.info(f"Train data: {len(train_df)}; Test data: {len(test_df)}")

    train_df.to_csv(
        f"./unsw_nb15/{dataset_name}_train_{num_max}_{imb_ratio}.csv", index=False)
    test_df.to_csv(
        f"./unsw_nb15/{dataset_name}_test_{num_max}_{imb_ratio}.csv")
    write2h5py(f'./unsw_nb15/feature_train.h5', train_df.drop('label',
               axis=1).to_numpy(), train_df['label'].to_numpy())
    write2h5py(f'./unsw_nb15/feature_test.h5', test_df.drop('label',
               axis=1).to_numpy(), test_df['label'].to_numpy())
    logging.info("Build dataset done, return dataset_df and dataloader")

    return train_dataloader, test_dataloader, train_df, test_df


if __name__ == "__main__":
    build_dataset('unsw_nb15', num_max=2000, num_classes=10, imb_ratio=500)
