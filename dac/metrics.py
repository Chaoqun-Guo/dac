# -*- coding: utf-8 -*-
"""
@File: metrics.py
@Time: 2023/08/25 13:42:20
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: define shot-acc, so on
"""
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import logging
import dac.logger
import torch
from sklearn.metrics import f1_score, normalized_mutual_info_score as NmiMetric
from sklearn.metrics import matthews_corrcoef as MccMetric
import torch.nn.functional as F
import math
from easydict import EasyDict
from itertools import zip_longest


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if epoch < args.warmup_epochs:
        lr = round(lr / args.warmup_epochs * (epoch + 1), 5)
    elif args.cos_lr:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup_epochs +
                     1) / (args.epoch - args.warmup_epochs + 1)))
    else:  # stepwise lr schedule
        for milestone in args.steps:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    # logging.info(f"LR: {lr}")


def adjust_learning_rate_v1(optimizer, epoch, warmup_epochs=20, total_epoch=150, lr=1e-4, use_cos_lr=False, steps=[500, 1000]):
    """adjust learning rate to train.
    @Time: 2023/08/25 14:11:06
    @Params: 
        param: desc
    @Return: 

    """
    lr = lr
    if epoch < warmup_epochs:
        lr /= warmup_epochs*(epoch+1)
    elif use_cos_lr:
        lr *= 0.5*(1.0+math.cos(math.pi*(epoch-warmup_epochs+1) /
                   (total_epoch-warmup_epochs+1)))
    else:
        for milestone in steps:
            lr *= 0.1 if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_weights(model, weights_path, caffe=False, classifier=False) -> torch.nn.Module:
    """Initialize weights"""
    print('Pretrained %s weights path: %s' % ('classifier' if classifier else 'feature model',
                                              weights_path))
    weights = torch.load(weights_path)
    if not classifier:
        if caffe:
            weights = {k: weights[k] if k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
        else:
            weights = weights['state_dict_best']['feat_model']
            weights = {k: weights['module.' + k] if 'module.' + k in weights else model.state_dict()[k]
                       for k in model.state_dict()}
    else:
        weights = weights['state_dict_best']['classifier']
        weights = {k: weights['module.fc.' + k] if 'module.fc.' + k in weights else model.state_dict()[k]
                   for k in model.state_dict()}
    model.load_state_dict(weights)
    return model


def shot_acc(preds, labels, train_data, many_shot_thr=1500, low_shot_thr=300, acc_per_cls=False):
    """shot acc.
    @Time: 2023/08/25 13:54:32
    @Params: 
        param: desc
    @Return: 

    """
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))

    train_class_count = []
    test_class_count = []
    class_correct = []

    for il in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == il]))
        test_class_count.append(len(labels[labels == il]))
        class_correct.append(
            (preds[labels == il] == labels[labels == il]).sum())

    logging.info(
        f"Train class count: {train_class_count}, total: {np.sum(train_class_count)}")
    logging.info(
        f"Test class count: {test_class_count}, total: {np.sum(test_class_count)}")
    logging.info(f"Class correct: {class_correct}")

    many_shot = []
    median_shot = []
    low_shot = []

    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))

    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    if acc_per_cls:
        class_accs = [c / cnt for c,
                      cnt in zip(class_correct, test_class_count)]

        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot), class_accs
    else:
        return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def weighted_shot_acc(preds, labels, ws, train_data, many_shot_thr=100, low_shot_thr=20):
    """weighted shot acc.
    @Time: 2023/08/25 13:54:16
    @Params: 
        param: desc
    @Return: 

    """

    training_labels = np.array(train_data.dataset.labels).astype(int)
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))
    train_class_count = []
    test_class_count = []
    class_correct = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        test_class_count.append(ws[labels == l].sum())
        class_correct.append(
            ((preds[labels == l] == labels[labels == l]) * ws[labels == l]).sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_correct[i] / test_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_correct[i] / test_class_count[i]))
        else:
            median_shot.append((class_correct[i] / test_class_count[i]))
    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def mcc_mni_metrics(preds, labels):
    return MccMetric(labels, preds), NmiMetric(labels, preds)


def F_measure(preds, labels, openset=False, theta=None):
    """f measure.
    @Time: 2023/08/25 13:55:15
    @Params: 
        param: desc
    @Return: 

    """
    if openset:
        # f1 score for openset evaluation
        true_pos = 0.
        false_pos = 0.
        false_neg = 0.
        for i in range(len(labels)):
            true_pos += 1 if preds[i] == labels[i] and labels[i] != -1 else 0
            false_pos += 1 if preds[i] != labels[i] and labels[i] != - \
                1 and preds[i] != -1 else 0
            false_neg += 1 if preds[i] != labels[i] and labels[i] == -1 else 0

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        return 2 * ((precision * recall) / (precision + recall + 1e-12))
    else:
        # Regular f1 score
        return f1_score(labels.detach().cpu().numpy(), preds.detach().cpu().numpy(), average='macro')


def mic_acc_cal(preds, labels):
    """desc
    @Time: 2023/08/25 13:55:48
    @Params: 
        param: desc
    @Return: 

    """
    if isinstance(labels, tuple):
        assert len(labels) == 3
        targets_a, targets_b, lam = labels
        acc_mic_top1 = (lam * preds.eq(targets_a.data).cpu().sum().float()
                        + (1 - lam) * preds.eq(targets_b.data).cpu().sum().float()) / len(preds)
    else:
        acc_mic_top1 = (preds == labels).sum().item() / len(labels)
    return acc_mic_top1


def weighted_mic_acc_cal(preds, labels, ws):
    acc_mic_top1 = ws[preds == labels].sum() / ws.sum()
    return acc_mic_top1


def class_count(data):
    labels = np.array(data.dataset.labels)
    class_data_num = []
    for l in np.unique(labels):
        class_data_num.append(len(labels[labels == l]))
    return class_data_num


def torch2numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, (list, tuple)):
        return tuple([torch2numpy(xi) for xi in x])
    else:
        return x


def logits2score(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy()
    return score


def logits2entropy(logits):
    scores = F.softmax(logits, dim=1)
    scores = scores.cpu().numpy() + 1e-30
    ent = -scores * np.log(scores)
    ent = np.sum(ent, 1)
    return ent


def logits2CE(logits, labels):
    scores = F.softmax(logits, dim=1)
    score = scores.gather(1, labels.view(-1, 1))
    score = score.squeeze().cpu().numpy() + 1e-30
    ce = -np.log(score)
    return ce


def get_priority(ptype, logits, labels):
    if ptype == 'score':
        ws = 1 - logits2score(logits, labels)
    elif ptype == 'entropy':
        ws = logits2entropy(logits)
    elif ptype == 'CE':
        ws = logits2CE(logits, labels)
    return ws


def numpy2torch(item, to_long=False) -> Any:
    """dict of numpy -> torch"""
    if isinstance(item, torch.Tensor):
        if to_long:
            return item.long()
        else:
            return item

    elif isinstance(item, np.ndarray):
        if to_long:
            return torch.from_numpy(item).long()
        else:
            return torch.from_numpy(item)

    elif isinstance(item, list):
        return [numpy2torch(x, to_long) for x in item]

    elif type(item) == dict:
        return {k: numpy2torch(v, to_long) for k, v in item.items()}

    elif type(item) == EasyDict:
        return EasyDict({k: numpy2torch(v, to_long) for k, v in item.items()})

    else:
        raise NotImplementedError(str(type(item)))
