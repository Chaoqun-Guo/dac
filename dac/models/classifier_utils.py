# -*- coding: utf-8 -*-
"""
@File: classifier_utils.py
@Time: 2023/08/25 17:23:55
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: classifier utils.
"""
import torch
import torch.utils.data as tdata
import torch.nn.utils
import copy
import numpy as np
import logging
import dac.logger
from dac.loss.losses import get_classification_loss
from dac.metrics import adjust_learning_rate, shot_acc


from dac.models.ClusterAidedClassifier import CosineClusterAidedClassifier, LinearClusterAidedClassifier, TauClusterAidedClassifier


def get_classifier(cls_args, args):
    if cls_args.type == "linear":
        ModelClass = LinearClusterAidedClassifier
    elif cls_args.type == "cosine":
        ModelClass = CosineClusterAidedClassifier
    elif cls_args.type == 'tau':
        ModelClass = TauClusterAidedClassifier
    else:
        raise NotImplementedError(cls_args.type)
    return ModelClass(cls_args.inputs,
                      args.feat_dim,
                      args.num_class,
                      hidden_layers=cls_args.hidden_layers,
                      batchnorm=cls_args.batch_norm).to(args.device)


def train_cluster_aided_classifier(
        cls_model, train_loader, test_loader,
        train_label, all_train_label, args):
    cls_args = args.cacls
    optimizer = torch.optim.SGD(cls_model.parameters(),
                                momentum=cls_args.optim.momentum,
                                lr=cls_args.optim.lr)
    criterion = get_classification_loss(cls_args.loss.type)(
        train_label, cls_model.out_dim, cls_args.loss.label_smoothing).to(args.device)
    best_acc = 0.
    best_epoch = None
    best_state_dict = None
    best_message = None

    for epoch in range(1, cls_args.optim.epoch+1):
        cls_model.train()
        adjust_learning_rate(optimizer, epoch, cls_args.optim)
        total_loss = 0.0
        for X, Z, P, Y in train_loader:
            X = X.to(args.device)
            Y = Y.to(args.device)
            Z = Z.to(args.device)
            P = P.to(args.device)
            logit = cls_model(X, Z, P)
            loss = criterion(logit, Y)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(cls_model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            total_loss += loss.detach().cpu().item()

        if epoch % cls_args.eval_freq == 0:
            logging.info(f"cacls => Epoch: {epoch}, loss: {total_loss}.")
            pred, _, gt = eval_cluster_aided_classifier(
                cls_model, args, test_loader)
            assert pred.shape == gt.shape, str(pred.shape)+", "+str(gt.shape)
            many_shot, median_shot, low_shot = shot_acc(
                pred, gt, all_train_label)
            mean_acc = np.mean(pred == gt)
            msg = f"Test [{epoch}] {mean_acc:.3f} / {many_shot:.3f}, {median_shot:.3f}, {low_shot:.3f}"
            logging.info(msg)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(cls_model.state_dict())
                best_message = msg
    logging.getLogger("file").info(
        f"[ cacls ] Best epoch: {best_epoch}, {best_message}")
    cls_model.load_state_dict(best_state_dict)
    return cls_model


@torch.no_grad()
def eval_cluster_aided_classifier(cls_model, args, test_loader):
    assert isinstance(test_loader, tdata.DataLoader)
    cls_model.eval()
    pred = []
    logit = []
    gt = []
    for X, Z, P, Y in test_loader:
        X, P = X.to(args.device), P.to(args.device)
        Z = Z.to(args.device)
        t = cls_model(X, Z, P)
        logit.append(t)
        pred.append(torch.argmax(t, 1))
        gt.append(Y)
    pred = torch.cat(pred).cpu().numpy()
    logit = torch.cat(logit).cpu().numpy()
    gt = torch.cat(gt).cpu().numpy()
    return pred, logit, gt


def train_regular_classifier(cls_model, train_loader, test_loader, train_label, all_train_label, args, **kwargs):
    logging.warning(f"Ignoring args: {list(kwargs.keys())}")
    cls_args = args.regcls
    optimizer = torch.optim.SGD(cls_model.parameters(),
                                momentum=cls_args.optim.momentum,
                                lr=cls_args.optim.lr)
    criterion = get_classification_loss(cls_args.loss.type)(
        train_label, cls_model.out_dim, cls_args.loss.label_smoothing).to(args.device)
    best_acc = 0.0
    best_epoch = None
    best_state_dict = None
    best_message = None
    for epoch in range(1, cls_args.optim.epoch+1):
        cls_model.train()
        adjust_learning_rate(optimizer, epoch, cls_args.optim)
        total_loss = 0.0
        for X, Z, Y in train_loader:
            X, Z, Y = X.to(args.device), Z.to(args.device), Y.to(args.device)
            logit = cls_model(X, Z)
            loss = criterion(logit, Y)
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(cls_model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()
            total_loss += loss.detach().cpu().item()
        if epoch % cls_args.eval_freq == 0:
            logging.info(f"regcls => Epoch: {epoch}, loss: {total_loss}.")
            pred, _, gt = eval_regular_classifier(cls_model, args, test_loader)
            assert pred.shape == gt.shape, str(pred.shape)+", "+str(gt.shape)
            many_shot, median_shot, low_shot = shot_acc(
                pred, gt, all_train_label)
            mean_acc = np.mean(pred == gt)
            msg = f"Test [{epoch}] {mean_acc:.3f} / {many_shot:.3f}, {median_shot:.3f}, {low_shot:.3f}"
            logging.info(msg)
            if mean_acc > best_acc:
                best_acc = mean_acc
                best_epoch = epoch
                best_state_dict = copy.deepcopy(cls_model.state_dict())
                best_message = msg
    logging.getLogger("file").info(
        f"[ regcls ] Best epoch: {best_epoch}, {best_message}")
    cls_model.load_state_dict(best_state_dict)
    return cls_model


@torch.no_grad()
def eval_regular_classifier(cls_model, args, test_loader):
    assert isinstance(test_loader, tdata.DataLoader)
    cls_model.eval()
    pred = []
    logit = []
    gt = []
    for X, Z, Y in test_loader:
        X, Z = X.to(args.device), Z.to(args.device)
        t = cls_model(X, Z)
        logit.append(t)
        pred.append(torch.argmax(t, 1))
        gt.append(Y)
    pred = torch.cat(pred).cpu().numpy()
    logit = torch.cat(logit).cpu().numpy()
    gt = torch.cat(gt).cpu().numpy()
    return pred, logit, gt
