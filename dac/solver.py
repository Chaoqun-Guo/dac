# -*- coding: utf-8 -*-
"""
@File: solver.py
@Time: 2023/08/25 16:57:44
@Author: Chaoqun Guo <chaoqunguo317@gmail.com>
@Version: 0.0.1
@Desc: use dlsa to solve lt.
"""
from collections import Counter, defaultdict
import copy
import os.path as osp
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import math
import logging
import torch
import torch.utils.data as tdata
import torch.nn as nn
import torch.nn.functional as F
from dac.data.data_utils import ContrastiveDataset, get_data_sampler, make_tensor_dataloader
import dac.logger
from dac.loss.losses import get_contrast_loss
from dac.metrics import adjust_learning_rate, mcc_mni_metrics, numpy2torch, shot_acc
from dac.models.classifier_utils import eval_cluster_aided_classifier, eval_regular_classifier, get_classifier, train_cluster_aided_classifier, train_regular_classifier
from dac.models.FlowGMM import FlowGMM
from dac.utils import state_dict_strip_prefix


@torch.no_grad()
def finetuned_results(model, features, labels, args, return_latent_logp=False, splits=["train", "test"]):
    model.eval()

    def tqdm_wrapper(x):
        return x

    representations = EasyDict()
    class_labels = EasyDict()
    cluster_labels = EasyDict()
    sample_llh = EasyDict()
    latent_llh = EasyDict()
    for split in splits:
        dataloder = tdata.DataLoader(
            tdata.TensorDataset(features[split], labels[split]),
            args.gmflow.data.batchsize, shuffle=False)
        x = []
        y = []
        c = []  # cluster
        px = []  # llh
        pz = []
        for (pos_x, pos_y) in tqdm_wrapper(dataloder):
            pos_x = pos_x.to(args.device)
            _, log_p_pos, log_llh, log_lh_u = model(pos_x, return_lh_u=True)
            # logging.info(f"Cluster Prob: {log_p_pos}")
            # logging.info(f"Cluster Prob argmax: {log_p_pos.argmax(1)}")
            # logging.info(f"Cluster Prob softmax: {log_p_pos.softmax(1)}")
            # logging.info(
            #     f"Cluster Prob max: {torch.max(log_p_pos.softmax(1),dim=1)[0]}")
            # logging.info(
            #     f"Cluster Prob max index: {torch.where(torch.max(log_p_pos.softmax(1),dim=1)[0]>0.8)}")
            # exit()
            cluster = log_p_pos.argmax(1)
            x.append(pos_x.cpu())
            y.append(pos_y.cpu().long())
            c.append(cluster.cpu().long())
            px.append(log_llh.cpu())
            if return_latent_logp:
                pz.append(log_lh_u.cpu())

        representations[split] = torch.cat(x).numpy()
        class_labels[split] = torch.cat(y).numpy()
        cluster_labels[split] = torch.cat(c).numpy()
        sample_llh[split] = torch.cat(px).numpy()
        latent_llh[split] = torch.cat(pz).numpy() if len(pz) > 0 else None

    if return_latent_logp:
        return representations, class_labels, cluster_labels, sample_llh, latent_llh
    else:
        return representations, class_labels, cluster_labels, sample_llh, None


def finetune_representations(model, features, labels, args, save_dir):
    """finetune representations.
    @Time: 2023/08/25 17:45:27
    @Params: 
        param: desc
    @Return: 

    """
    def tqdm_wrapper(x):
        return tqdm(enumerate(x), total=len(x))

    flow_args = args.gmflow
    loss_args = flow_args.loss
    dataset = ContrastiveDataset(
        features.train, labels.train, flow_args.data.weight_q)  # constrastive dataset
    dataloder = tdata.DataLoader(
        dataset, flow_args.data.batchsize, shuffle=False, num_workers=flow_args.data.num_workers)
    optimizer = torch.optim.SGD(
        model.parameters(), momentum=flow_args.optim.momentum, lr=flow_args.optim.lr)
    contrast_loss = get_contrast_loss(
        loss_args.contrast_type)(loss_args.contrast_margin)

    posterior_p_k_momentum = 0.0
    iteration_index = 1
    best_remain_cluster = -1
    best_state_dict = None
    best_epoch = None

    purity_decay = np.linspace(1.0, 0.8, flow_args.optim.epoch+1)
    for epoch_id in range(1, flow_args.optim.epoch+1):
        model.train()
        logging.info(f"Epoch {epoch_id}")
        adjust_learning_rate(optimizer, epoch_id, flow_args.optim)
        total_loss = defaultdict(float)

        for batch_i, (pos_x, neg_x, rev_wgt) in tqdm_wrapper(dataloder):
            pos_x = pos_x.to(args.device)
            neg_x = neg_x.to(args.device)
            rev_wgt = rev_wgt.to(args.device)
            _, log_p_pos, log_lh_pos = model(pos_x)
            _, log_p_neg, log_lh_neg = model(neg_x)

            # mask = log_p_pos.ge(args.threshold).float()
            # logging.info(f"args.threshold: {args.threshold}")
            # logging.info(f"Mask: {mask}")

            losses = {}
            losses["log_lh"] = - \
                (log_lh_pos*rev_wgt).mean() * loss_args.lambda_lh
            losses["total"] = losses["log_lh"]

            if loss_args.lambda_purity > 0:
                if loss_args.contrast_freeze:
                    loss_purity = contrast_loss(
                        log_p_neg, log_p_pos.detach())
                else:
                    loss_purity = contrast_loss(
                        log_p_neg, log_p_pos)

                losses["purity"] = loss_purity * loss_args.lambda_purity
                losses["total"] = losses["total"] + losses["purity"]
                # losses["total"] = losses["total"]

            if loss_args.lambda_balance > 0:
                posterior_p_k = torch.softmax(log_p_pos, -1).mean(0) + 1e-7
                if loss_args.posterior_momentum is not None:
                    posterior_p_k = posterior_p_k_momentum * loss_args.posterior_momentum + \
                        posterior_p_k * (1.0 - loss_args.posterior_momentum)
                    posterior_p_k_momentum = posterior_p_k.detach()
                    unbiased_adjustment = 1.0 / \
                        (1.0-math.pow(loss_args.posterior_momentum, iteration_index))
                    posterior_p_k = posterior_p_k*unbiased_adjustment
                    iteration_index += 1

                # entropy
                neg_ent = (posterior_p_k.log() * posterior_p_k).sum()
                losses["balance"] = (
                    neg_ent + math.log(flow_args.model.num_cluster)) * loss_args.lambda_balance

                losses["total"] = losses["total"] + losses["balance"]

            optimizer.zero_grad()
            if flow_args.optim.grad_clip:
                torch.nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), flow_args.optim.grad_clip)
            losses["total"].backward()
            optimizer.step()

            for k, v in losses.items():
                total_loss[k] += v.detach().cpu().item()

        logging.info("Loss: " +
                     ", ".join([f"{k} {v:.2f}" for k, v in total_loss.items()])
                     )

        model.eval()
        _, class_labels, cluster_labels, _, _ = finetuned_results(
            model, features, labels, args, splits=["train"])
        class_labels = class_labels["train"]
        cluster_labels = cluster_labels["train"]

        purity = []
        cluster_size = []
        num_empty_cluster = 0

        for i in range(flow_args.model.num_cluster):
            c_ids = class_labels[cluster_labels == i].tolist()
            if len(c_ids) == 0:
                purity.append(0)
                cluster_size.append(0)
                num_empty_cluster += 1
            else:
                cnt = list(Counter(c_ids).values())
                purity.append(np.max(cnt)*1.0 / len(c_ids))
                cluster_size.append(len(c_ids))
        purity = np.array(purity)
        cluster_size = np.array(cluster_size)

        logging.info(f"Purity: {purity}")
        logging.info(
            f"Purity: >0.9 {np.sum(purity>0.9)}, >0.7 {np.sum(purity>0.7)}, >0.5 {np.sum(purity>0.5)}")

        remain_mask = (purity > args.filter.min_purity) & (
            cluster_size > args.filter.min_size)
        remain_samples = np.sum(cluster_size[remain_mask])
        remain_clusters = np.sum(remain_mask)
        cluster_size = np.sort(cluster_size)
        qt = len(cluster_size)//4

        logging.info(f"Cluster remains: {remain_clusters}")
        logging.info(
            f"Quartiles: [{cluster_size[qt]} {cluster_size[qt*2]} {cluster_size[qt*3]}]")
        logging.info(
            f"Cluster_size: {cluster_size}; Total {np.sum(cluster_size)}, Min {cluster_size[0]}, Max {cluster_size[-1]}")
        logging.info(f"Empty cluster: {num_empty_cluster}")

        if remain_clusters > best_remain_cluster and cluster_size[-1] < 5000:
            best_remain_cluster = remain_clusters
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch_id

        logging.info(
            f"Best clusters remaining: {best_remain_cluster} at epoch {best_epoch}, {remain_samples} samples.")

        if flow_args.save_freq > 0 and epoch_id % flow_args.save_freq == 0:
            torch.save({
                "model": model.state_dict(),
            }, osp.join(save_dir, f"ckpt_{epoch_id:03d}.t7"))

    if best_state_dict is None:
        logging.warning("No best_state_dict")
    else:
        model.load_state_dict(best_state_dict)

    logging.getLogger("file").info(
        "Best clusters remaining: %d", best_remain_cluster)

    return model


def divide_samples(class_labels, cluster_labels, sample_llh, args):
    """divide samples to classifier.
    @Time: 2023/09/13 15:41:27
    @Params: 
        param: desc
    @Return: 

    """
    # determine label mapping
    classify_index = EasyDict(
        train=sample_llh.train < args.filter.llh_thres,
        test=sample_llh.test < args.filter.llh_thres
    )
    labelmap_index = EasyDict()
    # cluster_id -> {class_id->freq}
    cluster_prior = {}
    # train
    cluster_sizes = Counter(cluster_labels.train.tolist())

    for c_id, c_size in cluster_sizes.items():
        ids = np.where(cluster_labels.train == c_id)[0]
        class_sizes = Counter(class_labels.train[ids].tolist())
        pure_class, pure_size = max(
            list(class_sizes.items()), key=lambda x: x[1])
        purity = pure_size*1./len(ids)

        cluster_prior[c_id] = {k: v*1.0/c_size for k, v in class_sizes.items()}

    labelmap_index.train = ~classify_index.train

    logging.info(
        f"train: {np.sum(classify_index.train)} for classification, {np.sum(labelmap_index.train)} for labelmapping")
    cluster_sizes = Counter(cluster_labels.test.tolist())
    for c_id, c_size in cluster_sizes.items():
        ids = np.where(cluster_labels.test == c_id)[0]
    labelmap_index.test = ~classify_index.test
    logging.info(
        f"test: {np.sum(classify_index.test)} for classification, {np.sum(labelmap_index.test)} for labelmapping")

    return classify_index, labelmap_index, cluster_prior


def divide_and_conquer(all_features, all_inputs, all_labels, args, save_dir, global_mask=None, rec_depth=1, max_depth=5):
    """divide and conquer.
    @Time: 2023/08/25 17:06:05
    @Params: 
        param: desc
    @Return: 

    """
    if global_mask is None:
        def apply_global_mask(pair):
            return pair
    else:
        def apply_global_mask(pair):
            return EasyDict({"train": pair.train[global_mask.train, ...], "test":  pair.test[global_mask.test, ...], })

    all_features = numpy2torch(all_features)
    all_inputs = numpy2torch(all_inputs)
    all_labels = numpy2torch(all_labels, to_long=True)

    features = apply_global_mask(all_features)
    inputs = apply_global_mask(all_inputs)
    labels = apply_global_mask(all_labels)

    if global_mask is not None:
        logging.info(
            f"Masked data sizes {features.train.shape}, {features.test.shape}, {labels.train.shape}, {labels.test.shape}")

    args.threshold = torch.tensor(args.threshold).to(args.device)

    logging.info("Clustering finetune.")
    ft_model = FlowGMM(args).to(args.device)
    if args.ft_model_weight is not None and len(args.ft_model_weight) >= rec_depth:
        state_dict = torch.load(args.ft_model_weight[rec_depth-1])["model"]
        state_dict = state_dict_strip_prefix(state_dict)
        ft_model.load_state_dict(state_dict, strict=True)
    else:
        if torch.cuda.device_count() > 1:
            ft_model = nn.parallel.DataParallel(ft_model)
        ft_model = finetune_representations(
            ft_model, inputs, labels, args, save_dir)
        torch.save({"model": ft_model.state_dict(), }, osp.join(
            save_dir, f"ft_models_D{rec_depth}.t7"))

    if args.only_cluster:
        return

    logging.info("Divide.")
    representations, class_labels, cluster_labels, sample_llh, _ = finetuned_results(
        ft_model, all_inputs, all_labels, args)

    for x in [representations, class_labels, cluster_labels, sample_llh]:
        for k, v in x.items():
            print(k, v.shape)

    if args.filter.llh_proportion is not None:
        if args.filter.llh_thres is not None:
            logging.info(f"Filter configuring...")
            logging.warning(
                "Both 'llh_thres' and 'llh_proportion' are set: 'llh_proportion' is used")
        if global_mask is None:
            available_llh = sample_llh.train
        else:
            available_llh = sample_llh.train[global_mask.train]

        llh_proportion = max(
            (1-rec_depth/max_depth), args.filter.llh_proportion)
        thres_index = len(available_llh)*(1.-llh_proportion)

        # thres_index = len(available_llh)*(1.-args.filter.llh_proportion)
        quantile = sorted(available_llh)[int(thres_index)]
        # quantile = np.median(sorted(available_llh))
        args.filter.update("llh_thres", quantile)

        logging.info(f"available llh: {available_llh}")
        logging.info(f"thres_index: {thres_index}")
        logging.info(f"quantile: {quantile}")

    classify_index, labelmap_index, cluster_prior = divide_samples(
        class_labels, cluster_labels, sample_llh, args)

    for clus_id in range(args.gmflow.model.num_cluster):
        zeros = np.zeros([args.num_class])
        if clus_id in cluster_prior:
            for k, v in cluster_prior[clus_id].items():
                zeros[k] = v
        cluster_prior[clus_id] = zeros

    if global_mask is not None:
        for k in global_mask:
            classify_index[k] = classify_index[k] & global_mask[k]
            labelmap_index[k] = labelmap_index[k] & global_mask[k]

    logging.info("Label mapping.")
    sample_priors = EasyDict()
    for split in ["train", "test"]:
        sample_priors[split] = np.array([
            cluster_prior[x] for x in cluster_labels[split].tolist()
        ], dtype=np.float32)

    logging.info("Loading classifier.")
    cls_model_1 = get_classifier(args.cacls.model, args)
    train_sampler = get_data_sampler(
        args.cacls.data.sampler, all_labels["train"][labelmap_index["train"], ...])

    train_loader = make_tensor_dataloader(
        [all_features, all_inputs, sample_priors,
            all_labels], labelmap_index, "train",
        args.cacls.data.batchsize, num_workers=args.cacls.data.num_workers,
        shuffle=(train_sampler is None), sampler=train_sampler)

    test_loader = make_tensor_dataloader(
        [all_features, all_inputs, sample_priors,
            all_labels], labelmap_index, "test",
        args.cacls.data.batchsize, num_workers=args.cacls.data.num_workers,
        shuffle=False)

    cls_model_1 = train_cluster_aided_classifier(
        cls_model_1, train_loader, test_loader,
        train_label=class_labels.train[labelmap_index.train],
        all_train_label=class_labels.train, args=args)

    torch.save({
        "model": cls_model_1.state_dict(),
    }, osp.join(save_dir, f"cacls_models_D{rec_depth}.t7"))

    full_test_loader = make_tensor_dataloader(
        [all_features, all_inputs, sample_priors, all_labels], None, "test",
        args.cacls.data.batchsize, shuffle=False)
    test_map_pred, _, _ = eval_cluster_aided_classifier(
        cls_model_1, args, full_test_loader)

    del ft_model
    del cls_model_1
    torch.cuda.empty_cache()

    if rec_depth < max_depth:
        rec_depth += 1
        test_cls_pred = divide_and_conquer(all_features, all_inputs, all_labels, args,
                                           save_dir, global_mask=classify_index,
                                           rec_depth=rec_depth, max_depth=max_depth)
    else:
        cls_model_2 = get_classifier(args.regcls.model, args)
        mask = classify_index
        train_sampler = get_data_sampler(
            args.regcls.data.sampler, all_labels["train"][mask["train"], ...])

        train_loader = make_tensor_dataloader(
            [all_features, all_inputs, all_labels], mask, "train",
            args.regcls.data.batchsize, num_workers=args.regcls.data.num_workers,
            shuffle=(train_sampler is None), sampler=train_sampler)

        test_loader = make_tensor_dataloader(
            [all_features, all_inputs, all_labels], mask, "test",
            args.regcls.data.batchsize, num_workers=args.regcls.data.num_workers,
            shuffle=False)

        cls_model_2 = train_regular_classifier(
            cls_model_2, train_loader, test_loader,
            train_label=class_labels.train,
            all_train_label=class_labels.train,
            args=args)

        torch.save({
            "model": cls_model_2.state_dict(),
        }, osp.join(save_dir, f"regcls_models_D{rec_depth}.t7"))

        test_loader = make_tensor_dataloader(
            [all_features, all_inputs, all_labels], None, "test",
            args.regcls.data.batchsize, shuffle=False)

        test_cls_pred, _, _ = eval_regular_classifier(
            cls_model_2, args, test_loader)

    logging.getLogger("file").info("Classification results.")
    for split in ["train", "test"]:
        logging.info(
            f"{split.upper()}: {np.sum(classify_index[split])} for classification, {np.sum(labelmap_index[split])} for labelmapping")

    for split, pred in zip(["test"], [test_map_pred]):

        gt = class_labels[split]
        pred = pred[labelmap_index[split]]
        gt = gt[labelmap_index[split]]

        many_shot, median_shot, low_shot = shot_acc(
            pred, gt, class_labels.train)
        mcc_score, nmi_score = mcc_mni_metrics(pred, gt)
        mean_acc = np.mean(pred == gt)

        logging.getLogger("file").info(
            f"[ D{rec_depth} ] CA Classifier: {split}: {mean_acc:.3f} / {many_shot:.3f}, {median_shot:.3f}, {low_shot:.3f} ({mcc_score:.3f}, {nmi_score:.3f})")

    for split, pred in zip(["test"], [test_cls_pred]):

        gt = class_labels[split]
        pred = pred[classify_index[split]]
        gt = gt[classify_index[split]]

        many_shot, median_shot, low_shot = shot_acc(
            pred, gt, class_labels.train)
        mcc_score, nmi_score = mcc_mni_metrics(pred, gt)
        mean_acc = np.mean(pred == gt)

        logging.getLogger("file").info(
            f"[ D{rec_depth} ] Regular Classifier: {split}: {mean_acc:.3f} / {many_shot:.3f}, {median_shot:.3f}, {low_shot:.3f} ({mcc_score:.3f}, {nmi_score:.3f})")

    final_test_pred = -1 * np.ones_like(all_labels["test"].cpu())
    final_test_pred[labelmap_index.test] = test_map_pred[labelmap_index.test]
    final_test_pred[classify_index.test] = test_cls_pred[classify_index.test]

    if global_mask is None:
        gt = all_labels["test"].cpu().numpy()
        pred = final_test_pred
    else:
        gt = all_labels["test"].cpu().numpy()[global_mask.test]
        pred = final_test_pred[global_mask.test]

    assert gt.shape == pred.shape, str(gt.shape)+", "+str(pred.shape)
    # logging.info("pred: %s; gt: %s", pred, gt)

    many_shot, median_shot, low_shot = shot_acc(pred, gt, class_labels.test)
    mcc_score, nmi_score = mcc_mni_metrics(pred, gt)
    mean_acc = np.mean(pred == gt)

    logging.getLogger("file").info(
        f"Final Test: mean acc {mean_acc:.3f} / {many_shot:.3f}, {median_shot:.3f}, {low_shot:.3f} ( mcc score {mcc_score:.3f}, nmi score {nmi_score:.3f})")
    return final_test_pred


if __name__ == "__main__":
    logging.info("test")
