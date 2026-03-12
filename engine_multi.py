# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
Train and eval functions used in main.py
"""
import json
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import util.misc as utils
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher_multi import data_prefetcher
from datasets.vidvrd_eval import VidVRDEvaluator

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, text_embeddings: torch.Tensor = None, obj_text_embeddings: torch.Tensor = None,
                    global_step: int = 0, periodic_eval_kwargs: dict = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, text_embeddings, obj_text_embeddings)
        loss_dict = criterion(outputs, targets)

        

        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
 
        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced_scaled, **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

        global_step += 1
        if periodic_eval_kwargs is not None and global_step % periodic_eval_kwargs['eval_interval'] == 0:
            _run_periodic_eval(model, criterion, global_step, **{k: v for k, v in periodic_eval_kwargs.items() if k != 'eval_interval'})
            model.train()
            criterion.train()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step

def _run_periodic_eval(model, criterion, global_step, postprocessors, data_loader_val,
                       base_ds, device, output_dir, text_embeddings, obj_text_embeddings, args):
    """Lightweight in-training evaluation on SGDet and PredCls with pred_split='all'."""
    print(f"\n[Periodic Eval @ step {global_step}]")
    # Temporarily override pred_split to 'all' for stable monitoring signal
    original_pred_split = args.pred_split
    args.pred_split = 'all'
    eval_stats, _ = evaluate(
        model, criterion, postprocessors, data_loader_val, base_ds,
        device, output_dir, text_embeddings, obj_text_embeddings, args,
        eval_types=['sgdet', 'predcls'], save_preds=False,
    )
    args.pred_split = original_pred_split

    # Extract and log the key metrics: mAP and R@50 for sgdet and predcls
    keys_of_interest = [k for k in eval_stats if
                        any(t in k for t in ('sgdet', 'predcls')) and
                        any(m in k for m in ('mAP', 'R@50'))]
    log_stats = {'periodic_step': global_step, **{k: eval_stats[k] for k in keys_of_interest}}
    print(f"[Periodic Eval @ step {global_step}] " + "  ".join(f"{k}: {v:.4f}" for k, v in log_stats.items() if k != 'periodic_step'))

    if output_dir and utils.is_main_process():
        with (Path(output_dir) / "periodic_eval_log.txt").open("a") as f:
            f.write(json.dumps(log_stats) + "\n")


@torch.no_grad()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, text_embeddings, obj_text_embeddings, args,
             eval_types=None, save_preds=True):
    model.eval()
    criterion.eval()

    obj_cls, pred_cls = set(), set()
    ds = data_loader.dataset.datasets[0] if hasattr(data_loader.dataset, 'datasets') else data_loader.dataset
    num_base_obj = ds.prepare.num_base_classes
    num_base_pred = ds.prepare.num_base_pred_classes

    for ann in ds.coco.dataset.get('annotations', []):
        obj_cls.add(ann['category_id'])

    for vid in ds.coco.dataset.get('videos', []):
        for rel in vid.get('relation_instances', []):
            if 'predicate_label' in rel:
                pred_cls.add(rel['predicate_label'])

    if args.obj_split == 'base':
        obj_cls = {c for c in obj_cls if c <= num_base_obj}
    elif args.obj_split == 'novel':
        obj_cls = {c for c in obj_cls if c > num_base_obj}

    if args.pred_split == 'base':
        pred_cls = {c for c in pred_cls if c <= num_base_pred}
    elif args.pred_split == 'novel':
        pred_cls = {c for c in pred_cls if c > num_base_pred}

    print(f"\n--- Pre-evaluation Dataset Scan ---")
    print(f"Obj Classes ({len(obj_cls)}): {sorted(list(obj_cls))}")
    print(f"Pred Classes ({len(pred_cls)}): {sorted(list(pred_cls))}")

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Test:'

    if eval_types is None:
        eval_types = ['relation_detection', 'relation_tagging', 'sgdet', 'sgcls', 'predcls']
    vidvrd_evaluator = VidVRDEvaluator(base_ds, eval_types)

    # iou_types = tuple(k for k in ('segm', 'bbox') if k in postprocessors.keys())
    # coco_evaluator = CocoEvaluator(base_ds, iou_types)
    # # coco_evaluator.coco_eval[iou_types[0]].params.iouThrs = [0, 0.1, 0.5, 0.75]

    # panoptic_evaluator = None
    # if 'panoptic' in postprocessors.keys():
    #     panoptic_evaluator = PanopticEvaluator(
    #         data_loader.dataset.ann_file,
    #         data_loader.dataset.ann_folder,
    #         output_dir=os.path.join(output_dir, "panoptic_eval"),
    #     )

    for samples, targets  in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, text_embeddings, obj_text_embeddings)

        for t in targets:
            if 'labels' in t and t['labels'].numel() > 0:
                t['labels'] = t['labels'].clamp(min=0, max=outputs['pred_logits'].shape[-1] - 1)
            if 'rel_labels' in t and t['rel_labels'].numel() > 0:
                t['rel_labels'] = t['rel_labels'].clamp(min=0, max=outputs['pred_rel_logits'].shape[-1] - 1)

            if 'sub_boxes' in t and t['sub_boxes'].numel() > 0:
                t['sub_boxes'] = t['sub_boxes'].clamp(min=0.0, max=1.0)
            if 'obj_boxes' in t and t['obj_boxes'].numel() > 0:
                t['obj_boxes'] = t['obj_boxes'].clamp(min=0.0, max=1.0)
            if 'boxes' in t and t['boxes'].numel() > 0:
                t['boxes'] = t['boxes'].clamp(min=0.0, max=1.0)

        loss_dict = criterion(outputs, targets)
        if args.pred_split == 'novel':
            outputs['pred_rel_logits'][:, :, 1:num_base_pred + 1] = -1e9
        elif args.pred_split == 'base':
            outputs['pred_rel_logits'][:, :, num_base_pred + 1:] = -1e9

        results = postprocessors['vidvrd'](outputs, targets)

        weight_dict = criterion.weight_dict

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict_reduced.items() if k in weight_dict}
        loss_dict_reduced_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict_reduced.items()}
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        if vidvrd_evaluator is not None:
            vidvrd_evaluator.update(results)
            if 'vidvrd_sgcls' in postprocessors:
                sgcls_res = postprocessors['vidvrd_sgcls'](outputs, targets)
                vidvrd_evaluator.update_cls(sgcls_res, 'sgcls')
            if 'vidvrd_predcls' in postprocessors:
                predcls_res = postprocessors['vidvrd_predcls'](outputs, targets)
                vidvrd_evaluator.update_cls(predcls_res, 'predcls')

        # orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        # results = postprocessors['bbox'](outputs, orig_target_sizes)
        # if 'segm' in postprocessors.keys():
        #     target_sizes = torch.stack([t["size"] for t in targets], dim=0)
        #     results = postprocessors['segm'](results, outputs, orig_target_sizes, target_sizes)
        # res = {target['image_id'].item(): output for target, output in zip(targets, results)}
        # if coco_evaluator is not None:
        #     coco_evaluator.update(res)

        # if panoptic_evaluator is not None:
        #     res_pano = postprocessors["panoptic"](outputs, target_sizes, orig_target_sizes)
        #     for i, target in enumerate(targets):
        #         image_id = target["image_id"].item()
        #         file_name = f"{image_id:012d}.png"
        #         res_pano[i]["image_id"] = image_id
        #         res_pano[i]["file_name"] = file_name

        #     panoptic_evaluator.update(res_pano)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    if vidvrd_evaluator is not None:
        vidvrd_evaluator.synchronize_between_processes()

    if vidvrd_evaluator is not None:
        if save_preds:
            torch.save(vidvrd_evaluator.predictions, args.output_dir + "/backup_eval_preds.pth")
        vidvrd_evaluator.accumulate()
        vidvrd_evaluator.summarize()

    # if coco_evaluator is not None:
    #     coco_evaluator.synchronize_between_processes()
    # if panoptic_evaluator is not None:
    #     panoptic_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    # if coco_evaluator is not None:
    #     coco_evaluator.accumulate()
    #     coco_evaluator.summarize()
    # panoptic_res = None
    # if panoptic_evaluator is not None:
    #     panoptic_res = panoptic_evaluator.summarize()
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    # if coco_evaluator is not None:
    #     if 'bbox' in postprocessors.keys():
    #         stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
    #     if 'segm' in postprocessors.keys():
    #         stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()
    # if panoptic_res is not None:
    #     stats['PQ_all'] = panoptic_res["All"]
    #     stats['PQ_th'] = panoptic_res["Things"]
    #     stats['PQ_st'] = panoptic_res["Stuff"]

    if vidvrd_evaluator is not None:
        for eval_type in eval_types:
            if eval_type in vidvrd_evaluator.results:
                for metric_name, value in vidvrd_evaluator.results[eval_type].items():
                    stats[f'vidvrd_{eval_type}_{metric_name}'] = value

    return stats, vidvrd_evaluator
