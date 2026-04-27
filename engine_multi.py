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
import itertools
import json
from collections import defaultdict
import math
import os
import sys
from pathlib import Path
from typing import Iterable

import torch
import util.misc as utils
from util.box_ops import box_iou as _box_iou
from datasets.coco_eval import CocoEvaluator
from datasets.panoptic_eval import PanopticEvaluator
from datasets.data_prefetcher_multi import data_prefetcher
from datasets.vidvrd_eval import VidVRDEvaluator

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, text_embeddings: torch.Tensor = None, obj_text_embeddings: torch.Tensor = None,
                    global_step: int = 0, periodic_eval_kwargs: dict = None, scaler=None,
                    text_prompt_encoder=None):
    model.train()
    criterion.train()
    if text_prompt_encoder is not None:
        text_prompt_encoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):

        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=scaler is not None):
            # If text prompts are active, compute dynamic predicate embeddings with grad.
            # Otherwise fall back to static pre-computed embeddings.
            if text_prompt_encoder is not None:
                live_text_embeddings = text_prompt_encoder()  # [N_pred+1, 512], carries grad
            else:
                live_text_embeddings = text_embeddings
            outputs = model(samples, live_text_embeddings, obj_text_embeddings)
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
        # Collect all trainable parameters for grad norm clipping (model + text prompts)
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            # text_prompt_encoder is not DDP-wrapped (no-arg forward), so manually all-reduce its grads
            if text_prompt_encoder is not None and utils.is_dist_avail_and_initialized():
                world_size = utils.get_world_size()
                for p in text_prompt_encoder.parameters():
                    if p.requires_grad and p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                        p.grad.div_(world_size)
            all_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if text_prompt_encoder is not None:
                all_params += [p for p in text_prompt_encoder.parameters() if p.requires_grad and p.grad is not None]
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(all_params, max_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if text_prompt_encoder is not None and utils.is_dist_avail_and_initialized():
                world_size = utils.get_world_size()
                for p in text_prompt_encoder.parameters():
                    if p.requires_grad and p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.SUM)
                        p.grad.div_(world_size)
            all_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            if text_prompt_encoder is not None:
                all_params += [p for p in text_prompt_encoder.parameters() if p.requires_grad and p.grad is not None]
            if max_norm > 0:
                grad_total_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm)
            else:
                grad_total_norm = utils.get_total_grad_norm(all_params, max_norm)
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
            if text_prompt_encoder is not None:
                text_prompt_encoder.train()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step

def _merge_cls_buffer(buffer, batch_res):
    """Merge per-batch SGCls/PredCls results into a cross-batch buffer,
    accumulating trajectories and scores for the same (vid_id, key)."""
    for vid_id, pair_dict in batch_res.items():
        for key, data in pair_dict.items():
            if key not in buffer[vid_id]:
                buffer[vid_id][key] = data
            else:
                existing = buffer[vid_id][key]
                existing['sub_trajectory'].update(data['sub_trajectory'])
                existing['obj_trajectory'].update(data['obj_trajectory'])
                existing['rel_prob_sum'] += data['rel_prob_sum']
                existing['frame_count'] += data['frame_count']


def _link_frame_preds_to_tracklets(video_frame_preds, iou_thresh=0.3):
    """Link per-frame per-video predictions into multi-frame tracklets via greedy IoU matching."""
    linked = {}
    for vid_id, frame_data in video_frame_preds.items():
        tracklets = []

        for frame_id in sorted(frame_data.keys()):
            fd = frame_data[frame_id]
            K = fd['scores'].shape[0]

            if not tracklets:
                for k in range(K):
                    tracklets.append({
                        'sub_traj':  {frame_id: fd['sub_boxes'][k].tolist()},
                        'obj_traj':  {frame_id: fd['obj_boxes'][k].tolist()},
                        'sub_label': fd['sub_labels'][k].item(),
                        'obj_label': fd['obj_labels'][k].item(),
                        'predicate': fd['predicates'][k].item(),
                        'scores':    [fd['scores'][k].item()],
                        'last_sub':  fd['sub_boxes'][k],
                        'last_obj':  fd['obj_boxes'][k],
                    })
                continue

            used = set()
            for k in range(K):
                sub_b    = fd['sub_boxes'][k]
                obj_b    = fd['obj_boxes'][k]
                pred_cl  = fd['predicates'][k].item()

                best_combined, best_t = iou_thresh - 1e-9, -1
                for t_idx, t in enumerate(tracklets):
                    if t_idx in used or t['predicate'] != pred_cl:
                        continue
                    s_iou = _box_iou(sub_b.unsqueeze(0), t['last_sub'].unsqueeze(0))[0][0, 0].item()
                    o_iou = _box_iou(obj_b.unsqueeze(0), t['last_obj'].unsqueeze(0))[0][0, 0].item()
                    avg = (s_iou + o_iou) / 2
                    if avg > best_combined:
                        best_combined, best_t = avg, t_idx

                if best_t >= 0:
                    t = tracklets[best_t]
                    t['sub_traj'][frame_id] = sub_b.tolist()
                    t['obj_traj'][frame_id] = obj_b.tolist()
                    t['scores'].append(fd['scores'][k].item())
                    t['last_sub'] = sub_b
                    t['last_obj'] = obj_b
                    used.add(best_t)
                else:
                    tracklets.append({
                        'sub_traj':  {frame_id: sub_b.tolist()},
                        'obj_traj':  {frame_id: obj_b.tolist()},
                        'sub_label': fd['sub_labels'][k].item(),
                        'obj_label': fd['obj_labels'][k].item(),
                        'predicate': pred_cl,
                        'scores':    [fd['scores'][k].item()],
                        'last_sub':  sub_b,
                        'last_obj':  obj_b,
                    })

        linked[vid_id] = {
            'sub_trajectories': [t['sub_traj'] for t in tracklets],
            'obj_trajectories': [t['obj_traj'] for t in tracklets],
            'scores':     torch.tensor([max(t['scores']) for t in tracklets]),
            'predicates': torch.tensor([t['predicate']  for t in tracklets]),
            'sub_labels': torch.tensor([t['sub_label']  for t in tracklets]),
            'obj_labels': torch.tensor([t['obj_label']  for t in tracklets]),
        }
    return linked


def _run_periodic_eval(model, criterion, global_step, postprocessors, data_loader_val,
                       base_ds, device, output_dir, text_embeddings, obj_text_embeddings, args,
                       max_eval_batches=0, text_prompt_encoder=None):
    """Lightweight in-training evaluation on SGDet and PredCls with pred_split='all'."""
    print(f"\n[Periodic Eval @ step {global_step}]")
    # Snapshot dynamic text embeddings for eval (no grad, no side effects on ctx)
    if text_prompt_encoder is not None:
        with torch.no_grad():
            eval_text_embeddings = text_prompt_encoder()
    else:
        eval_text_embeddings = text_embeddings
    # Temporarily override pred_split to 'all' for stable monitoring signal
    original_pred_split = args.pred_split
    args.pred_split = 'all'
    eval_stats, _ = evaluate(
        model, criterion, postprocessors, data_loader_val, base_ds,
        device, output_dir, eval_text_embeddings, obj_text_embeddings, args,
        eval_types=['sgdet', 'predcls'], save_preds=False,
        max_eval_batches=max_eval_batches, compute_loss=False,
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
            serializable = {k: float(v) if hasattr(v, 'item') else v for k, v in log_stats.items()}
            f.write(json.dumps(serializable) + "\n")


class _LimitedLoader:
    """Wraps a DataLoader to limit iteration count while preserving len() for log_every."""
    def __init__(self, loader, n):
        self.loader = loader
        self.n = min(n, len(loader))
    def __len__(self):
        return self.n
    def __iter__(self):
        return itertools.islice(iter(self.loader), self.n)


@torch.inference_mode()
def evaluate(model, criterion, postprocessors, data_loader, base_ds, device, output_dir, text_embeddings, obj_text_embeddings, args,
             eval_types=None, save_preds=True, max_eval_batches=0, compute_loss=True, text_prompt_encoder=None):
    model.eval()
    criterion.eval()

    if text_prompt_encoder is not None:
        text_prompt_encoder.eval()
        with torch.no_grad():
            text_embeddings = text_prompt_encoder()
        print(f"[Eval] Using live text embeddings from TextPromptEncoder (shape={tuple(text_embeddings.shape)})")

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
    vidvrd_evaluator = VidVRDEvaluator(
        base_ds, eval_types,
        pred_split=args.pred_split, obj_split=args.obj_split,
        num_base_pred=num_base_pred, num_base_obj=num_base_obj,
    )
    video_frame_preds = defaultdict(dict)  # vid_id → {frame_id → frame_data}
    sgcls_buffer = defaultdict(dict)       # vid_id → {key → data}
    predcls_buffer = defaultdict(dict)     # vid_id → {key → data}

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

    eval_loader = _LimitedLoader(data_loader, max_eval_batches) if max_eval_batches > 0 else data_loader
    for samples, targets in metric_logger.log_every(eval_loader, 10, header):
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

        if compute_loss:
            loss_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            loss_dict_scaled = {k: v * weight_dict[k]
                                for k, v in loss_dict.items() if k in weight_dict}
            loss_dict_unscaled = {f'{k}_unscaled': v for k, v in loss_dict.items()}
            metric_logger.update(loss=sum(loss_dict_scaled.values()),
                                 **loss_dict_scaled,
                                 **loss_dict_unscaled)
            metric_logger.update(class_error=loss_dict.get('class_error', torch.tensor(0.0)))

        if args.pred_split == 'novel':
            outputs['pred_rel_logits'][:, :, 1:num_base_pred + 1] = -1e9
        elif args.pred_split == 'base':
            outputs['pred_rel_logits'][:, :, num_base_pred + 1:] = -1e9

        results = postprocessors['vidvrd'](outputs, targets)

        if vidvrd_evaluator is not None:
            # Buffer per-frame preds; tracklet linking happens after the full loop
            for vid_id, frame_data in results.items():
                video_frame_preds[vid_id].update(frame_data)
            if 'vidvrd_sgcls' in postprocessors:
                sgcls_res = postprocessors['vidvrd_sgcls'](outputs, targets)
                _merge_cls_buffer(sgcls_buffer, sgcls_res)
            if 'vidvrd_predcls' in postprocessors:
                predcls_res = postprocessors['vidvrd_predcls'](outputs, targets)
                _merge_cls_buffer(predcls_buffer, predcls_res)

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

    # Link per-frame predictions into multi-frame tracklets and feed to evaluator
    print(f"\n[Eval] SGDet: {len(video_frame_preds)} vids, "
          f"SGCls: {sum(len(v) for v in sgcls_buffer.values())} pairs, "
          f"PredCls: {sum(len(v) for v in predcls_buffer.values())} pairs")

    if vidvrd_evaluator is not None and video_frame_preds:
        linked = _link_frame_preds_to_tracklets(video_frame_preds)
        vidvrd_evaluator.update(linked)
    if vidvrd_evaluator is not None and sgcls_buffer:
        vidvrd_evaluator.update_cls(dict(sgcls_buffer), 'sgcls')
    if vidvrd_evaluator is not None and predcls_buffer:
        vidvrd_evaluator.update_cls(dict(predcls_buffer), 'predcls')

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
