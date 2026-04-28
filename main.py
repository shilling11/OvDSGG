# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import os
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch.utils.data import DataLoader
import datasets

import datasets.samplers as samplers
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def get_args_parser():
    parser = argparse.ArgumentParser('Deformable DETR Detector', add_help=False)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=15, type=int)
    parser.add_argument('--lr_drop', default=5, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    
    parser.add_argument('--num_ref_frames', default=3, type=int, help='number of reference frames')

    parser.add_argument('--sgd', action='store_true')

    # Variants of Deformable DETR
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=4, type=int, help='number of feature levels')


    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=1024, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=300, type=int,
                        help="Number of query slots")
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--n_temporal_decoder_layers', default=1, type=int)
    parser.add_argument('--interval1', default=20, type=int)
    parser.add_argument('--interval2', default=60, type=int)

    parser.add_argument("--fixed_pretrained_model", default=False, action='store_true')
    parser.add_argument("--freeze_rel_head", default=False, action='store_true',
                        help="Stage 1: freeze relation head, train object detector only.")
    parser.add_argument("--freeze_det_head", default=False, action='store_true',
                        help="Stage 2: freeze object detection head (use with --fixed_pretrained_model for full stage 2).")

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=2, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--rel_cls_loss_coef', default=2, type=float)
    parser.add_argument('--rel_bbox_loss_coef', default=5, type=float)
    parser.add_argument('--rel_giou_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    # dataset parameters
    parser.add_argument('--num_obj_classes', default=25, type=int,
                        help='number of object classes the model outputs. For training: set to base class count (25). '
                             'For eval on novel/all: set to total class count so the model head is built large enough '
                             'and the checkpoint biases are padded accordingly.')
    parser.add_argument('--num_pred_classes', default=71, type=int,
                        help='number of predicate classes the model outputs. For training: set to base class count (71). '
                             'For eval on novel/all: set to total class count.')
    parser.add_argument('--num_rel_queries', default=100, type=int)

    # New feature flags
    parser.add_argument('--use_union_features', action='store_true',
                        help='Enable union region features via ROIAlign')
    parser.add_argument('--use_spatial_motion_features', action='store_true',
                        help='Enable spatial + velocity relation encoding')
    parser.add_argument('--roi_size', default=7, type=int,
                        help='ROIAlign output size for union features')

    # Multi-modal prompting
    parser.add_argument('--use_visual_prompts', action='store_true',
                        help='Inject learnable visual prompt tokens into CLIP ViT (VPT-shallow)')
    parser.add_argument('--n_visual_ctx', default=16, type=int,
                        help='Number of visual prompt tokens')
    parser.add_argument('--use_text_prompts', action='store_true',
                        help='Learn CoOp-style context vectors for predicate text embeddings')
    parser.add_argument('--n_text_ctx', default=16, type=int,
                        help='Number of text prompt context tokens for predicate embeddings')
    parser.add_argument('--text_ctx_init', default='', type=str,
                        help='Optional string to initialise text context vectors from')
    parser.add_argument('--lr_prompts', default=2e-3, type=float,
                        help='Learning rate for prompt parameters (visual and text)')

    parser.add_argument('--dataset_file', default='vidvrd')
    parser.add_argument('--coco_path', default='./data/vidvrd', type=str)
    parser.add_argument('--vid_path', default='./data/vidvrd', type=str)
    parser.add_argument('--coco_pretrain', default=False, action='store_true')
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--obj_split', default='base', type=str, choices=['all', 'base', 'novel'],
                        help='eval only: which object split to evaluate on (base/novel/all). Training always uses base.')
    parser.add_argument('--pred_split', default='all', type=str, choices=['all', 'base', 'novel'],
                        help='eval only: which predicate split to evaluate on (base/novel/all). Training always uses base.')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--cache_mode', default=False, action='store_true', help='whether to cache images on memory')
    parser.add_argument('--eval_interval', default=0, type=int,
                        help='Evaluate every N training iterations on SGDet+PredCls (mAP, R@50). 0 disables periodic eval.')
    parser.add_argument('--periodic_eval_batches', default=0, type=int,
                        help='Max val batches per periodic eval. 0 = full val set (slow). E.g. 1000 reduces ~4.5h to ~15min.')

    return parser


def main(args):
    print(args.dataset_file, 11111111)
    if args.dataset_file == "vid_single":
        from engine_single import evaluate, train_one_epoch
        import util.misc as utils
        
    else:
        from engine_multi import evaluate, train_one_epoch
        import util.misc_multi as utils

    print(args.dataset_file)
    device = torch.device(args.device)
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)


    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)

    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    dataset_train = build_dataset(image_set='train_vid', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        if args.cache_mode:
            sampler_train = samplers.NodeDistributedSampler(dataset_train)
            sampler_val = samplers.NodeDistributedSampler(dataset_val, shuffle=False)
        else:
            sampler_train = samplers.DistributedSampler(dataset_train)
            sampler_val = samplers.DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                   pin_memory=True, persistent_workers=args.num_workers > 0)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers,
                                 pin_memory=True, persistent_workers=args.num_workers > 0)

    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out

    for n, p in model_without_ddp.named_parameters():
        print(n)

    # Visual prompt token names sit under backbone.0.* and would match lr_backbone_names.
    # We exclude them from the backbone group so lr_backbone=0 doesn't silence them,
    # and put them in a dedicated prompt group at lr_prompts instead.
    _PROMPT_PARAM_NAMES = ('visual_prompt_tokens', 'visual_prompt_pos')

    def is_prompt(n):
        return any(k in n for k in _PROMPT_PARAM_NAMES)

    param_dicts = [
        {
            "params":
                [p for n, p in model_without_ddp.named_parameters()
                 if not match_name_keywords(n, args.lr_backbone_names)
                 and not match_name_keywords(n, args.lr_linear_proj_names)
                 and not is_prompt(n) and p.requires_grad],
            "lr": args.lr,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, args.lr_backbone_names)
                       and not is_prompt(n) and p.requires_grad],
            "lr": args.lr_backbone,
        },
        {
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
            "lr": args.lr * args.lr_linear_proj_mult,
        },
        {
            # Visual prompt tokens — always trained at lr_prompts regardless of lr_backbone
            "params": [p for n, p in model_without_ddp.named_parameters()
                       if is_prompt(n) and p.requires_grad],
            "lr": args.lr_prompts,
        },
    ]
    if args.sgd:
        optimizer = torch.optim.SGD(param_dicts, lr=args.lr, momentum=0.9,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                      weight_decay=args.weight_decay)
    print(args.lr_drop_epochs)
    if args.start_epoch > 0:
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', group['lr'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, args.lr_drop_epochs,
        last_epoch=args.start_epoch - 1 if args.start_epoch > 0 else -1
    )

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    # text_prompt_encoder must be DDP-wrapped separately (built after model DDP wrap).
    # We track the unwrapped module for checkpoint save/load.
    # (DDP wrap happens after param_dicts so ctx is already in the optimizer.)
    text_prompt_encoder_module = None  # placeholder — set after encoder is built below

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = dataset_val.vidvrd_gt

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)

        if args.eval:
            tmp_dict = checkpoint['model'].copy()
            for k in list(tmp_dict.keys()):
                if k in model_without_ddp.state_dict() and tmp_dict[k].shape != model_without_ddp.state_dict()[k].shape:
                    ckpt_tensor = tmp_dict[k]
                    model_tensor = model_without_ddp.state_dict()[k]
                    if len(ckpt_tensor.shape) == 1:
                        new_tensor = model_tensor.clone()
                        min_d0 = min(ckpt_tensor.shape[0], model_tensor.shape[0])
                        new_tensor[:min_d0] = ckpt_tensor[:min_d0]
                        tmp_dict[k] = new_tensor
                    elif len(ckpt_tensor.shape) == 2:
                        new_tensor = model_tensor.clone()
                        min_d0 = min(ckpt_tensor.shape[0], model_tensor.shape[0])
                        min_d1 = min(ckpt_tensor.shape[1], model_tensor.shape[1])
                        new_tensor[:min_d0, :min_d1] = ckpt_tensor[:min_d0, :min_d1]
                        tmp_dict[k] = new_tensor
                    else:
                        del tmp_dict[k]
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(tmp_dict, strict=False)
        else:
            tmp_dict = model_without_ddp.state_dict().copy()
            if args.coco_pretrain: # singleBaseline
                for k, v in checkpoint['model'].items():
                    if ('class_embed' not in k) :
                        tmp_dict[k] = v 
                    else:
                        print('k', k)
            else:
                tmp_dict = checkpoint['model']
                # Stage 2: freeze backbone+transformer+input_proj only when explicitly requested.
                # Without this guard, Stage 3 end-to-end fine-tuning would accidentally freeze
                # the transformer every time a non-coco checkpoint is resumed.
                if args.fixed_pretrained_model:
                    for name, param in model_without_ddp.named_parameters():
                        if 'backbone' in name or 'transformer' in name or 'input_proj' in name:
                            param.requires_grad = False
                        else:
                            param.requires_grad = True
            
            for k in list(tmp_dict.keys()):
                if k in model_without_ddp.state_dict():
                    if tmp_dict[k].shape != model_without_ddp.state_dict()[k].shape:
                        del tmp_dict[k]
            missing_keys, unexpected_keys = model_without_ddp.load_state_dict(tmp_dict, strict=False)

        unexpected_keys = [k for k in unexpected_keys if not (k.endswith('total_params') or k.endswith('total_ops'))]
        if len(missing_keys) > 0:
            print('Missing Keys: {}'.format(missing_keys))
        if len(unexpected_keys) > 0:
            print('Unexpected Keys: {}'.format(unexpected_keys))
        # text_prompt_encoder is built later; load its state after the encoder is ready.
        _text_prompt_ckpt = checkpoint.get('text_prompt_encoder', None)

        # Only restore optimizer/scheduler on a mid-stage resume (start_epoch > 0).
        # On a stage transition (start_epoch == 0), the scheduler's last_epoch counter
        # from the previous stage would misalign this stage's lr_drop_epochs milestones,
        # and the optimizer's Adam buffers from frozen-param training are stale context.
        if not args.eval and args.start_epoch > 0:
            if 'optimizer' in checkpoint:
                ckpt_sizes = [len(g['params']) for g in checkpoint['optimizer']['param_groups']]
                live_sizes = [len(g['params']) for g in optimizer.param_groups]
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    print(f"[resume] optimizer state loaded ({len(checkpoint['optimizer']['state'])} entries, "
                          f"groups={live_sizes})")
                except ValueError as e:
                    print(f"[resume] WARNING: optimizer state skipped -- param_group mismatch "
                          f"(ckpt={ckpt_sizes}, live={live_sizes}): {e}")
            if 'lr_scheduler' in checkpoint:
                try:
                    lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
                    print(f"[resume] scheduler state loaded (last_epoch={lr_scheduler.last_epoch})")
                except Exception as e:
                    print(f"[resume] WARNING: lr_scheduler state skipped: {e}")
        elif not args.eval and args.start_epoch == 0 and ('optimizer' in checkpoint or 'lr_scheduler' in checkpoint):
            print("[resume] start_epoch=0 -> skipping optimizer/scheduler state (stage transition, fresh schedule)")

    # Stage-based freezing (applied after checkpoint loading so requires_grad is set correctly)
    _REL_HEAD_NAMES = ('rel_proj_head', 'rel_query_embed', 'sub_pointer', 'obj_pointer',
                       'rel_obj_attn', 'triplet_fusion', 'rel_logit_scale', 'rel_class_bias')
    _DET_HEAD_NAMES = ('class_embed', 'bbox_embed', 'query_embed', 'obj_logit_scale', 'obj_class_bias',
                       'temp_class_embed', 'temp_bbox_embed')
    if args.freeze_rel_head:
        for n, p in model_without_ddp.named_parameters():
            if any(k in n for k in _REL_HEAD_NAMES):
                p.requires_grad_(False)
        # Frozen params still pass gradients through to the shared transformer decoder,
        # so zero relation loss weights to prevent random rel head from corrupting decoder.
        for k in list(criterion.weight_dict.keys()):
            if 'rel' in k:
                criterion.weight_dict[k] = 0.0
        print("Stage 1: relation head frozen, relation loss weights zeroed.")
    if args.freeze_det_head:
        for n, p in model_without_ddp.named_parameters():
            if any(k in n for k in _DET_HEAD_NAMES):
                p.requires_grad_(False)
        for k in list(criterion.weight_dict.keys()):
            if 'rel' not in k:
                criterion.weight_dict[k] = 0.0
        print("Stage 2: detection head frozen, detection loss weights zeroed.")

    print("Loading cached CLIP embeddings...")
    text_embeddings = torch.load("predicate_embeddings.pt").to(device)[:args.num_pred_classes + 1]
    text_embeddings.requires_grad_(False)
    obj_text_embeddings = torch.load("object_embeddings.pt").to(device)[:args.num_obj_classes + 1]
    obj_text_embeddings.requires_grad_(False)

    # Text prompt encoder (CoOp-style): replaces static predicate embeddings with
    # dynamic embeddings that carry gradients through learned context vectors.
    text_prompt_encoder = None
    if args.use_text_prompts:
        import clip as clip_lib
        from util.text_prompt_encoder import TextPromptEncoder
        print("Building TextPromptEncoder...")
        clip_model_tmp, _ = clip_lib.load('ViT-B/16', device='cpu')
        with open('datasets/vidvrd_dataset/VidVRD_pred_class_split_info_v2.json') as f:
            id2cls = json.load(f)['id2cls']
        predicate_names = [id2cls[str(i)] for i in range(args.num_pred_classes + 1)]
        text_prompt_encoder = TextPromptEncoder(
            clip_model_tmp, predicate_names, args.n_text_ctx, args.text_ctx_init
        ).to(device)
        del clip_model_tmp
        print(f"TextPromptEncoder built: n_ctx={args.n_text_ctx}, "
              f"n_classes={len(predicate_names)}")

    # Stage-based prompt freezing:
    # Stage 1 (freeze_rel_head=True): text prompts frozen — visual prompts train
    # Stage 2 (freeze_det_head=True): visual prompts frozen — text prompts train
    if args.freeze_rel_head and text_prompt_encoder is not None:
        for p in text_prompt_encoder.parameters():
            p.requires_grad_(False)
        print("Stage 1: text prompt encoder frozen.")
    if args.freeze_det_head and getattr(args, 'use_visual_prompts', False):
        for n, p in model_without_ddp.named_parameters():
            if is_prompt(n):
                p.requires_grad_(False)
        print("Stage 2: visual prompt tokens frozen.")

    # NOTE: text_prompt_encoder.forward() takes no inputs (returns embeddings from
    # learned ctx params), which DDP's input scatter cannot handle. Skip DDP wrapping
    # and rely on manual gradient all-reduce in engine_multi.py to keep ranks in sync.
    text_prompt_encoder_module = text_prompt_encoder

    # Deferred checkpoint load for text_prompt_encoder (built after model checkpoint load)
    if text_prompt_encoder_module is not None and '_text_prompt_ckpt' in dir():
        if _text_prompt_ckpt is not None:
            # Strip tokenized_classes: it's a fixed buffer that changes size when
            # num_pred_classes differs between training (72-way) and OV eval (133-way).
            # strict=False alone doesn't help — PyTorch raises on size mismatches regardless.
            # Only ctx (shape [n_ctx, 512]) needs to survive the checkpoint.
            filtered = {k: v for k, v in _text_prompt_ckpt.items()
                        if 'tokenized_classes' not in k}
            missing, unexpected = text_prompt_encoder_module.load_state_dict(
                filtered, strict=False)
            if missing:
                print(f"[text_prompt_encoder] WARNING: missing keys: {missing}")
            ctx = text_prompt_encoder_module.ctx
            print(f"Loaded text_prompt_encoder ctx from checkpoint "
                  f"(shape={tuple(ctx.shape)}, norm={ctx.norm():.4f})")

    # Add text prompt encoder ctx parameters to the optimizer
    if text_prompt_encoder_module is not None:
        ctx_params = list(text_prompt_encoder_module.parameters())
        if any(p.requires_grad for p in ctx_params):
            optimizer.add_param_group({"params": [p for p in ctx_params if p.requires_grad],
                                       "lr": args.lr_prompts})

    if args.eval:
        if hasattr(criterion, 'empty_weight'):
            num_obj_out = obj_text_embeddings.shape[0]
            if criterion.empty_weight.shape[0] != num_obj_out:
                criterion.empty_weight = torch.ones(num_obj_out).to(device)
                criterion.num_classes = num_obj_out - 1

        if hasattr(criterion, 'rel_empty_weight'):
            num_rel_out = args.num_pred_classes + 1 
            if criterion.rel_empty_weight.shape[0] != num_rel_out:
                criterion.rel_empty_weight = torch.ones(num_rel_out).to(device)
        test_stats, vidvrd_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir, text_embeddings, obj_text_embeddings, args,
                                              max_eval_batches=args.periodic_eval_batches, compute_loss=False, text_prompt_encoder=text_prompt_encoder)
        if args.output_dir:
            utils.save_on_master(vidvrd_evaluator.results, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    scaler = torch.amp.GradScaler('cuda')
    # scaler = None
    periodic_eval_kwargs = None
    if args.eval_interval > 0:
        periodic_eval_kwargs = dict(
            eval_interval=args.eval_interval,
            postprocessors=postprocessors,
            data_loader_val=data_loader_val,
            base_ds=base_ds,
            device=device,
            output_dir=args.output_dir,
            text_embeddings=text_embeddings,
            obj_text_embeddings=obj_text_embeddings,
            args=args,
            max_eval_batches=args.periodic_eval_batches,
            text_prompt_encoder=text_prompt_encoder,
        )
    global_step = args.start_epoch * len(data_loader_train)
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats, global_step = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm, text_embeddings, obj_text_embeddings,
            global_step=global_step, periodic_eval_kwargs=periodic_eval_kwargs, scaler=scaler,
            text_prompt_encoder=text_prompt_encoder)
        lr_scheduler.step()
        print('args.output_dir', args.output_dir)
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 5 epochs
            # if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 1 == 0:
            if (epoch + 1) % 1 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'text_prompt_encoder': text_prompt_encoder_module.state_dict() if text_prompt_encoder_module is not None else None,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        #test_stats, coco_evaluator = evaluate(
         #   model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        #)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
