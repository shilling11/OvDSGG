#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP_DIR=exps/r50_rel/stage1_frozen_v2
python -u main.py \
    --lr 5e-5 \
    --resume exps/exps_multi/r50/checkpoint0006.pth \
    --coco_pretrain \
    --backbone resnet50 \
    --epochs 15 \
    --eval_interval 1000 \
    --num_feature_levels 1 \
    --num_queries 200 \
    --num_rel_queries 50 \
    --set_cost_class 0.1 \
    --cls_loss_coef 4 \
    --rel_cls_loss_coef 4 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --lr_drop 10 \
    --obj_split base \
    --pred_split base \
    --lr_drop_epochs 10 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vidvrd \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
