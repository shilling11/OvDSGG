#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP_DIR=exps/clip_vitb16/stage1_new_rel_frozen
python -u main.py \
    --lr 5e-5 \
    --lr_prompts 5e-5 \
    --lr_backbone 0 \
    --coco_pretrain \
    --resume exps/exps_multi/r50/checkpoint0006.pth \
    --freeze_rel_head \
    --backbone clip_vitb16 \
    --epochs 60 \
    --eval_interval 0 \
    --clip_max_norm 1.0 \
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
    --use_visual_prompts \
    --n_visual_ctx 16 \
    --obj_split base \
    --pred_split base \
    --lr_drop_epochs 40 \
    --num_workers 4 \
    --with_box_refine \
    --dataset_file vidvrd \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
