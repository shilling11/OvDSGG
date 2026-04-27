#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP_DIR=exps/clip_vitb16/stage3_union_spatial
mkdir -p ${EXP_DIR}

python -u main.py \
    --lr 2e-5 \
    --lr_backbone 0 \
    --resume exps/clip_vitb16/stage2_det_frozen/checkpoint0014.pth \
    --start_epoch 0 \
    --backbone clip_vitb16 \
    --epochs 40 \
    --eval_interval 6000 \
    --periodic_eval_batches 1000 \
    --num_feature_levels 1 \
    --num_queries 200 \
    --num_rel_queries 50 \
    --set_cost_class 0.1 \
    --cls_loss_coef 4 \
    --rel_cls_loss_coef 4 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --lr_drop 20 \
    --obj_split base \
    --pred_split base \
    --lr_drop_epochs 20 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vidvrd \
    --use_union_features \
    --use_spatial_motion_features \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
