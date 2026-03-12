#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/r101_rel/stage1_frozen
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet101 \
    --start_epoch 7 \
    --epochs 14 \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --obj_split base \
    --pred_split base \
    --resume ./exps/exps_multi/r101/checkpoint0006.pth \
    --lr_drop_epochs 10 12 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vidvrd \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
