#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/r50_rel/stage1_frozen_v2
mkdir -p ${EXP_DIR}/eval
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet50 \
    --num_obj_classes 35 \
    --num_pred_classes 132 \
    --eval \
    --obj_split all \
    --pred_split novel \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume ${EXP_DIR}/checkpoint0014.pth \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vidvrd \
    --output_dir ${EXP_DIR}/eval \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/eval/log.eval.$T
