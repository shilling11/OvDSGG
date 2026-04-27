#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

export PYTORCH_ALLOC_CONF=expandable_segments:True

EXP_DIR=exps/clip_vitb16/stage3_finetune
mkdir -p ${EXP_DIR}/eval
PY_ARGS=${@:1}
python -u main.py \
    --eval \
    --obj_split all \
    --pred_split all \
    --num_feature_levels 1 \
    --num_queries 200 \
    --num_rel_queries 50 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume ${EXP_DIR}/checkpoint.pth \
    --backbone clip_vitb16 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vidvrd \
    --output_dir ${EXP_DIR}/eval \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/eval/log.eval.$T

