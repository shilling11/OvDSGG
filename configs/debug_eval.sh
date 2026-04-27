#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

PY_ARGS=${@:1}
python -u main.py \
    --eval \
    --resume exps/clip_vitb16/stage3_finetune/checkpoint.pth \
    --backbone clip_vitb16 \
    --num_feature_levels 1 \
    --num_queries 200 \
    --num_rel_queries 50 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --obj_split base \
    --pred_split base \
    --with_box_refine \
    --dataset_file vidvrd \
    --periodic_eval_batches 500 \
    --output_dir exps/clip_vitb16/debug_eval \
    2>&1 | tail -100
