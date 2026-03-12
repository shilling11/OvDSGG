#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

EXP_DIR=exps/multibaseline/r101_grad/start_7_e14_nf1_ld10,12_lr0.0002_nq300_wbox_MEGA_detrNorm_preSingle_nr14_dc5_nql3_filter150_75_40
mkdir -p ${EXP_DIR}
PY_ARGS=${@:1}
python -u main.py \
    --backbone resnet101 \
    --eval \
    --obj_split all \
    --pred_split all \
    --num_feature_levels 1 \
    --num_queries 300 \
    --dilation \
    --batch_size 1 \
    --num_ref_frames 14 \
    --resume ${EXP_DIR}/checkpoint0013.pth \
    --lr_drop_epochs 10 12 \
    --num_workers 16 \
    --with_box_refine \
    --dataset_file vidvrd \
    --output_dir ${EXP_DIR}/eval \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/eval/log.eval.$T
