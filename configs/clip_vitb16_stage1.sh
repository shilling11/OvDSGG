#!/usr/bin/env bash

set -x
T=`date +%m%d%H%M`

export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/torch/lib:$LD_LIBRARY_PATH
export PYTHONPATH=/workspace/OvDSGG/models/ops:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=600

# set --set_cost_class to 2 later for better convergence
EXP_DIR=exps/clip_vitb16/stage1_new_rel_frozen
python -u main.py \
    --lr 5e-5 \
    --lr_backbone 0 \
    --coco_pretrain \
    --start_epoch 0 \
    --freeze_rel_head \
    --backbone clip_vitb16 \
    --epochs 66 \
    --clip_max_norm 1.0 \
    --eval_interval 0 \
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
    --lr_drop_epochs 43 \
    --num_workers 4 \
    --with_box_refine \
    --dataset_file vidvrd \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS} 2>&1 | tee ${EXP_DIR}/log.train.$T
