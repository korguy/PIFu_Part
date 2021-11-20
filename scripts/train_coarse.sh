#!/bin/sh

ml purge
ml load cuda/11

CUDA_VISIBLE_DEVICES=0 python -m apps.train_pifu_part --name "part_pifu" --random_flip --random_scale --random_trans --random_bg  --batch_size=1 --resume_epoch 0
