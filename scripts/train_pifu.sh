#!/bin/sh

ml purge
ml load cuda/11

CUDA_VISIBLE_DEVICES=0 python -m apps.train_pifu --name "pifu" --random_flip --random_scale --random_trans --random_bg  --batch_size=4

