#!/bin/sh

ml purge
ml load cuda/10.2

python -m apps.train_pifu_part --random_flip --random_scale --random_trans --random_bg --gpu_id=$CUDA_VISIBLE_DEVICES --batch_size=4
