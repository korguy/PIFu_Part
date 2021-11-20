#!/bin/sh

ml purge
ml load cuda/11

python -m apps.train_pifu_part --name "part_pifu" --random_flip --random_scale --random_trans --random_bg  --batch_size=4 --resume_epoch 0 --use_cache
