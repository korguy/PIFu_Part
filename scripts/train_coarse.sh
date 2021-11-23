#!/bin/sh

ml purge
ml load cuda/11

python -m apps.train_pifu_part --name "part_pifu" --random_flip --random_scale --random_trans --random_bg  --batch_size=3 --resume_epoch 22 --use_cache --step 36 --freq_mesh 100 --freq_eval 100 --pin_memory
