#!/bin/sh

ml purge
ml load cuda/11

python -m apps.train_color --name "color_pifu" --random_flip --random_scale --random_trans --random_bg  --batch_size=2 --num_sample_inout 0 --num_sample_color 6000 --sigma 0.01
