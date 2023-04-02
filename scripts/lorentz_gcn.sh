#!/bin/bash
python main.py \
    --model lorentz_gcn \
    --layers 4 \
    --dim 64 \
    --scale 0.1 \
    --margin 1.0 \
    --k_list [50] \
    --random_seed 2020 \
    --cuda True \
    --device 4 \
    --epochs 500 \
    --patience 10 \
    --eval_freq 5 \
    --batch_size 2048 \
    --optimizer rsgd \
    --lr 0.001 \
    --weight_decay 0.002 \
    --momentum 0.9 \