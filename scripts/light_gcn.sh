#!/bin/bash
python main.py \
    --model light_gcn \
    --layers 3 \
    --dim 64 \
    --scale 0.1 \
    --margin 1.0 \
    --k_list [50] \
    --random_seed 2020 \
    --cuda True \
    --device 0 \
    --epochs 500 \
    --patience 5 \
    --eval_freq 5 \
    --batch_size 1024 \
    --optimizer sgd \
    --lr 0.001 \
    --weight_decay 0.001 \
    --momentum 0.9 \