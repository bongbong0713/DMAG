#!/bin/bash

data_root='../multimodal-prompt-learning/data'
testsets=$1
gpu=$2
dir=$3
# arch=RN50
arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a

python ./tpt_oracle.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} --lr 0.003 -p 100 --no-reset --selection_p 0.1 --output-dir ${dir} \
--tpt --ctx_init ${ctx_init}