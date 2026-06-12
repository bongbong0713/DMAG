#!/bin/bash

data_root='../multimodal-prompt-learning/data'
testsets=$1
gpu=$2
arch=ViT-B/16
# arch=ViT-B/16
bs=64
ctx_init=a_photo_of_a
proto_ctx_init=a_photo_of_a


python ./tpt_classification1.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} --lr 0.01 --no-reset --proto_lambda 1.0 --proto_contrastive_lambda 1.0 --alpha 1.0 --beta 5.0 --proto_reset_step 1 \
--tpt --ctx_init ${ctx_init} --proto_ctx_init ${proto_ctx_init}