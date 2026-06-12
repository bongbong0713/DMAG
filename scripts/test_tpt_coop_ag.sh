#!/bin/bash

data_root='../multimodal-prompt-learning/data'
coop_weight='./weights/to_gdrive/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
testsets=$1
gpu=$2
arch=ViT-B/16
# arch=ViT-B/16
bs=64

python ./tpt_classification1.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} --lr 0.005 --no-reset --proto_lambda 1.0 --proto_contrastive_lambda 0.5 --alpha 2.0 --beta 2.0 \
--tpt --load ${coop_weight}