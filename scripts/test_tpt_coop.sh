#!/bin/bash

data_root='../multimodal-prompt-learning/data'
coop_weight='./weights/to_gdrive/vit_b16_ep50_16shots/nctx4_cscFalse_ctpend/seed1/prompt_learner/model.pth.tar-50'
testsets=$1
gpu=$2
dir=$3
# arch=RN50
arch=ViT-B/16
bs=64

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu ${gpu} --reset --output-dir ${dir} \
--tpt --load ${coop_weight}

# --load ${coop_weight}