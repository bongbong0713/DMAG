#!/bin/bash

data_root='../multimodal-prompt-learning/data'
coop_weight='./weights/to_gdrive/rn50_ep50_16shots/nctx16_cscFalse_ctpend/seed1/model.pth.tar-50'
testsets=$1
arch=RN50
# arch=ViT-B/16
bs=64

python ./tpt_classification.py ${data_root} --test_sets ${testsets} \
-a ${arch} -b ${bs} --gpu 5 \
--tpt --cocoop --load ${cocoop_weight}