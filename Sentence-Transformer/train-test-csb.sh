#!/bin/bash
# useage : ./train.sh 0 100 2022-03-11 1 128 --is-test

GPU_ID=$1
EPOCHS=$2
YMD=$3
DUPLICATES=$4
OUTPUT_DIM=$5
IS_TEST=$6

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

nohup /usr/bin/python /home1/irteam/sbchoi/workspace/Work/BM/3rd_modelling/matching-ko/bm_train/train_pca.py \
    --root-dir /home1/irteam/sbchoi/workspace/Work/BM/3rd_modelling \
    --data-dir /home1/irteam/sbchoi/workspace/Work/BM/3rd_modelling/data \
    --pt-model-dir /pretrained_model/roberta_68M \
    --output-model-dir /output_model \
    --larva-model larva/roberta-68M-larva-text-kr-distil	 \
    --task classify \
    --batch-size 64 \
    --train-size 1000000 \
    --valid-size 100000 \
    --data-name q2q-exp-data \
    --epochs ${EPOCHS} \
    --warmup-rate 0.1 \
    --loss-type online_contrastive \
    --gpu ${GPU_ID} \
    --ymd ${YMD} \
    --test-size 200000 \
    --tensorboard-path /home1/irteam/sbchoi/workspace/Work/BM/3rd_modelling/log_dir/ \
    --train-dir train \
    --ngpus 8 \
    --start-gpu-num ${GPU_ID} \
    --rank 0 \
    --duplicates ${DUPLICATES} \
    --output-dim ${OUTPUT_DIM} \
    ${IS_TEST} > ./train_csb.log 2>&1 &
