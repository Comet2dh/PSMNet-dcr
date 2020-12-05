#!/usr/bin/env bash
DATAPATH="/media/data1/dh/DataSet/SceneFlowData/"
CUDA_VISIBLE_DEVICES='6,0'
python main.py --dataset sceneflow --cuda $CUDA_VISIBLE_DEVICES \
    --datapath $DATAPATH --trainlist ./filenames/sceneflow_train.txt --testlist ./filenames/sceneflow_test.txt \
    --epochs 16 --batch_size 4 --test_batch_size 4 --lrepochs "10,12,14,16:2" \
    --summary_freq 300 \
    --model PSM_init --logdir ./checkpoints/original