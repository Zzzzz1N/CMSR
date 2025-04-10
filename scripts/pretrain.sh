# pretrain

export ALPHA=1.2  # 1.2 1.4 1.6
export BETA=1.0  # 0.5 1.0 1.5
export LOSS_TYPE="WCE"  # BPR CE WCE
export DATASET="us"  # ca de fr in jp mx uk us
export CHECKPOINT_DIR="saved/"


CUDA_VISIBLE_DEVICES=0 python pretrain.py \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --loss_type ${LOSS_TYPE} \
    --dataset ${DATASET} \
    --gpu_id 0 \
    --pretrain_epochs 300 \
    --save_step 50 \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --with_adapter False
    