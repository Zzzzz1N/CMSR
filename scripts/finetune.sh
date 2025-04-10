# pretrain

export ALPHA=1.2  # 1.2 1.4 1.6
export BETA=1.0  # 0.5 1.0 1.5
export LOSS_TYPE="CE"  # BPR CE WCE
export DATASET="uk"  # ca de fr in jp mx uk us
export CHECKPOINT_DIR="saved/"
export WEIGHT_PATH="saved/CMSR-us-200.pth"


CUDA_VISIBLE_DEVICES=0 python finetune.py \
    --alpha ${ALPHA} \
    --beta ${BETA} \
    --loss_type ${LOSS_TYPE} \
    --dataset ${DATASET} \
    --gpu_id 0 \
    --epochs 100 \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --with_adapter True \
    --weight_path ${WEIGHT_PATH}
    