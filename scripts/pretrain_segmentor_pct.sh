GPU_ID=0

# Dataset settings
DATASET='s3dis'
SPLIT=0
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1'
SAVE_PATH='./log_s3dis/'
NUM_POINTS=2048
PC_ATTRIBS='xyzrgbXYZ'
# K=20

# Training settings
EVAL_INTERVAL=1
BATCH_SIZE=16
NUM_WORKERS=16
NUM_EPOCHS=25
LR=0.001
WEIGHT_DECAY=0.0001
DECAY_STEP=50
DECAY_RATIO=0.5

# settings specific to SPCT (transformer)
NBLOCKS=4
NNEIGHBOR=16
CLASS_LABELS=0


args=(--phase 'pretrain_pct' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --n_iters $NUM_EPOCHS --eval_interval $EVAL_INTERVAL
      --batch_size $BATCH_SIZE --n_workers $NUM_WORKERS
      --pretrain_lr $LR --pretrain_weight_decay $WEIGHT_DECAY
      --pretrain_step_size $DECAY_STEP --pretrain_gamma $DECAY_RATIO
      --nblocks $NBLOCKS --nneighbor $NNEIGHBOR --class_labels $CLASS_LABELS)

CUDA_VISIBLE_DEVICES=$GPU_ID venv/bin/python3 main.py "${args[@]}"
