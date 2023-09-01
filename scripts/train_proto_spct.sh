# comment it to run in apollo server
# GPU_ID=0

# Dataset settings
DATASET='s3dis'
SPLIT=0
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1'
SAVE_PATH='./log_s3dis/'
NUM_POINTS=2048  
PC_ATTRIBS='xyzrgbXYZ'

# Pretrained checkpoints
# log_pretrain_spct_s3dis_S0_T25082023_012142 -> 72% : SPCT : 2048 pts 
# log_pretrain_spct_s3dis_S0_T25082023_051904 -> 81% : SPCT : 2048 pts : cls_lbl
PRETRAIN_CHECKPOINT='./pretrained/log_pretrain_spct_s3dis_S0_T25082023_012142' 

# Prototypical Network settings
N_WAY=3
K_SHOT=5
N_QUERIES=1
N_TEST_EPISODES=100
DIST_METHOD='cosine'  # choice ['cosune', 'euclidean']

# Training settings
NUM_ITERS=40000
EVAL_INTERVAL=1000
LR=0.001   
DECAY_STEP=5000
DECAY_RATIO=0.5

# Settings specific to SPCT (transformer)
# NBLOCKS=4  # currently not in use
# NNEIGHBOR=16  # currently not in use
CLASS_LABELS=1

args=(--phase 'prototrain_spct' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT" 
      --dist_method "$DIST_METHOD" --class_labels $CLASS_LABELS
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR  --step_size $DECAY_STEP --gamma $DECAY_RATIO
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUERIES 
      --n_episode_test $N_TEST_EPISODES)

#CUDA_VISIBLE_DEVICES=$GPU_ID 
venv/bin/python main.py "${args[@]}"
