# comment it to run in apollo server
# GPU_ID=0

# Dataset settings
DATASET='s3dis'
SPLIT=0
DATA_PATH='./datasets/S3DIS/blocks_bs1_s1'
SAVE_PATH='./log_s3dis/'
NUM_POINTS=2048  
PC_ATTRIBS='xyzrgbXYZ'

# Data augmentation
PC_AUGM_SCALE=0  #0
PC_AUGM_ROT=1
PC_AUG_MIRROR_PROB=0  #0
PC_AUGM_JITTER=1

# Pretrained checkpoints
PRETRAIN_CHECKPOINT='./pretrained/log_pretrain_pointnet2_s3dis_S0_T03092023_220946' 

# Prototypical Network settings
N_WAY=2
K_SHOT=1
N_QUERIES=1
N_TEST_EPISODES=100
DIST_METHOD='cosine'  # choice ['cosune', 'euclidean']

# Training settings
NUM_ITERS=40000
EVAL_INTERVAL=100
LR=0.0001   
DECAY_STEP=5000
DECAY_RATIO=0.5

# Settings specific to SPCT (transformer)
# NBLOCKS=4  # currently not in use
# NNEIGHBOR=16  # currently not in use
CLASS_LABELS=0

args=(--phase 'prototrain_pointnet2' --dataset "${DATASET}" --cvfold $SPLIT
      --data_path  "$DATA_PATH" --save_path "$SAVE_PATH"
      --pretrain_checkpoint_path "$PRETRAIN_CHECKPOINT" 
      --dist_method "$DIST_METHOD" --class_labels $CLASS_LABELS
      --pc_npts $NUM_POINTS --pc_attribs "$PC_ATTRIBS" --pc_augm
      --pc_augm_scale $PC_AUGM_SCALE --pc_augm_rot $PC_AUGM_ROT
      --pc_augm_mirror_prob $PC_AUG_MIRROR_PROB --pc_augm_jitter $PC_AUGM_JITTER
      --n_iters $NUM_ITERS --eval_interval $EVAL_INTERVAL --batch_size 1
      --lr $LR  --step_size $DECAY_STEP --gamma $DECAY_RATIO
      --n_way $N_WAY --k_shot $K_SHOT --n_queries $N_QUERIES 
      --n_episode_test $N_TEST_EPISODES)

#CUDA_VISIBLE_DEVICES=$GPU_ID 
venv/bin/python main.py "${args[@]}"
