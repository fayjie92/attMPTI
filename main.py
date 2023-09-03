#!/usr/bin/env python
# Main function by Zhao Na, 2020
# Modified by Abdur R. Fayjie and Umamaheswaran Raman Kumar, 2023
import ast
import argparse
from datetime import datetime

# log each experiment by current time
now = datetime.now()
now_str = now.strftime("%d%m%Y_%H%M%S")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()

  #data
  parser.add_argument('--phase', type=str, default='prototrain_spct', 
                      choices=['pretrain_dgcnn', 'finetune_dgcnn', 
                               'prototrain_dgcnn', 'protoeval_dgcnn',
                               'mptitrain_dgcnn', 'mptieval_dgcnn', 
                               'protofoldtrain_dgcnn',
                               'pretrain_spct','pretrain_naivepct','pretrain_pct',
                               'prototrain_spct',
                               'pretrain_pointnet2'])
  
  parser.add_argument('--dataset', type=str, default='s3dis', help='Dataset name: s3dis|scannet')
  parser.add_argument('--cvfold', type=int, default=0, help='Fold left-out for testing in leave-one-out setting'
                      'Options:{0,1}')
  parser.add_argument('--data_path', type=str, default='./datasets/S3DIS/blocks_bs1_s1',
                      help='Directory to the source data')
  parser.add_argument('--pretrain_checkpoint_path', type=str, default=None,
                      help='Path to the checkpoint of pre model for resuming')
  parser.add_argument('--model_checkpoint_path', type=str, default=None,
                      help='Path to the checkpoint of model for resuming')
  parser.add_argument('--save_path', type=str, default='./log_s3dis/',
                      help='Directory to the save log and checkpoints')
  parser.add_argument('--eval_interval', type=int, default=1500,
                      help='iteration/epoch inverval to evaluate model')

  #optimization
  parser.add_argument('--batch_size', type=int, default=32, help='Number of samples/tasks in one batch')
  parser.add_argument('--n_workers', type=int, default=16, help='number of workers to load data')
  parser.add_argument('--n_iters', type=int, default=30000, help='number of iterations/epochs to train')

  parser.add_argument('--lr', type=float, default=0.001,
                      help='Model (eg. protoNet or MPTI) learning rate [default: 0.001]')
  parser.add_argument('--step_size', type=int, default=5000, help='Iterations of learning rate decay')
  parser.add_argument('--gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')
  parser.add_argument('--pretrain_lr', type=float, default=0.001, help='pretrain learning rate [default: 0.001]')
  parser.add_argument('--pretrain_weight_decay', type=float, default=0., help='weight decay for regularization')
  parser.add_argument('--pretrain_step_size', type=int, default=50, help='Period of learning rate decay')
  parser.add_argument('--pretrain_gamma', type=float, default=0.5, help='Multiplicative factor of learning rate decay')

  #few-shot episode setting
  parser.add_argument('--n_way', type=int, default=2, help='Number of classes for each episode: 1|3')
  parser.add_argument('--k_shot', type=int, default=1, help='Number of samples/shots for each class: 1|5')
  parser.add_argument('--n_queries', type=int, default=1, help='Number of queries for each class')
  parser.add_argument('--n_episode_test', type=int, default=100,
                      help='Number of episode per configuration during testing')

  # Point cloud processing
  parser.add_argument('--pc_npts', type=int, default=2048, help='Number of input points for PointNet.')
  parser.add_argument('--pc_attribs', default='xyzrgbXYZ',
                      help='Point attributes fed to PointNets, if empty then all possible. '
                      'xyz = coordinates, rgb = color, XYZ = normalized xyz')
  parser.add_argument('--pc_augm', action='store_true', help='Training augmentation for points in each superpoint')
  parser.add_argument('--pc_augm_scale', type=float, default=0,
                      help='Training augmentation: Uniformly random scaling in [1/scale, scale]')
  parser.add_argument('--pc_augm_rot', type=int, default=1,
                      help='Training augmentation: Bool, random rotation around z-axis')
  parser.add_argument('--pc_augm_mirror_prob', type=float, default=0,
                      help='Training augmentation: Probability of mirroring about x or y axes')
  parser.add_argument('--pc_augm_jitter', type=int, default=1,
                      help='Training augmentation: Bool, Gaussian jittering of all attributes')

  # feature extraction network -> DGCNN configuration
  parser.add_argument('--dgcnn_k', type=int, default=20, help='Number of nearest neighbors in Edgeconv')
  parser.add_argument('--edgeconv_widths', default='[[64,64], [64,64], [64,64]]', help='DGCNN Edgeconv widths')
  parser.add_argument('--dgcnn_mlp_widths', default='[512, 256]', help='DGCNN MLP (following stacked Edgeconv) widths')
  parser.add_argument('--base_widths', default='[128, 64]', help='BaseLearner widths')
  parser.add_argument('--output_dim', type=int, default=64,
                      help='The dimension of the final output of attention learner or linear mapper')
  parser.add_argument('--use_attention', action='store_true', help='if incorporate attention learner')

  # feature extraction network -> point transformer configuration
  parser.add_argument('--nblocks', type=int, default=4, help='Number of transformer blocks')
  parser.add_argument('--nneighbor', type=int, default=16, help='Nearest neighbors in transformer model')
  parser.add_argument('--transformer_dim', type=int, default=512, help='Transformer block dimension')
  parser.add_argument('--class_labels', type=int, default=0, 
                      help='One-hot features in segmenter for transformer models')

  # protoNet configuration
  parser.add_argument('--dist_method', default='euclidean',
                      help='Method to compute distance between query feature maps and prototypes.'
                      '[Option: cosine|euclidean]')

  # MPTI configuration
  parser.add_argument('--n_subprototypes', type=int, default=100,
                      help='Number of prototypes for each class in support set')
  parser.add_argument('--k_connect', type=int, default=200,
                      help='Number of nearest neighbors to construct local-constrained affinity matrix')
  parser.add_argument('--sigma', type=float, default=1., help='hyeprparameter in gaussian similarity function')

  args = parser.parse_args()

  args.edgeconv_widths = ast.literal_eval(args.edgeconv_widths)
  args.dgcnn_mlp_widths = ast.literal_eval(args.dgcnn_mlp_widths)
  args.base_widths = ast.literal_eval(args.base_widths)
  args.pc_in_dim = len(args.pc_attribs)

  print('experiment logs are stored in "log_<dataset>" directory')

  # ********* choices of phase: corresponding functions and log directories *********

  # -dgcnn- 
  if args.phase=='protoeval_dgcnn' or args.phase=='mptieval_dgcnn':
    args.log_dir = args.model_checkpoint_path
    from runs.eval import eval
    eval(args)
  elif args.phase=='pretrain_dgcnn':
    args.log_dir = args.save_path + 'log_pretrain_dgcnn%s_S%d_T%s' % (args.dataset, args.cvfold, now_str)
    from runs.pre_train_dgcnn import pretrain
    pretrain(args)
  elif args.phase=='finetune_dgcnn':
    args.log_dir = args.save_path + 'log_finetune_dgcnn%s_S%d_N%d_K%d_T%s' % (args.dataset, args.cvfold,
                                                                              args.n_way, args.k_shot, now_str)
    from runs.fine_tune_dgcnn import finetune
    finetune(args)
  elif args.phase=='prototrain_dgcnn':
    args.log_dir = args.save_path + 'log_prototrain_dgcnn%s_S%d_N%d_K%d_Att%d_T%s' %(args.dataset, args.cvfold,
                                                                                     args.n_way, args.k_shot,
                                                                                     args.use_attention, now_str)
    from runs.proto_train_dgcnn import train
    train(args)
  elif args.phase=='mptitrain_dgcnn':
    args.log_dir = args.save_path + 'log_mptitrain_dgcnn%s_S%d_N%d_K%d_Att%d_T%s' % (args.dataset, args.cvfold,
                                                                                     args.n_way, args.k_shot,
                                                                                     args.use_attention, now_str)
    from runs.mpti_train_dgcnn import train
    train(args)
  elif args.phase=='lpmanifold_dgcnn':
    args.log_dir = args.save_path + 'log_lpmanifold_dgcnn%s_S%d_N%d_K%d_Att%d_T%s' %(args.dataset, args.cvfold,
                                                                                     args.n_way, args.k_shot,
                                                                                     args.use_attention, now_str)
    from runs.lp_manifold_train import train
    train(args)
  elif args.phase=='protofoldtrain_dgcnn':
    args.log_dir = args.save_path + 'log_protomanifold_dgcnn%s_S%d_N%d_K%d_Att%d_T%s' %(args.dataset, args.cvfold,
                                                                                        args.n_way, args.k_shot,
                                                                                        args.use_attention, now_str)
    from runs.proto_manifold_train import train
    train(args)

  # -transformers-
  elif args.phase=='pretrain_spct':
    args.log_dir = args.save_path + 'log_pretrain_spct_%s_S%d_T%s' % (args.dataset, args.cvfold, now_str)
    from runs.pre_train_spct import pretrain
    pretrain(args)
  elif args.phase=='pretrain_naivepct':
    args.log_dir = args.save_path + 'log_pretrain_naivepct_%s_S%d_T%s' % (args.dataset, args.cvfold, now_str)
    from runs.pre_train_naivepct import pretrain
    pretrain(args)
  elif args.phase=='pretrain_pct':
    args.log_dir = args.save_path + 'log_pretrain_pct_%s_S%d_T%s' % (args.dataset, args.cvfold, now_str)
    from runs.pre_train_pct import pretrain
    pretrain(args)    
  elif args.phase=='prototrain_spct':
    args.log_dir = args.save_path + 'log_prototrain_spct_%s_S%d_N%d_K%d_T%s' % (args.dataset, args.cvfold, 
                                                                                args.n_way, args.k_shot, now_str)
    from runs.proto_train_spct import train
    train(args) 
  
  # -pointnet2-
  elif args.phase=='pretrain_pointnet2':
    args.log_dir = args.save_path + 'log_pretrain_pointnet2_%s_S%d_T%s' % (args.dataset, args.cvfold, now_str)
    from runs.pre_train_pointnet2 import pretrain
    pretrain(args)

  else:
    raise ValueError('Please set correct phase.')