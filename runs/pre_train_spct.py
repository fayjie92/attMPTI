# pretrain point cloud transformer
# modified version of pretrain for dgcnn by Zhao Na, @fayjie
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from dataloaders.loader import MyPretrainDataset
from utils.logger import init_logger
from utils.checkpoint_util import save_pretrain_checkpoint
from models.PointCloudTransformer.model import SPCT, TinySegmentor, Segmentation

# ********* Segmentation Networks without class labels *********
class SPCTSeg(nn.Module):
  def __init__(self, num_class):
    super().__init__()
  
    self.encoder = SPCT()
    self.segmenter = TinySegmentor(num_class)

  def forward(self, x):
    x_cat, x = self.encoder(x)
    x = self.segmenter(x_cat)
    return x

# ********* Segmentation model with class labels: one-hot labels *********
class SPCTSeg_ClassLabels(nn.Module):
  def __init__(self, num_class):
    super().__init__()
  
    self.encoder = SPCT()
    self.segmenter = Segmentation(num_class)

  def forward(self, x, cls_label):
    x, x_max, x_mean = self.encoder(x)
    x = self.segmenter(x, x_max, x_mean, cls_label)
    return x

def metric_evaluate(predicted_label, gt_label, NUM_CLASS):
  """
  :param predicted_label: (B,N) tensor
  :param gt_label: (B,N) tensor
  :return: iou: scaler
  """
  gt_classes = [0 for _ in range(NUM_CLASS)]
  positive_classes = [0 for _ in range(NUM_CLASS)]
  true_positive_classes = [0 for _ in range(NUM_CLASS)]

  for i in range(gt_label.size()[0]):
    pred_pc = predicted_label[i]
    gt_pc = gt_label[i]

    for j in range(gt_pc.shape[0]):
      gt_l = int(gt_pc[j])
      pred_l = int(pred_pc[j])
      gt_classes[gt_l] += 1
      positive_classes[pred_l] += 1
      true_positive_classes[gt_l] += int(gt_l == pred_l)

  oa = sum(true_positive_classes)/float(sum(positive_classes))
  print('Overall accuracy: {0}'.format(oa))
  iou_list = []

  for i in range(NUM_CLASS):
    iou_class = true_positive_classes[i] / float(gt_classes[i]+positive_classes[i]-true_positive_classes[i])
    print('Class_%d: iou_class is %f' % (i, iou_class))
    iou_list.append(iou_class)

  mean_IoU = np.array(iou_list[1:]).mean()
  return oa, mean_IoU, iou_list

def pretrain(args):
  logger = init_logger(args.log_dir, args)

  # Init datasets, dataloaders, and writer
  PC_AUGMENT_CONFIG = {'scale': args.pc_augm_scale,
                       'rot': args.pc_augm_rot,
                       'mirror_prob': args.pc_augm_mirror_prob,
                       'jitter': args.pc_augm_jitter
                      }

  if args.dataset == 's3dis':
    from dataloaders.s3dis import S3DISDataset
    DATASET = S3DISDataset(args.cvfold, args.data_path)
  elif args.dataset == 'scannet':
    from dataloaders.scannet import ScanNetDataset
    DATASET = ScanNetDataset(args.cvfold, args.data_path)
  else:
    raise NotImplementedError('Unknown dataset %s!' % args.dataset)

  CLASSES = DATASET.train_classes
  NUM_CLASSES = len(CLASSES) + 1
  CLASS2SCANS = {c: DATASET.class2scans[c] for c in CLASSES}

  TRAIN_DATASET = MyPretrainDataset(args.data_path, CLASSES, CLASS2SCANS, mode='train',
                    num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                    pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

  VALID_DATASET = MyPretrainDataset(args.data_path, CLASSES, CLASS2SCANS, mode='test',
                    num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                    pc_augm=args.pc_augm, pc_augm_config=PC_AUGMENT_CONFIG)

  logger.cprint('=== Pre-train Dataset (classes: {0}) | Train: {1} blocks | Valid: {2} blocks ==='.format(
                           CLASSES, len(TRAIN_DATASET), len(VALID_DATASET)))

  TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=True,
                drop_last=True)

  VALID_LOADER = DataLoader(VALID_DATASET, batch_size=args.batch_size, num_workers=args.n_workers, shuffle=False,
                drop_last=True)

  WRITER = SummaryWriter(log_dir=args.log_dir)

  # Init model and optimizer
  print(args.class_labels)
  if args.class_labels==1:
    model = SPCTSeg_ClassLabels(NUM_CLASSES)
  elif args.class_labels==0:
    model = SPCTSeg(NUM_CLASSES)
  else:
    print('wrong choice, choices: [0: no class labels, 1: class labels]')
  
  if torch.cuda.is_available():
    model.cuda()

  # optimizer with same learning rate for the whole model @fayjie
  optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, 
                               betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=args.pretrain_weight_decay)
  
  # Set learning rate scheduler
  lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.pretrain_step_size, gamma=args.pretrain_gamma)

  # train
  best_iou = 0
  global_iter = 0
  for epoch in range(args.n_iters):
    model.train()
    for batch_idx, (ptclouds, labels) in enumerate(TRAIN_LOADER):
      if torch.cuda.is_available():
        ptclouds = ptclouds.cuda()
        labels = labels.cuda()
        one_hot_class_labels = F.one_hot(labels)

      if args.class_labels==1:
        logits = model(ptclouds, one_hot_class_labels)
      elif args.class_labels==0:
        logits = model(ptclouds)
      else:
        print('wrong choice, choices: [0: no class labels, 1: class labels]')
      
      loss = F.cross_entropy(logits, labels)

      # Loss backwards and optimizer updates
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      WRITER.add_scalar('Train/loss', loss, global_iter)
      logger.cprint('=====[Train] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, batch_idx, loss.item()))
      global_iter += 1

    lr_scheduler.step()

    if (epoch) % args.eval_interval == 0:
      pred_total = []
      gt_total = []
      model.eval()
      with torch.no_grad():
        for i, (ptclouds, labels) in enumerate(VALID_LOADER):
          gt_total.append(labels.detach())

          if torch.cuda.is_available():
            ptclouds = ptclouds.cuda()
            labels = labels.cuda()
          
          if args.class_labels==1:
            logits = model(ptclouds, one_hot_class_labels)
          elif args.class_labels==0:
            logits = model(ptclouds)
          else:
            print('wrong choice, choices: [0: no class labels, 1: class labels]')
            
          loss = F.cross_entropy(logits, labels)

          # 　Compute predictions
          _, preds = torch.max(logits.detach(), dim=1, keepdim=False)
          pred_total.append(preds.cpu().detach())

          WRITER.add_scalar('Valid/loss', loss, global_iter)
          logger.cprint('=====[Valid] Epoch: %d | Iter: %d | Loss: %.4f =====' % (epoch, i, loss.item()))

      pred_total = torch.stack(pred_total, dim=0).view(-1, args.pc_npts)
      gt_total = torch.stack(gt_total, dim=0).view(-1, args.pc_npts)
      accuracy, mIoU, iou_perclass = metric_evaluate(pred_total, gt_total, NUM_CLASSES)
      logger.cprint('===== EPOCH [%d]: Accuracy: %f | mIoU: %f =====\n' % (epoch, accuracy, mIoU))
      WRITER.add_scalar('Valid/overall_accuracy', accuracy, global_iter)
      WRITER.add_scalar('Valid/meanIoU', mIoU, global_iter)

      if mIoU > best_iou:
        best_iou = mIoU
        logger.cprint('*******************Model Saved*******************')
        save_pretrain_checkpoint(model, args.log_dir)

  WRITER.close()
