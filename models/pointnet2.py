#!/usr/bin/env python3
"""
Pointnet2 segmentation model
"""
import torch.nn as nn
from models.encoder.pointnet2_backbone import Pointnet2Backbone


# *********segmentation head *********

class SegHead(nn.Module):
  def __init__(self, num_class):
    super(SegHead, self).__init__()

    self.fc_layer = nn.Sequential(
      nn.Conv1d(128, 128, kernel_size=1, bias=False),
      nn.BatchNorm1d(128),
      nn.ReLU(True),
      nn.Dropout(0.5),
      nn.Conv1d(128, num_class, kernel_size=1),
    )
  
  def forward(self, x): 
    return self.fc_layer(x)


# ******** segmentation head ********

class PointNet2SemSegPretrain(nn.Module):
  def __init__(self, num_class, input_feature_dim):
    super(PointNet2SemSegPretrain, self).__init__()

    self.num_class = num_class
    self.input_feature_dim = input_feature_dim
    self.encoder = Pointnet2Backbone(c_dim=128, padding=0.1, 
                                     input_feature_dim=self.input_feature_dim)
    self.seg_head = SegHead(self.num_class)

  def forward(self, x):
    # x (tensor): input point cloud
    pc, features = self.encoder(x)

    print('pc shape: ', pc)

    out = self.seg_head(features)
    return out

    
   