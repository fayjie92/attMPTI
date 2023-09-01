# point cloud transformer model for few-shot segmentation
# Abdur R. Fayjie

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.PointCloudTransformer.util import sample_and_knn_group
from models.PointCloudTransformer.module import OA, SA  # offset/self attention
# Note: to implement sample_and_ball_group

# Sampling and grouping for neighbour embedding
class SamplingGrouping(nn.Module):
  def __init__(self, s, in_channels, out_channels):
    super(SamplingGrouping, self).__init__()
    
    self.s = s
    self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
    self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm1d(out_channels)
    self.bn2 = nn.BatchNorm1d(out_channels)
  
  def forward(self, x, coords):
    '''
    Parameters:
      x (tensor): additional features, rgbXYZ
      coords (tensor): coordinates, xyz
    Returns: 
      new_xyz (tensor): new coordinates
      new_feature (tensor): new feature 
    '''    
    x = x.permute(0, 2, 1)           # (B, N, in_channels//2)
    new_xyz, new_feature = sample_and_knn_group(s=self.s, k=32, coords=coords, features=x)  # [B, s, 3], [B, s, 32, in_channels]
    b, s, k, d = new_feature.size()
    new_feature = new_feature.permute(0, 1, 3, 2)
    new_feature = new_feature.reshape(-1, d, k)                               # [Bxs, in_channels, 32]
    batch_size = new_feature.size(0)
    new_feature = F.relu(self.bn1(self.conv1(new_feature)))                   # [Bxs, in_channels, 32]
    new_feature = F.relu(self.bn2(self.conv2(new_feature)))                   # [Bxs, in_channels, 32]
    new_feature = F.adaptive_max_pool1d(new_feature, 1).view(batch_size, -1)  # [Bxs, in_channels]
    new_feature = new_feature.reshape(b, s, -1).permute(0, 2, 1)              # [B, in_channels, s]
    return new_xyz, new_feature

    
# Neighbour embedding
class NeighbourEmbedding(nn.Module):
  def __init__(self, samples, in_channel):
    super(NeighbourEmbedding, self).__init__()
    
    self.conv1 = nn.Conv1d(in_channel, 64, kernel_size=1, bias=False)
    self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm1d(64)
    self.bn2 = nn.BatchNorm1d(64)

    self.sg1 = SamplingGrouping(s=samples[0], in_channels=128, out_channels=128)
    self.sg2 = SamplingGrouping(s=samples[1], in_channels=256, out_channels=256)

  def forward(self, x):
    '''
    Parameters:
      x (tensor): [B, 3, N]
    Returns:
      feat (tensor): [B, 256, 256] 
    '''
    xyz = x.permute(0, 2, 1)  # [B, N ,3]
    features = F.relu(self.bn1(self.conv1(x)))        # [B, 64, N]
    features = F.relu(self.bn2(self.conv2(features))) # [B, 64, N]
    xyz1, features1 = self.sg1(features, xyz)         # [B, 128, 512]
    _, features2 = self.sg2(features1, xyz1)          # [B, 256, 256]
    return features2
  

# Point cloud Transformer for few-shot segmentation
class PCT(nn.Module):
  def __init__(self, samples=[512, 256], in_channel=9):
    super().__init__()

    self.neighbor_embedding = NeighbourEmbedding(samples, in_channel)    
    self.oa1 = OA(256)
    self.oa2 = OA(256)
  
  def forward(self, x):
    x = self.neighbor_embedding(x)
    x1 = self.oa1(x)
    x2 = self.oa2(x1)

    x = torch.cat([x, x1, x2], dim=1)

    return x


