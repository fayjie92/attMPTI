import torch
import torch.nn as nn
import torch.nn.functional as F
#from models.PointCloudTransformer.model import SPCT
from models.PointCloudTransformer.module import Embedding, OA
# triplet margin loss with distance
from pytorch_metric_learning import losses

# *************** Encoder: SPCT ******************

class SPCT_encoder(nn.Module):
  def __init__(self):
    super().__init__()
    # Note: change this to 3 from xyz, 6 for xyzrgb, and 9 for xyzrgbXYZ
    self.embedding = Embedding(9, 128)
    self.sa1 = OA(128)
    self.sa2 = OA(128)
    self.sa3 = OA(128)
    self.sa4 = OA(128)
    self.linear = nn.Sequential(
      nn.Conv1d(512, 1024, kernel_size=1, bias=False),
      nn.BatchNorm1d(1024),
      nn.LeakyReLU(negative_slope=0.2)
    )
  def forward(self, x):
    x = self.embedding(x)
    x1 = self.sa1(x)
    x2 = self.sa2(x1)
    x3 = self.sa3(x2)
    x4 = self.sa4(x3)
    x = torch.cat([x, x1, x2, x3, x4], dim=1)   # 128 * 5 = 640

    return x

# *************** Segmentation Head ***************

class SegmentHead(nn.Module):
  def __init__(self, n_pts, args):
    super().__init__()
    self.n_pts = n_pts
    self.output_dim = args.output_dim
    self.cls_lbl = args.class_labels

    self.label_conv = nn.Sequential(
      nn.Conv1d(self.n_pts, 64, kernel_size=1, bias=False), 
      nn.BatchNorm1d(64), 
      nn.LeakyReLU(negative_slope=0.2)
    )
    if self.cls_lbl == 1:
      self.convs1 = nn.Conv1d((128 * 5) + 64, 512, 1)
    else:
      self.convs1 = nn.Conv1d((128 * 5), 512, 1) 
    self.convs2 = nn.Conv1d(512, 256, 1)
    self.convs3 = nn.Conv1d(256, 128, 1)
    self.convs4 = nn.Conv1d(128, 64, 1)
    self.bns1 = nn.BatchNorm1d(512)
    self.bns2 = nn.BatchNorm1d(256)
    self.bns3 = nn.BatchNorm1d(128)
    self.bns4 = nn.BatchNorm1d(64)
    self.dp1 = nn.Dropout(0.5)

  def forward(self, x, cls_label):
    batch_size, _, N = x.size()
    if self.cls_lbl == 1:
      cls_label_one_hot = cls_label.view(batch_size, self.n_pts, 1)   
      cls_label_feature = self.label_conv(cls_label_one_hot).repeat(1, 1, N)

      x = torch.cat([x, cls_label_feature], dim=1) 
    else:
      x = x 

    #breakpoint() 
    x1 = F.relu(self.bns1(self.convs1(x)))
    x1 = self.dp1(x1)
    x2 = F.relu(self.bns2(self.convs2(x1)))
    x2 = self.dp1(x2)
    x3 = F.relu(self.bns3(self.convs3(x2)))
    x3 = self.dp1(x3)
    x4 = F.relu(self.bns4(self.convs4(x3)))
    x4 = self.dp1(x4)
    x_all = torch.cat([x, x1, x2, x3, x4], dim=1)

    return x_all
  
# *************** Prototypical Networks with SPCT encoder and Segmentation Head ***************
    
class ProtoNetSPCT(nn.Module):
  def __init__(self, args):
    super(ProtoNetSPCT, self).__init__()
    self.n_way = args.n_way
    self.k_shot = args.k_shot
    self.dist_method = args.dist_method
    self.in_channels = args.pc_in_dim
    self.n_points = args.pc_npts
    self.use_attention = args.use_attention
    self.k = args.k_connect

    self.encoder = SPCT_encoder()
    self.segmentlearner = SegmentHead(self.n_points, args)
    
  def forward(self, support_x, support_y, query_x, query_y):
    """
    Args:
      support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
      support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
      query_x: query point clouds with shape (n_queries, in_channels, num_points)
      query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
    Return:
      query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
    """
    # *************************** Prototypical Networks [Zhao Na work] *************************
    support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
    support_feat = self.getFeatures(support_x, support_y.float())
    support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
    query_feat = self.getFeatures(query_x, query_y.float())

    fg_mask = support_y
    bg_mask = torch.logical_not(support_y)

    support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
    support_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)

    # prototype learning
    fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, support_bg_feat)
    prototypes = [bg_prototype] + fg_prototypes

    # non-parametric metric learning
    similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]

    # (n_queries, n_way+1, num_points)
    query_pred = torch.stack(similarity, dim=1)

    loss = self.computeCrossEntropyLoss(query_pred, query_y)
  
    return query_pred, loss

  def getFeatures(self, x, cls_lbl):
    """
    Forward the input data to network and generate features
    :param x: input data with shape (B, C_in, L)
    :return: features with shape (B, C_out, L)
    """
    feat = self.encoder(x)
    feat = self.segmentlearner(feat, cls_lbl)
    return feat

  def getMaskedFeatures(self, feat, mask):
    """
    Extract foreground and background features via masked average pooling

    Args:
      feat: input features, shape: (n_way, k_shot, feat_dim, num_points)
      mask: binary mask, shape: (n_way, k_shot, num_points)
    Return:
      masked_feat: masked features, shape: (n_way, k_shot, feat_dim)
    """
    mask = mask.unsqueeze(2)
    masked_feat = torch.sum(feat * mask, dim=3) / (mask.sum(dim=3) + 1e-5)
    return masked_feat

  def getPrototype(self, fg_feat, bg_feat):
    """
    Average the features to obtain the prototype

    Args:
      fg_feat: foreground features for each way/shot, shape: (n_way, k_shot, feat_dim)
      bg_feat: background features for each way/shot, shape: (n_way, k_shot, feat_dim)
    Returns:
      fg_prototypes: a list of n_way foreground prototypes, each prototype is a vector with shape (feat_dim,)
      bg_prototype: background prototype, a vector with shape (feat_dim,)
    """
    fg_prototypes = [fg_feat[way, ...].sum(dim=0) / self.k_shot for way in range(self.n_way)]
    bg_prototype = bg_feat.sum(dim=(0, 1)) / (self.n_way * self.k_shot)
    return fg_prototypes, bg_prototype

  def calculateSimilarity(self, feat, prototype, method='cosine', scaler=10):
    """
    Calculate the Similarity between query point-level features and prototypes

    Args:
      feat: input query point-level features
          shape: (n_queries, feat_dim, num_points)
      prototype: prototype of one semantic class
             shape: (feat_dim,)
      method: 'cosine' or 'euclidean', different ways to calculate similarity
      scaler: used when 'cosine' distance is computed.
          By multiplying the factor with cosine distance can achieve comparable performance
          as using squared Euclidean distance (refer to PANet [ICCV2019])
    Return:
      similarity: similarity between query point to prototype
            shape: (n_queries, 1, num_points)
    """
    if method == 'cosine':
      similarity = F.cosine_similarity(feat, prototype[None, ..., None], dim=1) * scaler
    elif method == 'euclidean':
      similarity = - F.pairwise_distance(feat, prototype[None, ..., None], p=2)**2
    else:
      raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)
    return similarity

  def computeCrossEntropyLoss(self, query_logits, query_labels):
    """ Calculate the CrossEntropy Loss for query set
    """
    return F.cross_entropy(query_logits, query_labels)
  
