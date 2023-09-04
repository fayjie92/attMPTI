import torch
import torch.nn as nn
import torch.nn.functional as F
# import encoder and segmentation head
from models.Pointnet_Pointnet2_pytorch.models.pointnet2_sem_seg import PointNet2encoder
# focal_loss + dice_loss
from models.PointCloudTransformer.losses import PointNetSegLoss
  
# *** Prototypical Networks: PointNet2 ***

class BaseLearner(nn.Module):
  def __init__(self):
    super(BaseLearner, self).__init__()
    
    self.conv1 = nn.Conv1d(128, 128, 1)
    self.bn1 = nn.BatchNorm1d(128)

  def forward(self, x):
    x = F.relu(self.bn1(self.conv1(x)))
    return x


class ProtoNetPointNet2(nn.Module):
  def __init__(self, args):
    super(ProtoNetPointNet2, self).__init__()
    self.n_way = args.n_way
    self.k_shot = args.k_shot
    self.dist_method = args.dist_method
    self.in_channels = args.pc_in_dim
    self.n_points = args.pc_npts
    self.use_attention = args.use_attention
    self.k = args.k_connect

    self.encoder = PointNet2encoder(input_channel=9)
    self.segmentlearner = BaseLearner()
    
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
    support_feat = self.getFeatures(support_x)
    support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
    query_feat = self.getFeatures(query_x)

    fg_mask = support_y
    bg_mask = torch.logical_not(support_y)

    support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
    support_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)

    # prototype learning
    fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, support_bg_feat)
    prototypes = [bg_prototype] + fg_prototypes

    # non-parametric metric learning
    similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]

    # get predicted class logits (n_queries, n_way+1, num_points)
    query_pred = torch.stack(similarity, dim=1)
    
    # get class predictions
    pred_choice = torch.softmax(query_pred, dim=1).argmax(dim=1)

    #loss = self.computeCrossEntropyLoss(query_pred, query_y)
    loss = self.computeFocalDiceLoss(self.n_way, query_pred, query_y, pred_choice)

    return query_pred, loss

  def getFeatures(self, x):
    """
    Forward the input data to network and generate features
    :param x: input data with shape (B, C_in, L)
    :return: features with shape (B, C_out, L)
    """
    _, feat = self.encoder(x)
    feat = self.segmentlearner(feat)
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
  
  def computeFocalDiceLoss(self, num_class, query_logits, query_labels, pred_choice):
    ''' Calculate the focal+dice loss for query set
    '''
    alpha = torch.ones(num_class+1)
    alpha[-1] = 0.15 # background class
    alpha = alpha.to('cuda')

    gamma =1
    
    criterion = PointNetSegLoss(alpha, gamma, size_average=True, dice=True)
    
    return criterion(query_logits, query_labels, pred_choice)
  
