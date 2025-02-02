import torch
from torch import optim
from torch.nn import functional as F
from models.protonet_pointnet2 import ProtoNetPointNet2
from utils.checkpoint_util import load_pretrain_checkpoint, load_model_checkpoint

class ProtoPointNet2Learner(object):
  def __init__(self, args, mode='train'):
    # init model and optimizer
    self.model = ProtoNetPointNet2(args)
    print(self.model)
    if torch.cuda.is_available():
      self.model.cuda()
    if mode == 'train':
      self.optimizer = torch.optim.Adam(
        [{'params': self.model.encoder.parameters(), 'lr': 0.00001,},
         {'params': self.model.segmentlearner.parameters()}], lr=args.lr) 
           #betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
      # set learning rate scheduler
      # self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                              #step_size=args.step_size, 
                              #gamma=args.gamma)
      self.lr_scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, 
                                              base_lr=0.0001, 
                                              max_lr=0.001, 
                                              step_size_up=1000, 
                                              cycle_momentum=False)
      # load pretrained model for point cloud encoding
      self.model = load_pretrain_checkpoint(self.model, args.pretrain_checkpoint_path)
    elif mode == 'test':
      # Load model checkpoint
      self.model = load_model_checkpoint(
        self.model, args.model_checkpoint_path, mode='test')
    else:
      raise ValueError('Wrong Learner mode (%s)! Option:train/test' % mode)

  def train(self, data):
    """
    Args:
      data: a list of torch tensors wit the following entries.
      - support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
      - support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
      - query_x: query point clouds with shape (n_queries, in_channels, num_points)
      - query_y: query labels with shape (n_queries, num_points)
    """
    [support_x, support_y, query_x, query_y] = data
    self.model.train()
    query_logits, loss = self.model(support_x, support_y, query_x, query_y)

    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    self.lr_scheduler.step()

    query_pred = F.softmax(query_logits, dim=1).argmax(dim=1)
    correct = torch.eq(query_pred, query_y).sum().item()
    accuracy = correct / (query_y.shape[0]*query_y.shape[1])

    return loss, accuracy

  def test(self, data):
    """
    Args:
      support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
      support_y: support masks (foreground) with shape (n_way, k_shot, num_points), each point \in {0,1}.
      query_x: query point clouds with shape (n_queries, in_channels, num_points)
      query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
    """
    [support_x, support_y, query_x, query_y] = data
    self.model.eval()

    with torch.no_grad():
      logits, loss = self.model(support_x, support_y, query_x, query_y)
      pred = F.softmax(logits, dim=1).argmax(dim=1)
      correct = torch.eq(pred, query_y).sum().item()
      accuracy = correct / (query_y.shape[0]*query_y.shape[1])

    return pred, loss, accuracy
