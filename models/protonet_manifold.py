""" Prototypical Network 

Author: Zhao Na, 2020
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import knn_graph
from models.knn_graph import knn

from models.dgcnn import DGCNN
from models.attention import SelfAttention


class BaseLearner(nn.Module):
    """The class for inner loop."""
    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i-1]
            self.convs.append(nn.Sequential(
                              nn.Conv1d(in_dim, params[i], 1),
                              nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs-1:
                x = F.relu(x)
        return x


class ProtoNet(nn.Module):
    def __init__(self, args):
        super(ProtoNet, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = args.dist_method
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.alpha = torch.tensor(0.99, requires_grad=False)
        self.sigma = 0.25
        self.k = args.dgcnn_k # treat it as k for knn in manifold
        print(self.k)

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

    
    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x:  n_way x k_shot x in_channels x num_points  [support point cloud]
            support_y:  n_way x k_shot x num_points [support labels]
            query_x:    n_queries x in_channels x num_points [query point cloud]
            query_y:    n_queries x num_points [query labels, each point \in {0,..., n_way}]
        Return:
            query_pred:   [predicted query point cloud]
        """

        ### Manifold learning
        # embeddings
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        query_feat = self.getFeatures(query_x)
        
        emb_all = torch.cat((support_feat, query_feat), 0)
        # do we need this? yes. @UMA disucussion
        emb_s = support_feat.view(-1, support_feat.shape[0] * support_feat.shape[2]) 
        emb_q = query_feat.view(-1, query_feat.shape[0] * query_feat.shape[2])
        emb_all = torch.cat((emb_s, emb_q), 1)
        N, d = emb_all.shape[1], emb_all.shape[0]

        # knn graph 
        eps = np.finfo(float).eps
        emb_all = emb_all.transpose(1, 0) / (self.sigma + eps)  # dxT -> Txd
        emb1 = torch.unsqueeze(emb_all, 1)  # Tx1xd
        emb2 = torch.unsqueeze(emb_all, 0)  # 1xTxd
        W = ((emb1-emb2)**2).mean(2)  #  TxT 
        W = torch.exp(-W/2)

        # keep top-k values
        topk, indices = torch.topk(W, self.k)
        mask = torch.zeros_like(W)
        mask = mask.scatter(1, indices, 1)
        mask = ((mask+torch.t(mask))>0).type(torch.float32)
        W = W*mask

        # normalize
        D = W.sum(0)
        D_sqrt_inv = torch.sqrt(1.0/(D+eps))
        D1 = torch.unsqueeze(D_sqrt_inv,1).repeat(1,N)
        D2 = torch.unsqueeze(D_sqrt_inv,0).repeat(N,1)
        S = D1*W*D2

        # Label propagation, F = (I - \alpha S)^{-1}Y
        ys = support_y.view(self.n_way * self.k_shot, -1)
        yu = torch.zeros(query_y.shape[0], query_y.shape[1]).cuda()
        y = torch.cat((ys, yu), 0)
        F  = torch.matmul(torch.inverse(torch.eye(N).cuda(0)-self.alpha*S+eps), y.view(-1))
        F = F.view(-1, support_x.shape[2])
        Fq = F[4:, :]

        # step4: Cross-Entropy Loss
        ce = nn.CrossEntropyLoss().cuda()
        # both support and query loss

        import pdb
        pdb.set_trace()

        

       
        #return query_pred, loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        if self.use_attention:
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            att_feat = self.att_learner(feat_level2)
            return torch.cat((feat_level1, att_feat, feat_level3), dim=1)
        else:
            # return self.base_learner(self.encoder(x))
            feat_level1, feat_level2 = self.encoder(x)
            feat_level3 = self.base_learner(feat_level2)
            map_feat = self.linear_mapper(feat_level2)
            return torch.cat((feat_level1, map_feat, feat_level3), dim=1)

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
        bg_prototype =  bg_feat.sum(dim=(0,1)) / (self.n_way * self.k_shot)
        return fg_prototypes, bg_prototype

    def calculateSimilarity(self, feat,  prototype, method='cosine', scaler=10):
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
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' %method)
        return similarity

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)
