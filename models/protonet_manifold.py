""" 
Transductive Few-Shot Segmentation based on Prototypical Networks with Manifold Regularizer.
Author: Abdur R. Fayjie & Umamaheswaran Raman Kumar 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ProtoNetManifold(nn.Module):
    def __init__(self, args):
        super(ProtoNetManifold, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = args.dist_method
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        
        # parameters for manifold regularization
        # lambda: regularizer factor
        self.reg_factor = 0.5
        self.alpha = torch.tensor(0.99, requires_grad=False)
        self.sigma = 0.25
        self.k = args.dgcnn_k # treat it as k for knn in manifold

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

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
        # --------------------- Prototypical Networks starts here -----------------------
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        query_feat = self.getFeatures(query_x) #(n_queries, feat_dim, num_points)

        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
        suppoer_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)
        
        # prototype learning
        fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, suppoer_bg_feat)
        prototypes = [bg_prototype] + fg_prototypes

        # non-parametric metric learning
        similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]

        query_pred = torch.stack(similarity, dim=1) #(n_queries, n_way+1, num_points)
        # ------------------------ Prototypical Network ends here -------------------------

        
        
        if self.training:
            # ------------------------ Manifold learning starts here --------------------------
            support_feat = support_feat.view(self.n_way*self.k_shot, -1, self.n_points) 

            size_s = support_feat.shape[0]
            size_q = query_feat.shape[0]
            
            
            emb_s = support_feat.permute(0,2,1)
            emb_s = emb_s.view(emb_s.shape[0] * emb_s.shape[1], -1) 
            #[5N, 192]
            emb_q = query_feat.permute(0,2,1)
            emb_q = emb_q.view(emb_q.shape[0] * emb_q.shape[1], -1)

            #[N+(5*N), 192] -> [T, 192] -> T = N+(5*N)
            emb_all = torch.cat((emb_s, emb_q), 0)
            #N, dim=192
            T, dim = emb_all.shape[0], emb_all.shape[1]

            # step2: knn graph 
            eps = np.finfo(float).eps
            #[T, 192]
            emb_all = emb_all / (self.sigma + eps) 
            #[T, 1, 192]
            emb1 = torch.unsqueeze(emb_all, 1)  
            #[1, T, 192]
            emb2 = torch.unsqueeze(emb_all, 0)  # 1xTxd
            # [T, T]
            W = ((emb1-emb2)**2).mean(2) 
            # [T, T] 
            W = torch.exp(-W/2) 

            # keep top-k values, k=20
            #[T, 20=k], [T, 20=k]   
            topk, indices = torch.topk(W, self.k)
            #[T, T]
            mask = torch.zeros_like(W)
            #[T, T]
            mask = mask.scatter(1, indices, 1)
            #[T, T]
            mask = ((mask+torch.t(mask))>0).type(torch.float32)
            #[T, T]
            W = W*mask

            # normalize
            #[T]
            D = W.sum(0)
            #[T]
            D_sqrt_inv = torch.sqrt(1.0/(D+eps))
            #[T, T]
            D1 = torch.unsqueeze(D_sqrt_inv,1).repeat(1,T)
            #[T, T]
            D2 = torch.unsqueeze(D_sqrt_inv,0).repeat(T,1)
            #[T, T]
            S = D1*W*D2
            

            # step3: Label propagation, pred* = (I - \alpha S)^{-1} *Y
            # support_y: [1= way, 1= shot, N]
            #[1=shot, N, 1=way]
            ys = support_y.view(-1)
            ys = F.one_hot(ys.long(), self.n_way+1)
            # [num_query, N, 1=way]
            yu = torch.zeros(emb_q.shape[0], self.n_way+1).cuda()
            #[1+5, N, 1]
            y = torch.cat((ys, yu), 0)
            
            #[(1+5)*N, 1]
            pred  = torch.matmul(torch.inverse(torch.eye(T).cuda()-self.alpha*S+eps), y) ###


            #[(1+5), N, 1=way] 
            pred = pred.view(size_s + size_q, -1, self.n_way+1)
            #[5, N, 1=way]

            predq = pred[size_s: , :, :]

            preds = pred[:size_s, :, :]
            
            # step4: Cross-Entropy Loss
            ce = nn.CrossEntropyLoss().cuda()
        
            # both support and query loss
            gt = torch.cat((support_y.view(self.n_way* self.k_shot, -1), query_y), 0)
            
            loss_manifold = ce(pred.view(-1, self.n_way+1), gt.view(-1))
            
            # print('regularized_loss')  #to verify if correct loss is called for train/test
            loss = self.computeCrossEntropyLoss(query_pred, query_y) + self.reg_factor*loss_manifold

            # ------------------------ Manifold learning ends here ----------------------------
        else:
            # print('ce_loss')  #to verify if correct loss is called for train/test
            loss = self.computeCrossEntropyLoss(query_pred, query_y)
        
        return query_pred, loss

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
