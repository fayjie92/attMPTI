#Transductive Few-Shot Segmentation based on Prototypical Networks with Manifold Regularizer.
#Author: Abdur R. Fayjie & Umamaheswaran Raman Kumar 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import faiss
from faiss import normalize_L2
import scipy

from models.dgcnn import DGCNN
from models.attention import SelfAttention

class BaseLearner(nn.Module):
    # *** The class for inner loop ***
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
        
        self.l1 = 0.85 # loss1: cross-entropy  
        self.l2 = 0.15 # loss2: manifold
        
        self.alpha = torch.tensor(0.99, requires_grad=False)
        self.sigma = 0.25
        self.k = args.dgcnn_k # treat it as k for knn in manifold

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)

    def forward(self, support_x, support_y, query_x, query_y) -> 'query_pred':

        # support_x: [n_way, k_shot, in_channels, num_points]
        # support_y: [n_way, k_shot, num_points]
        # query_x: [n_queries, in_channels, num_points]
        # query_y: [n_queries, num_points] 
        
        # ***************** Prototypical Networks ********************
        
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
        
        #if self.training:
        # ************** Manifold learning ***************

        # ***** Embedding *****

        support_feat = support_feat.view(self.n_way*self.k_shot, -1, self.n_points) 

        size_s = support_feat.shape[0]
        size_q = query_feat.shape[0]
        emb_s = support_feat.permute(0,2,1)
        emb_s = emb_s.reshape(emb_s.shape[0] * emb_s.shape[1], -1) 

        emb_q = query_feat.permute(0,2,1)
        emb_q = emb_q.reshape(emb_q.shape[0] * emb_q.shape[1], -1)

        emb_all = torch.cat((emb_s, emb_q), 0)

        T, dim = emb_all.shape[0], emb_all.shape[1]

        # Debug
        # kNN search for the graph
        breakpoint()
        X = emb_all.cpu().detach().numpy()
        d = X.shape[1]
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.device = int(torch.cuda.device_count()) - 1
        index = faiss.GpuIndexFlatIP(res,d,flat_config)   # build the index

        normalize_L2(X)
        index.add(X) 
        N = X.shape[0]
        Nidx = index.ntotal
        
        D, I = index.search(X, self.k + 1)

        # Create the graph
        D = D[:,1:] ** 3
        I = I[:,1:]
        row_idx = np.arange(N)
        row_idx_rep = np.tile(row_idx,(self.k,1)).T
        W = scipy.sparse.csr_matrix((D.flatten('F'), (row_idx_rep.flatten('F'), I.flatten('F'))), shape=(N, N))
        W = W + W.T

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis = 1)
        S[S==0] = 1
        D = np.array(1./ np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1))
        Wn = D * W * D
        
         # Initiliaze the y vector for each class (eq 5 from the paper, normalized with the class size) and apply label propagation
        Z = np.zeros((N,self.n_way+1))
        A = scipy.sparse.eye(Wn.shape[0]) - 0.99 * Wn
        labels = np.concatenate([support_y.view(-1).detach().cpu(), np.zeros(emb_q.shape[0]*self.n_way)])
        labeled_idx = np.where(labels >= 0)[0][:size_s*self.n_points]
        for i in range(self.n_way+1):
            cur_idx = labeled_idx[np.where(labels[labeled_idx] ==i)]
            y = np.zeros((N,))
            if cur_idx.shape[0] > 0:
                y[cur_idx] = 1.0 / cur_idx.shape[0]
            f, _ = scipy.sparse.linalg.cg(A, y, tol=1e-6, maxiter=5)
            Z[:,i] = f

        # Handle numberical errors
        Z[Z < 0] = 0 
        pred = F.normalize(torch.tensor(Z).cuda(),1)
        pred[pred <0] = 0
        pred = pred.view(size_s + size_q, -1, self.n_way+1)#.to_dense()
        
        predq = pred[size_s: , :, :]
        preds = pred[:size_s, :, :]

        ce = nn.CrossEntropyLoss().cuda()

        gt = torch.cat((support_y.view(self.n_way* self.k_shot, -1), query_y), 0)

        loss_manifold = ce(pred.view(-1, self.n_way+1), gt.view(-1))

        loss = self.computeCrossEntropyLoss(query_pred, query_y) + loss_manifold
        
        query_pred = (predq.permute(0, 2, 1) + query_pred)/2
        
        #else:
        #    loss = self.computeCrossEntropyLoss(query_pred, query_y)
        
        return query_pred, loss
    
    
    # Note: we do not need it. We apply manifold regularizer both in train and test.
    
    #def forward_test(self, support_x, support_y, query_x, query_y) -> 'query_pred':

        # support_x: [n_way, k_shot, in_channels, num_points]
        # support_y: [n_way, k_shot, num_points]
        # query_x: [n_queries, in_channels, num_points]
        # query_y: [n_queries, num_points] 
        
        # ***************** Prototypical Networks ********************
        
        #support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)
        #support_feat = self.getFeatures(support_x)
        #support_feat = support_feat.view(self.n_way, self.k_shot, -1, self.n_points)
        #query_feat = self.getFeatures(query_x) #(n_queries, feat_dim, num_points)

        #fg_mask = support_y
        #bg_mask = torch.logical_not(support_y)

        #support_fg_feat = self.getMaskedFeatures(support_feat, fg_mask)
        #suppoer_bg_feat = self.getMaskedFeatures(support_feat, bg_mask)
        
        # prototype learning
        
        #fg_prototypes, bg_prototype = self.getPrototype(support_fg_feat, suppoer_bg_feat)
        #prototypes = [bg_prototype] + fg_prototypes

        # non-parametric metric learning
        
        #similarity = [self.calculateSimilarity(query_feat, prototype, self.dist_method) for prototype in prototypes]

        #query_pred = torch.stack(similarity, dim=1) #(n_queries, n_way+1, num_points)
        
        #loss = self.computeCrossEntropyLoss(query_pred, query_y)
        
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

