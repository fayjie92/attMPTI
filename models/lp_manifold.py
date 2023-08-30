"""
Transductive Few-Shot Segmentation with Manifold learning for point cloud.
Author: Abdur R. Fayjie & Umamaheswaran Raman Kumar, 2023 
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP,DynamicEdgeConv
#from torch_cluster import knn_graph
#import faiss
#from faiss import normalize_L2
#import scipy

from models.dgcnn import DGCNN
from models.attention import SelfAttention


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn import knn_graph

class LabelPropagation(MessagePassing):
    def __init__(self):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))


        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        #print(norm)
        #print(x_j)
        #breakpoint()
        return norm.view(-1, 1) * x_j


class Manifold(LabelPropagation):
    def __init__(self, n_way=2, k=20):
        super().__init__()
        self.n_way = n_way
        self.k = k
        #self.mlp = MLP(in_channels=192, hidden_channels=100, out_channels=50, num_layers=3)
        self.linear = nn.Linear(192, 50)

    def forward(self, x, batch=None):
        shape = x.shape[1] - (self.n_way+1)
        x1 = x[:,:shape]
        #x1 = self.mlp(x1)
        x1 = self.linear(x1)
        x2 = x[:,shape:]
        edge_index = knn_graph(x1, self.k, batch, loop=False, flow=self.flow)
        return super().forward(x2, edge_index)


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


class LPManifold(nn.Module):
    def __init__(self, args):
        super(LPManifold, self).__init__()
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.dist_method = args.dist_method
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.use_attention = args.use_attention
        self.alpha = torch.tensor(0.99, requires_grad=False)
        self.sigma = 0.25
        self.knn = 20 # treat it as k for knn in manifold

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)

        if self.use_attention:
            self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)
        else:
            self.linear_mapper = nn.Conv1d(args.dgcnn_mlp_widths[-1], args.output_dim, 1, bias=False)
        '''self.mlp1 = nn.Sequential(
            MLP(2*197, 100, 100, 3, norm="batchnorm"),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
        )
        self.mlp2 = nn.Sequential(
            MLP(2 * 100, 50, 2, 1, norm="batchnorm"),
            nn.ReLU(),
            nn.BatchNorm1d(2 * hidden_dim),
        )'''
        #self.mlp1 = MLP(in_channels=2*(192+self.n_way+1), hidden_channels=100, out_channels=50, num_layers=3)
        #self.mlp2 = MLP(in_channels=2*50, hidden_channels=10, out_channels=self.n_way+1, num_layers=3)
        #self.label_prop1 = DynamicEdgeConv(self.mlp1, k=20, aggr='max')
        #self.label_prop2 = DynamicEdgeConv(self.mlp2, k=20, aggr='max')
        self.manifold = Manifold(self.n_way, k=75)
        

    
    def forward(self, support_x, support_y, query_x, query_y):
        """
        Args:
            support_x:  n_way x k_shot x in_channels x num_points  [support point cloud]
            support_y:  Fy x k_shot x num_points [support labels]
            query_x:    n_queries x in_channels x num_points [query point cloud]
            query_y:    n_queries x num_points [query labels, each point \in {0,..., n_way}]
        Return:
            query_pred:   [predicted query point cloud]
        """
        # Manifold with label propagation
        # Query is unlabelled, support is labelled
        # Shares all queries in test time -> Transductive

        # Step1: Embedding
        # support_x -> [n_way, k_shot, 3, N] 
        # query_x   -> [n_way*num_query, 3, N]    
        # support_y -> [n_way, k_shot, N] 
        # query_y   -> [n_way*num_query, N] 
        
        #import pdb; pdb.set_trace()

        #[1, 3, 128]
        support_x = support_x.view(self.n_way*self.k_shot, self.in_channels, self.n_points)   
        #1, 5
        size_s, size_q = support_x.shape[0], query_x.shape[0]
        #[1, 192, N]
        support_feat = self.getFeatures(support_x)
        #[5, 192, N]
        query_feat = self.getFeatures(query_x)
        
        #[N, 192]
        emb_s = support_feat.permute(0,2,1)
        emb_s = emb_s.reshape(emb_s.shape[0] * emb_s.shape[1], -1) 
        #support_x = support_x.permute(0,2,1)
        #support_x = support_x.view(support_x.shape[0] * support_x.shape[1], -1) 
        y_s = support_y.permute(0,2,1).view(-1)
        y_s = F.one_hot(y_s.long(), self.n_way+1)
        
        #[5N, 192]
        emb_q = query_feat.permute(0,2,1)
        emb_q = emb_q.reshape(emb_q.shape[0] * emb_q.shape[1], -1)
        #query_x = query_x.permute(0,2,1)
        #query_x = query_x.view(query_x.shape[0] * query_x.shape[1], -1)
        #y_q = torch.mul(torch.ones(emb_q.shape[0], self.n_way+1), 1.0/(self.n_way+1)).cuda()
        y_q = torch.zeros(emb_q.shape[0], self.n_way+1).cuda()

        #[N+(5*N), 192] -> [T, 192] -> T = N+(5*N)
        emb_all = torch.cat((torch.cat((emb_s,y_s),1), torch.cat((emb_q,y_q),1)), 0)
        #N, dim=192
        T, dim = emb_all.shape[0], emb_all.shape[1]

        #emb_inter = self.label_prop1(emb_all)
        #pred = self.label_prop2(emb_inter)
        #manifold = Manifold(self.n_way, k=50)
        pred = self.manifold(emb_all)
        #print(emb_all)
        #print(pred)
        #breakpoint()
        pred[pred <0] = 0
        pred = pred.view(size_s + size_q, -1, self.n_way+1)#.to_dense()
        
        predq = pred[size_s: , :, :]
       
        loss = self.computeCrossEntropyLoss(predq.view(-1, self.n_way+1), query_y.view(-1))
        query_pred = predq.permute(0, 2, 1)
        
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
