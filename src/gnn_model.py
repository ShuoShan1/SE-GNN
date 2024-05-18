import torch.nn as nn
import torch.nn.functional as F
from gcn_layer import NR_GraphAttention
from tabulate import tabulate
from utils import *
import torch

class Encoder_Model(nn.Module):
    def __init__(self, node_hidden, rel_hidden,triple_size, node_size, rel_size,device,
                 adj_matrix, r_index, r_val, rel_matrix, ent_matrix,ill_ent,high_adj,dropout_rate=0.0,
                gamma=3, lr=0.005, depth=2):
        super(Encoder_Model, self).__init__()
        self.node_hidden = node_hidden
        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.depth = depth
        self.device = device
        self.gamma = gamma
        self.lr = lr
        self.dropout = nn.Dropout(dropout_rate)
        self.adj_list = adj_matrix.to(device)
        self.r_index = r_index.to(device)
        self.r_val = r_val.to(device)
        self.rel_adj = rel_matrix.to(device)
        self.ent_adj = ent_matrix.to(device)
        self.ill_ent = ill_ent
        self.high_adj = high_adj.to(device)

        self.ent_embedding = nn.Embedding(node_size, node_hidden)
        self.rel_embedding = nn.Embedding(rel_size, rel_hidden)

        torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
        torch.nn.init.xavier_uniform_(self.rel_embedding.weight)


        self.e_encoder = NR_GraphAttention(node_size=self.node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )
        self.r_encoder = NR_GraphAttention(node_size=self.node_size,
                                           rel_size=self.rel_size,
                                           triple_size=self.triple_size,
                                           node_dim=self.node_hidden,
                                           depth=self.depth,
                                           use_bias=True
                                           )


    def avg(self, adj, emb, size: int):
        adj = torch.sparse_coo_tensor(indices=adj, values=torch.ones_like(adj[0, :], dtype=torch.float),
                                      size=[self.node_size, size])
        adj = torch.sparse.softmax(adj, dim=1)
        return torch.sparse.mm(adj, emb)

    #
    #
    # def weight_gcn(self,adj,emb):
    #     return torch.sparse.mm(adj, emb)

    def gcn_forward(self):
        ent_feature = self.avg(self.ent_adj, self.ent_embedding.weight,self.node_size)
        rel_feature = self.avg(self.rel_adj, self.rel_embedding.weight, self.rel_size)
        opt = [self.rel_embedding.weight, self.adj_list, self.r_index, self.r_val,self.high_adj]
        out_feature = torch.cat([self.e_encoder([ent_feature] + opt),self.r_encoder([rel_feature] + opt)], dim=-1)
        out_feature = self.dropout(out_feature)
        return out_feature

    def forward(self, train_paris:torch.Tensor, flag):
        if flag:
            out_feature = self.gcn_forward()
            loss = self.align_loss(train_paris, out_feature)
        else:
            torch.nn.init.xavier_uniform_(self.ent_embedding.weight)
            torch.nn.init.xavier_uniform_(self.rel_embedding.weight)
            out_feature = self.gcn_forward()
            loss = self.align_loss(train_paris, out_feature)
        return loss

    def align_loss(self, pairs, emb):
        def squared_dist(A, B):
            row_norms_A = torch.sum(torch.square(A), dim=1)
            row_norms_A = torch.reshape(row_norms_A, [-1, 1])
            row_norms_B = torch.sum(torch.square(B), dim=1)
            row_norms_B = torch.reshape(row_norms_B, [1, -1])
            return row_norms_A + row_norms_B - 2 * torch.matmul(A, B.t())

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]

        pos_dis = torch.sum(torch.square(l_emb - r_emb), dim=-1, keepdim=True)
        l_neg_dis = squared_dist(l_emb, emb)
        r_neg_dis = squared_dist(r_emb, emb)

        del l_emb, r_emb

        l_loss = pos_dis - l_neg_dis + self.gamma
        l_loss = l_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))
        r_loss = pos_dis - r_neg_dis + self.gamma
        r_loss = r_loss * (1 - F.one_hot(l, num_classes=self.node_size) - F.one_hot(r, num_classes=self.node_size))

        del r_neg_dis, l_neg_dis

        r_loss = (r_loss - torch.mean(r_loss, dim=-1, keepdim=True).detach()) / torch.std(r_loss, dim=-1, unbiased=False, keepdim=True).detach()
        l_loss = (l_loss - torch.mean(l_loss, dim=-1, keepdim=True).detach()) / torch.std(l_loss, dim=-1, unbiased=False, keepdim=True).detach()

        lamb, tau = 30, 10
        l_loss = torch.logsumexp(lamb * l_loss + tau, dim=-1)
        r_loss = torch.logsumexp(lamb * r_loss + tau, dim=-1)
        return torch.mean(l_loss + r_loss)

    def loss_no_neg_samples(self, pairs, emb):
        if len(pairs) == 0:
            return 0.0

        l, r = pairs[:, 0].long(), pairs[:, 1].long()
        l_emb, r_emb = emb[l], emb[r]
        loss = torch.sum(torch.square(l_emb - r_emb), dim=-1)
        loss = torch.sum(loss)

        return loss



    def get_embeddings(self, index_a, index_b):
        # forward
        out_feature = self.gcn_forward()
        out_feature = out_feature.cpu()
        # get embeddings
        index_a = torch.Tensor(index_a).long()
        index_b = torch.Tensor(index_b).long()
        Lvec = out_feature[index_a]
        Rvec = out_feature[index_b]
        Lvec = Lvec / (torch.linalg.norm(Lvec, dim=-1, keepdim=True) + 1e-5)
        Rvec = Rvec / (torch.linalg.norm(Rvec, dim=-1, keepdim=True) + 1e-5)
        out_feature = out_feature / (torch.linalg.norm(out_feature, dim=-1, keepdim=True) + 1e-5)
        return Lvec, Rvec,out_feature



    def get_emb(self):
        # forward
        out_feature = self.gcn_forward()
        out_feature = out_feature.cpu()
        # get embeddings
        ill_ent = torch.Tensor(self.ill_ent).long()
        emb = out_feature[ill_ent]
        emb = emb / (torch.linalg.norm(emb, dim=-1, keepdim=True) + 1e-5)
        return emb



