import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum


class NR_GraphAttention(nn.Module):
    def __init__(self,
                 node_size,
                 rel_size,
                 triple_size,
                 node_dim,
                 depth=2,
                 attn_heads=1,
                 attn_heads_reduction='concat',
                 use_bias=False):
        super(NR_GraphAttention, self).__init__()

        self.node_size = node_size
        self.rel_size = rel_size
        self.triple_size = triple_size
        self.node_dim = node_dim
        self.attn_heads = attn_heads
        self.attn_heads_reduction = attn_heads_reduction
        self.activation = torch.nn.Tanh()
        self.use_bias = use_bias
        self.depth = depth
        self.attn_kernels = nn.ParameterList()
        self.high_atts = nn.ParameterList()
        # create parameters
        feature = self.node_dim*(self.depth+1)

        # gate
        self.gate = torch.nn.Linear(feature, feature)
        torch.nn.init.xavier_uniform_(self.gate.weight)
        torch.nn.init.zeros_(self.gate.bias)

        # proxy node
        # self.proxy = torch.nn.Parameter(data=torch.empty(64, feature, dtype=torch.float32))
        # torch.nn.init.xavier_uniform_(self.proxy)

        # self.weight_matrix = torch.nn.Parameter(torch.empty(300, 300))
        # nn.init.xavier_uniform_(self.weight_matrix)

        # attention kernel
        for l in range(self.depth):
            attn_kernel = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(attn_kernel)
            self.attn_kernels.append(attn_kernel)
        for l in range(self.depth):
            high_att = torch.nn.Parameter(data=torch.empty(self.node_dim, 1, dtype=torch.float32))
            torch.nn.init.xavier_uniform_(high_att)
            self.high_atts.append(high_att)

    def forward(self, inputs):
        outputs = []
        features = inputs[0]
        rel_emb = inputs[1]
        adj = inputs[2]
        r_index = inputs[3]
        r_val = inputs[4]
        high_nei = inputs[5]

        features = self.activation(features)
        outputs.append(features)

        for l in range(self.depth):
            attention_kernel = self.attn_kernels[l]
            tri_rel = torch.sparse_coo_tensor(indices=r_index, values=r_val,size=[self.triple_size, self.rel_size], dtype=torch.float32)
            tri_rel = torch.sparse.mm(tri_rel, rel_emb)

            neighs = features[adj[1, :].long()]

            tri_rel = F.normalize(tri_rel, dim=1, p=2)
            neighs = neighs - 2*torch.sum(neighs*tri_rel, dim=1, keepdim=True)*tri_rel

            att = torch.squeeze(torch.mm(tri_rel, attention_kernel), dim=-1)
            att = torch.sparse_coo_tensor(indices=adj, values=att, size=[self.node_size, self.node_size])
            att = torch.sparse.softmax(att, dim=1)

            new_features = scatter_sum(src=neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=adj[0,:].long())

            features = self.activation(new_features)
            outputs.append(features)

        #从这里开始修改高阶聚合
        for l in range(self.depth):
            high_att = self.high_atts[l]

            neighs = features[high_nei[1, :].long()]
            att = torch.squeeze(torch.mm(neighs, high_att), dim=-1)
            att = torch.sparse_coo_tensor(indices=high_nei, values=att, size=[self.node_size, self.node_size])
            att = torch.sparse.softmax(att, dim=1)

            new_features = scatter_sum(src=neighs * torch.unsqueeze(att.coalesce().values(), dim=-1), dim=0,
                                       index=high_nei[0,:].long())

            features = self.activation(new_features)
            outputs.append(features)

        outputs1 = torch.cat(outputs, dim=-1)







        return outputs1
