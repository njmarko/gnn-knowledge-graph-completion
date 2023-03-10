import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import RGCNConv


class RGCN(torch.nn.Module):
    """
    Based on R-RGN network from DGL library
    https://docs.dgl.ai/en/0.4.x/tutorials/models/1_gnn/4_rgcn.html#sphx-glr-download-tutorials-models-1-gnn-4-rgcn-py
    """

    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases=-1, num_h_layers=1, num_blocks=5):
        super().__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_h_layers = num_h_layers
        self.num_blocks = num_blocks
        self.layers = nn.ModuleList()
        self.node_emb = Parameter(torch.Tensor(self.num_nodes, self.h_dim))
        self.layers.append(RGCNConv(self.h_dim, self.h_dim, self.num_rels, self.num_bases, self.num_blocks))
        for _ in range(self.num_h_layers):
            self.layers.append(RGCNConv(self.h_dim, self.h_dim, self.num_rels, self.num_bases, num_blocks))
        self.layers.append(
            RGCNConv(self.h_dim, self.out_dim, self.num_rels, self.num_bases))

        self.rel_emb = Parameter(torch.Tensor(self.num_rels, self.h_dim))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        torch.nn.init.xavier_uniform_(self.rel_emb)
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_type = data.x, data.edge_index, data.edge_type

        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index, edge_type)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
        x = self.layers[-1](x, edge_index, edge_type)
        src, dst = x[edge_index[0], x[edge_index[1]]]
        rel = self.rel_emb[edge_type]
        return torch.sum(src * dst * rel, dim=1)
