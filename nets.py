import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch_geometric.transforms as T
import torch_cluster
from torch_geometric.nn import NNConv, GCNConv, GraphConv
from torch_geometric.nn import PointConv, EdgeConv, SplineConv
from hgcn.layers.layers import FermiDiracDecoder


class EmulsionConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.mp = torch.nn.Linear(in_channels * 2, out_channels)

    def forward(self, x, edge_index, orders):
        for order in orders:
            x = self.propagate(torch.index_select(edge_index[:, order],
                                                  0,
                                                  torch.LongTensor([1, 0]).to(x.device)), x=x)
        return x

    def message(self, x_j, x_i):
        return self.mp(torch.cat([x_i, x_j - x_i], dim=1))

    def update(self, aggr_out, x):
        return aggr_out + x


class GraphNN_KNN_v1(nn.Module):
    def __init__(self, input_dim, output_dim=10):
        super().__init__()
        self.input_dim = input_dim
        self.emconv = EmulsionConv(self.input_dim, self.input_dim)
        self.wconv1 = EdgeConv(Sequential(nn.Linear(20, 10)), 'max')
        self.wconv2 = EdgeConv(Sequential(nn.Linear(20, 10)), 'max')
        self.wconv3 = EdgeConv(Sequential(nn.Linear(20, 10)), 'max')
        self.output = nn.Linear(10, output_dim)

    def forward(self, data):
        x, edge_index, orders = data.x, data.edge_index, data.orders
        x = self.emconv(x=x, edge_index=edge_index, orders=orders)
        x1 = self.wconv1(x=x, edge_index=edge_index)
        x2 = self.wconv2(x=x1, edge_index=edge_index)
        x3 = self.wconv3(x=x2, edge_index=edge_index)
        return self.output(x3)


class EdgeClassifier_v1(nn.Module):
    def __init__(self, output_dim, **kwargs):
        super().__init__()
        self._layers = nn.ModuleList([
            nn.Linear(output_dim * 2, 144),
            nn.Tanh(),
            nn.Linear(144, 144),
            nn.Tanh(),
            nn.Linear(144, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        ])

    def forward(self, shower, embeddings, edge_index):
        embeddings = torch.cat([
            embeddings[edge_index[0]],
            embeddings[edge_index[1]]
        ], dim=1)
        for layer in self._layers:
            embeddings = layer(embeddings)
        return embeddings


class DiracClassifier(nn.Module):
    def __init__(self, manifold, c=1., r=0., t=1., **kwargs):
        super().__init__()
        self.manifold = manifold
        self.c = c
        self.dc = FermiDiracDecoder(r=r, t=t)

    def forward(self, shower, embeddings, edge_index):
        emb_in = embeddings[edge_index[0], :]
        emb_out = embeddings[edge_index[1], :]
        sqdist = self.manifold.sqdist(emb_in, emb_out, self.c)
        probs = self.dc.forward(sqdist)
        return probs


class EdgeDenseClassifier(nn.Module):
    def __init__(self, input_dim=10, **kwargs):
        super(EdgeDenseCl, self).__init__()
        self.edge_classifier = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, embeddings, edge_index):
        return self.edge_classifier(torch.cat([embeddings[edge_index[0]], embeddings[edge_index[0]]], 1))


class EdgeDenseClassifierEdgeAttribute(nn.Module):
    def __init__(self, input_dim=10, **kwargs):
        super(EdgeDenseClassifierEdgeAttribute, self).__init__()
        self.edge_classifier = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, shower, embeddings, edge_index):
        x = torch.cat([embeddings[edge_index[0]], embeddings[edge_index[0]]], 1)
        x = torch.cat([x, shower.edge_attr], 1)
        return self.edge_classifier(x)
