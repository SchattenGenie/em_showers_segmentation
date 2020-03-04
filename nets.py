import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import Sequential
import torch_geometric
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, scatter_
import torch_geometric.transforms as T
import torch_cluster
from torch_geometric.nn import NNConv, GCNConv, GraphConv
from torch_geometric.nn import PointConv, EdgeConv, SplineConv
from torch.utils.checkpoint import checkpoint
import numpy as np


def extract_subgraph(h, adj, edge_attr, order):
    adj_selected = adj[:, order]
    edge_attr_selected = edge_attr[order, :]
    nodes_selected = adj_selected.unique()
    h_selected = h[nodes_selected]
    nodes_selected_new = torch.arange(len(nodes_selected))
    dictionary = dict(zip(nodes_selected.cpu().numpy(), nodes_selected_new.cpu().numpy()))
    adj_selected_new = torch.tensor(np.vectorize(dictionary.get)(adj_selected.cpu().numpy())).long().to(adj)
    return h_selected, nodes_selected, edge_attr_selected, adj_selected_new


class EmulsionConv(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=1, direction=0):
        super().__init__(aggr='add')
        self.mp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),
            nn.ReLU()
        )
        self._direction = direction  # TODO: defines direction

    def forward(self, x, edge_index, orders, edge_attr, orders_preprocessed):
        x = x.clone()
        for i, order in enumerate(orders):
            if order.sum():
                # print(i, len(orders_preprocessed), orders_preprocessed[i])
                nodes_selected, adj_selected_new = orders_preprocessed[i]
                nodes_selected = nodes_selected.to(x).long()
                adj_selected_new = adj_selected_new.to(x).long()
                x_selected = x[nodes_selected]
                edge_attr_selected = edge_attr[order, :]
                x_selected = self.message(
                    x_j=x_selected[adj_selected_new[0]],
                    x_i=x_selected[adj_selected_new[1]],
                    edge_attr=edge_attr_selected
                )
                x_selected = scatter_('add', x_selected, adj_selected_new[self._direction],
                                      dim=0, dim_size=len(nodes_selected))
                x[nodes_selected] = (x[nodes_selected] + x_selected) / 2.
        return x

    def message(self, x_j, x_i, edge_attr):
        return self.mp(torch.cat([x_i, x_j - x_i, edge_attr.view(len(x_i), -1)], dim=1))

    def update(self, aggr_out, x):
        return aggr_out + x


class EmulsionConvSlow(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_dim=1, direction=0):
        super().__init__(aggr='add')
        self.mp = nn.Sequential(
            nn.Linear(in_channels * 2 + edge_dim, out_channels),
            nn.ReLU()
        )
        self._direction = direction  # TODO: defines direction

    def forward(self, x, edge_index, orders, edge_attr):
        x = x.clone()
        for order in orders:
            if order.sum():
                x_selected, nodes_selected, edge_attr_selected, adj_selected_new = extract_subgraph(
                    h=x,
                    adj=edge_index,
                    edge_attr=edge_attr,
                    order=order
                )
                x_selected = self.message(
                    x_j=x_selected[adj_selected_new[0]],
                    x_i=x_selected[adj_selected_new[1]],
                    edge_attr=edge_attr_selected
                )
                x_selected = scatter_('add', x_selected, adj_selected_new[self._direction],
                                      dim=0, dim_size=len(nodes_selected))
                x[nodes_selected] = (x[nodes_selected] + x_selected) / 2.
        return x

    def message(self, x_j, x_i, edge_attr):
        return self.mp(torch.cat([x_i, x_j - x_i, edge_attr.view(len(x_i), -1)], dim=1))

    def update(self, aggr_out, x):
        return aggr_out + x


class EmulsionConvOldSlow(MessagePassing):
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


def init_bias_model(model, b: float):
    for module in model.modules():
        if hasattr(module, 'bias'):
            module.bias.data.fill_(b)


class GraphNN_KNN_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim=10, bias_init=0., **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.linear1 = nn.Sequential(nn.Linear(self.input_dim, self.hidden_dim), nn.ReLU())
        self.emconv1 = EmulsionConv(self.hidden_dim, self.hidden_dim)
        self.linear2 = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU())
        self.emconv2 = EmulsionConv(self.hidden_dim, self.hidden_dim)
        self.wconv1 = EdgeConv(Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ReLU()), 'max')
        self.wconv2 = EdgeConv(Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ReLU()), 'max')
        self.wconv3 = EdgeConv(Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ReLU()), 'max')
        self.wconv4 = EdgeConv(Sequential(nn.Linear(2 * self.hidden_dim, self.hidden_dim), nn.ReLU()), 'max')
        self.output = nn.Linear(self.hidden_dim, output_dim)
        init_bias_model(self, b=0.)

    def forward(self, data):
        x, edge_index, orders, edge_attr = data.x, data.edge_index, data.orders, data.edge_attr
        orders_preprocessed = data.orders_preprocessed[0]
        print("orders preprocessed", len(orders_preprocessed), len(orders))

        x = self.linear1(x)
        x = self.emconv1(x=x, edge_index=edge_index, orders=orders, edge_attr=edge_attr, orders_preprocessed=orders_preprocessed)
        # x = x + x_new
        x = self.linear2(x)
        x = self.emconv2(x=x, edge_index=edge_index, orders=orders, edge_attr=edge_attr, orders_preprocessed=orders_preprocessed)
        # x = x + x_new
        # x = checkpoint(self.emconv, x=x, edge_index=edge_index, orders=orders)
        x = self.wconv1(x=x, edge_index=edge_index)
        # x = x + x_new
        # x = checkpoint(self.emconv, x=x, edge_index=edge_index, orders=orders)
        x = self.wconv2(x=x, edge_index=edge_index)
        # x = x + x_new
        # x = checkpoint(self.emconv, x=x, edge_index=edge_index, orders=orders)
        x = self.wconv3(x=x, edge_index=edge_index)
        # x = x + x_new
        x = self.wconv4(x=x, edge_index=edge_index)
        return self.output(x)


class EdgeClassifier_v1(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, prior_proba=1. - 0.025, **kwargs):
        super().__init__()
        self._layers = nn.ModuleList([
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        ])
        init_bias_model(self, b=0.)
        init_bias_model(self._layers[-2], b=-np.log((1 - prior_proba) / prior_proba))

    def forward(self, shower, embeddings, edge_index):
        embeddings = torch.cat([
            embeddings[edge_index[0]],
            embeddings[edge_index[1]],
            shower.edge_attr
        ], dim=1)
        for layer in self._layers:
            embeddings = layer(embeddings)
            # x = checkpoint(layer, embeddings)
        return embeddings


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
        x = torch.cat([embeddings[edge_index[0]], embeddings[edge_index[1]]], 1)
        x = torch.cat([x, shower.edge_attr], 1)
        return self.edge_classifier(x)
