from math import ceil

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import DenseSAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

from torch_geometric.nn import DenseGraphConv

class SAGEConvolutions(nn.Module):
    def __init__(self, num_layers,
                 in_channels,
                 out_channels,
                 residual=True):
        super().__init__()

        self.num_layers = num_layers
        self.residual = residual
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(num_layers-1):
            if i == 0:
                self.layers.append(DenseSAGEConv(in_channels, out_channels, normalize=True))
            else:
                self.layers.append(DenseSAGEConv(out_channels, out_channels, normalize=True))
            self.bns.append(nn.BatchNorm1d(out_channels))

        if num_layers == 1:
            self.layers.append(DenseSAGEConv(in_channels, out_channels, normalize=True))
        else:
            self.layers.append(DenseSAGEConv(out_channels, out_channels, normalize=True))

    def forward(self, x, adj, mask=None):
        for i in range(self.num_layers - 1):
            x_new = F.relu(self.layers[i](x, adj, mask))
            batch_size, num_nodes, num_channels = x_new.size()
            x_new = x_new.view(-1, x_new.shape[-1])
            x_new = self.bns[i](x_new)
            x_new = x_new.view(batch_size, num_nodes, num_channels)
            if self.residual and x.shape == x_new.shape:
                x = x + x_new
            else:
                x = x_new
        x = self.layers[self.num_layers-1](x, adj, mask)
        return x


class DiffPoolLayer(nn.Module):

    def __init__(self, dim_input, dim_embedding, current_num_clusters,
                 no_new_clusters):

        super().__init__()

        self.gnn_pool = SAGEConvolutions(1, dim_input, no_new_clusters)
        self.gnn_embed = SAGEConvolutions(1, dim_input, dim_embedding)

    def forward(self, x, adj, mask=None):

        s = self.gnn_pool(x, adj, mask)
        x = self.gnn_embed(x, adj, mask)

        x, adj, l, e = dense_diff_pool(x, adj, s, mask)
        return x, adj, l, e


class DiffPool(nn.Module):

    def __init__(self, num_features, num_classes, max_num_nodes, num_layers, gnn_hidden_dim,
                 gnn_output_dim, args, encode_edge=False,
                 pre_sum_aggr=False):
        super().__init__()
        self.args = args

        # gnn_hidden_dim equals gnn_output_dim
        self.encode_edge = encode_edge
        self.max_num_nodes = max_num_nodes
        self.pooling_type = args.pooling_type
        self.num_pooling_layers = num_layers

        gnn_dim_input = num_features

        # Reproduce paper choice about coarse factor
        coarse_factor = 0.1 if num_layers == 1 else 0.25

        if pre_sum_aggr:  # this is only used for IMDB
            self.initial_embed = DenseGraphConv(gnn_dim_input, gnn_output_dim)
        else:
            self.initial_embed = SAGEConvolutions(1, gnn_dim_input, gnn_output_dim)

        no_new_clusters = ceil(coarse_factor * self.max_num_nodes)

        layers = []
        after_pool_layers = []
        current_num_clusters = self.max_num_nodes
        for i in range(num_layers):
            layer_input_dim = num_features if i == 0 else gnn_hidden_dim
            layer_output_dim = gnn_output_dim if i == num_layers-1 else gnn_hidden_dim
            diffpool_layer = DiffPoolLayer(layer_input_dim, layer_output_dim, current_num_clusters,
                                           no_new_clusters)
            layers.append(diffpool_layer)

            # Update embedding sizes
            current_num_clusters = no_new_clusters
            no_new_clusters = ceil(no_new_clusters * coarse_factor)

            after_pool_layers.append(SAGEConvolutions(self.args.after_pooling_layer, layer_output_dim, layer_output_dim))

        self.diffpool_layers = nn.ModuleList(layers)
        self.after_pool_layers = nn.ModuleList(after_pool_layers)

        # After DiffPool layers, apply again layers of GraphSAGE convolutions
        final_embed_dim_output = gnn_output_dim

    def forward(self, x, adj, mask=None):
        x_all, l_total, e_total = [], 0, 0
        for i in range(self.num_pooling_layers):
            if i != 0:
               mask = None

            x, adj, l, e = self.diffpool_layers[i](x, adj, mask)  # x has shape (batch, MAX_no_nodes, feature_size)

            x = self.after_pool_layers[i](x, adj)

            l_total += l
            e_total += e

        #x = torch.max(x, dim=1)[0]

        #x = F.relu(self.lin1(x))
        #x = self.lin2(x)
        return x, l_total, e_total