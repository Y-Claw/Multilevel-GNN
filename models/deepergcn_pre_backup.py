import torch
from .gcn_lib.sparse.torch_vertex import GENConv
from .gcn_lib.sparse.torch_nn import norm_layer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, TopKPooling
import logging
import math

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args
    def forward(self, x):
        return x.view((x.size(0),)+self.shape)

class DeeperGCN(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN, self).__init__()

        self.num_layers = args.num_layers
        self.dropout = args.dropout
        self.block = args.block

        self.mul_attr = args.mul_attr
        hidden_channels = args.hidden_channels
        self.hidden_channels = hidden_channels
        num_tasks = args.num_tasks
        conv = args.conv
        aggr = args.gcn_aggr
        t = args.t
        self.learn_t = args.learn_t
        p = args.p
        self.learn_p = args.learn_p
        self.msg_norm = args.msg_norm
        learn_msg_scale = args.learn_msg_scale

        conv_encode_edge = args.conv_encode_edge
        norm = args.norm
        mlp_layers = args.mlp_layers
        
        graph_pooling = args.graph_pooling

        print('The number of layers {}'.format(self.num_layers),
              'Aggr aggregation method {}'.format(aggr),
              'block: {}'.format(self.block))
        if self.block == 'res+':
            print('LN/BN->ReLU->GraphConv->Res')
        elif self.block == 'res':
            print('GraphConv->LN/BN->ReLU->Res')
        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')
        elif self.block == "plain":
            print('GraphConv->LN/BN->ReLU')
        else:
            raise Exception('Unknown block Type')

        self.pca_only = args.pca_only
        self.gnn_encoder = args.gnn_encoder
        self.no_inter_drop = args.no_inter_drop
        self.no_inter_norm = args.no_inter_norm
        self.gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              gnn_encoder=self.gnn_encoder,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=conv_encode_edge, edge_feat_dim=hidden_channels,
                              norm=norm, mlp_layers=mlp_layers, pca_only=self.pca_only)
            else:
                raise Exception('Unknown Conv Type')
            self.gcns.append(gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.node_embedding = args.node_embedding
        if self.node_embedding:
            self.node_embedding_encoder = torch.nn.Embedding(args.node_num, args.node_embedding_dim)
        input_dim = 3 + (args.node_embedding_dim if (self.node_embedding and args.gnn_encoder == 'linear')  else 0) + (2 if self.mul_attr else 0)
        
        if args.gnn_encoder == 'linear':
            self.node_features_encoder = torch.nn.Linear(input_dim, hidden_channels)
            self.edge_encoder = torch.nn.Linear(7 if args.use_column is None else 1, hidden_channels)
        elif args.gnn_encoder == 'conv1x1':
            self.node_features_encoder = torch.nn.Sequential(
                torch.nn.Conv1d(1, hidden_channels*2, 1),
                #torch.nn.ReLU()
            )
            self.edge_encoder = torch.nn.Sequential(
                torch.nn.Linear(7 if args.use_column is None else 1, input_dim*2),
                Reshape(1, input_dim*2),
                torch.nn.Conv1d(1, hidden_channels, 1)
            )

        self.use_edge_attr = args.use_edge_attr
        self.pathway_global_node = args.pathway_global_node
        if self.pathway_global_node:
            self.pathway_num = args.pathway_num
            if args.gnn_encoder == 'linear':
                self.pathway_features_encoder = torch.nn.Linear(6, hidden_channels)
            elif args.gnn_encoder == 'conv1x1':
                self.pathway_features_encoder = torch.nn.Sequential(
                    Reshape(1, -1),
                    torch.nn.Conv1d(1, hidden_channels, 1),
                    torch.nn.ReLU()
                )
            #self.pathway_encoder = torch.nn.Embedding(args.pathway_num, hidden_channels)

        self.num_layer_head = args.num_layer_head
        self.pathway_readout = args.pathway_readout
        if self.pathway_readout is None:
            pass
        elif self.pathway_readout == 'MSA':
            self.pred_norm = nn.BatchNorm1d(self.pathway_num)
            self.readout_func = nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=8, batch_first=True)
            """self.readout_func = nn.Sequential(
                nn.Linear(self.pathway_num, hidden_channels),
                nn.TransformerEncoderLayer(d_model=hidden_channels, nhead=8, batch_first=True)
            )"""
        elif self.pathway_readout == 'maxpool':
            readout_in = (self.pathway_num // 4) * hidden_channels * (3 if args.gnn_encoder == 'conv1x1' else 1)
            self.readout_func = nn.Linear(readout_in, hidden_channels)

        if graph_pooling == "sum":
                self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception('Unknown Pool Type')
        
        self.graph_pred_linear = torch.nn.Sequential()
        self.use_age = args.use_age

        head_embedding = (hidden_channels + 1) if args.use_age else hidden_channels

        for i in range(args.num_layer_head - 1):
            self.graph_pred_linear.add_module(str(2*i), torch.nn.Linear(head_embedding, head_embedding))
            self.graph_pred_linear.add_module(str(2*i+1), torch.nn.ReLU())
            if args.head_dropout:
                self.graph_pred_linear.add_module("drop{}".format(str(i)), torch.nn.Dropout(self.dropout))
        self.graph_pred_linear.add_module(str(2*args.num_layer_head), torch.nn.Linear(head_embedding, num_tasks))

        if args.all_init:
            self.init_weight()
        elif args.head_init:
            for m in self.graph_pred_linear.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight.data)
                    torch.nn.init.constant_(m.bias.data, 0.0)
        



    def forward(self, input_batch):
        x = input_batch.x
        edge_index = input_batch.edge_index
        #816136, 128
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch
        age = input_batch.age
        pathway_node_feature = input_batch.pathway_node_attr
        if not self.pca_only:
            if self.node_embedding:
                if self.gnn_encoder == 'linear':
                    h = self.node_features_encoder(torch.cat([x[:, :-1], self.node_embedding_encoder(x[:, -1].to(torch.long))], dim=-1))
                elif self.gnn_encoder == 'conv1x1':
                    h = self.node_features_encoder((x[:, :-1] + self.node_embedding_encoder(x[:, -1].to(torch.long)))[:, None, :])
                    b, c, n = h.shape
                    h = h.reshape(b, c//2, -1)
            else:
                if self.gnn_encoder == 'linear':
                    h = self.node_features_encoder(x)
                elif self.gnn_encoder == 'conv1x1':
                    h = self.node_features_encoder(x[:, None, :])
                    b, c, n = h.shape
                    h = h.reshape(b, c//2, -1)

            if self.use_edge_attr:
                edge_emb = self.edge_encoder(edge_attr)
            else:
                edge_emb = None
        
        if self.pathway_global_node:
            #pathway_embedding = self.pathway_encoder(input_batch.pathway_global_nodes)
            pathway_embedding = self.pathway_features_encoder(pathway_node_feature)
            if not self.pca_only:
                for i, node_size in enumerate(torch.cumsum(input_batch.node_size, dim=0).detach().cpu().numpy()):
                    #h[node_size-self.pathway_num:node_size] = torch.zeros(self.pathway_num, self.hidden_channels)
                    h[node_size-self.pathway_num:node_size] = pathway_embedding[i*self.pathway_num:(i+1)*self.pathway_num]
        
        if self.gnn_encoder == 'conv1x1':
            if not self.pca_only:
                h = F.relu(h)
            else:
                h = F.relu(pathway_embedding)
                edge_emb = None

        if self.block == 'res+':
            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                if not self.no_inter_norm:
                    h1 = self.norms[layer - 1](h)
                else:
                    h1 = h
                h2 = F.relu(h1)
                if not self.no_inter_drop:
                    h2 = F.dropout(h2, p=self.dropout, training=self.training)
                h = self.gcns[layer](h2, edge_index, edge_emb) + h

            h = self.norms[self.num_layers - 1](h)
            if not self.no_inter_drop:
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'res':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                h2 = self.norms[layer](h1)
                h = F.relu(h2) + h
                h = F.dropout(h, p=self.dropout, training=self.training)

        elif self.block == 'dense':
            raise NotImplementedError('To be implemented')

        elif self.block == 'plain':

            h = F.relu(self.norms[0](self.gcns[0](h, edge_index, edge_emb)))
            h = F.dropout(h, p=self.dropout, training=self.training)

            for layer in range(1, self.num_layers):
                h1 = self.gcns[layer](h, edge_index, edge_emb)
                if not self.no_inter_norm:
                    h2 = self.norms[layer](h1)
                else:
                    h2 = h1
                if layer != (self.num_layers - 1) or self.pca_only:
                    h = F.relu(h2)
                else:
                    h = h2
                if not self.no_inter_drop:
                    h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')

        if self.pathway_global_node:
            pathway_results = []
            pathway_batch = []
            if self.pca_only:
                for i, node_size in enumerate(torch.cumsum(input_batch.node_size, dim=0).detach().cpu().numpy()):
                    pathway_results.append(h[i*self.pathway_num:(i+1)*self.pathway_num])
            else:
                for i, node_size in enumerate(torch.cumsum(input_batch.node_size, dim=0).detach().cpu().numpy()):
                    pathway_results.append(h[node_size-self.pathway_num:node_size])
                    pathway_batch.append(batch[node_size-self.pathway_num:node_size])
                    #pathway_batch.append(batch[node_size-self.hidden_channels:node_size])
            if self.pathway_readout is None:
                h_graph = self.pool(torch.cat(pathway_results), torch.cat(pathway_batch))
            elif self.pathway_readout == 'MSA':
                if not self.training:
                    #torch.cosine_similarity(pathway_results[:,0,:], pathway_results[0,0,:])
                    #torch.cosine_similarity(h[5606:5606+142], h[5606+5748:5606+5748+142], dim=0)
                    import pdb
                    #pdb.set_trace()
                pathway_results = self.pred_norm(torch.stack(pathway_results))
                #pathway_results = torch.stack(pathway_results)
                pathway_results = self.readout_func(pathway_results)
                h_graph = self.pool(torch.flatten(pathway_results, end_dim=1), torch.cat(pathway_batch))
            elif self.pathway_readout == 'maxpool':
                pathway_results = torch.stack(pathway_results)
                if self.gnn_encoder == 'linear':
                    h_graph = torch.flatten(F.max_pool1d(pathway_results.transpose(1,2), 4), start_dim=1)
                    h_graph = self.readout_func(h_graph)
                elif self.gnn_encoder == 'conv1x1':
                    h_graph = torch.flatten(F.max_pool2d(pathway_results.transpose(1,2), (4, 2)), start_dim=1)
                    h_graph = self.readout_func(h_graph)
        else:
            h_graph = self.pool(h, batch)
        
        if self.use_age:
            h_graph = torch.cat([h_graph, age[:, None]], dim=-1)
        return F.softmax(self.graph_pred_linear(h_graph), dim=-1)

    def print_params(self, epoch=None, final=False):
        if self.learn_t:
            ts = []
            for gcn in self.gcns:
                ts.append(gcn.t.item())
            if final:
                print('Final t {}'.format(ts))
            else:
                logging.info('Epoch {}, t {}'.format(epoch, ts))
        if self.learn_p:
            ps = []
            for gcn in self.gcns:
                ps.append(gcn.p.item())
            if final:
                print('Final p {}'.format(ps))
            else:
                logging.info('Epoch {}, p {}'.format(epoch, ps))
        if self.msg_norm:
            ss = []
            for gcn in self.gcns:
                ss.append(gcn.msg_norm.msg_scale.item())
            if final:
                print('Final s {}'.format(ss))
            else:
                logging.info('Epoch {}, s {}'.format(epoch, ss))
    
    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
    