import torch
from .gcn_lib.sparse.torch_vertex import GENConv, PathwayConv
from .gcn_lib.sparse.torch_nn import norm_layer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, TopKPooling
import logging
import math


class MultiOmixGCN(torch.nn.Module):

    def __init__(self, args):
        super(MultiOmixGCN, self).__init__()

        self.mrna_encoder = DeeperGCN_Vnode(args)
        self.cnv_encoder = DeeperGCN_Vnode(args)
        self.mt_encoder = DeeperGCN_Vnode(args)
        self.omixs = ['mrna', 'cnv', 'mt']

        self.graph_pred_linear = torch.nn.Sequential()
        self.use_age = args.use_age
        self.dropout = args.dropout
        num_tasks = 2

        hidden_channels = len(self.omixs) * args.hidden_channels
        head_embedding = (hidden_channels + 1) if args.use_age else hidden_channels

        for i in range(args.num_layer_head - 1):
            self.graph_pred_linear.add_module(str(2*i), torch.nn.Linear(head_embedding, head_embedding))
            self.graph_pred_linear.add_module(str(2*i+1), torch.nn.ReLU())
            if args.head_dropout:
                self.graph_pred_linear.add_module("drop{}".format(str(i)), torch.nn.Dropout(self.dropout))
        self.graph_pred_linear.add_module(str(2*args.num_layer_head), torch.nn.Linear(head_embedding, num_tasks))

    def forward(self, input_batch):
        age = input_batch.age
        results = []
        for omix_name in self.omixs:
            omix_result = eval("self.{}_encoder".format(omix_name))(input_batch, omix_name)
            results.append(omix_result)
        
        results = torch.cat(results, dim=1)
        if self.use_age:
            results = torch.cat([results, age[:, None]], dim=-1)
            
        return F.softmax(self.graph_pred_linear(results), dim=-1)

    def format_data(self, input_batch, omix_name):
        pass

class DeeperGCN_Vnode(torch.nn.Module):
    def __init__(self, args):
        super(DeeperGCN_Vnode, self).__init__()

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

        self.gcns = torch.nn.ModuleList()
        self.pathway_gcns = torch.nn.ModuleList()
        self.norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):

            if conv == 'gen':
                gcn = GENConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=conv_encode_edge, edge_feat_dim=hidden_channels,
                              norm=norm, mlp_layers=mlp_layers)
            else:
                raise Exception('Unknown Conv Type')
            pathway_gcn = PathwayConv(hidden_channels, hidden_channels,
                              aggr=aggr,
                              t=t, learn_t=self.learn_t,
                              p=p, learn_p=self.learn_p,
                              msg_norm=self.msg_norm, learn_msg_scale=learn_msg_scale,
                              encode_edge=conv_encode_edge, edge_feat_dim=hidden_channels,
                              norm=norm, mlp_layers=mlp_layers)
            self.gcns.append(gcn)
            self.pathway_gcns.append(pathway_gcn)
            self.norms.append(norm_layer(norm, hidden_channels))

        self.node_embedding = args.node_embedding
        if self.node_embedding:
            self.node_embedding_encoder = torch.nn.Embedding(args.node_num, args.node_embedding_dim)
        input_dim = 3 + (args.node_embedding_dim if self.node_embedding else 0) + (2 if self.mul_attr else 0)
        self.node_features_encoder = torch.nn.Linear(input_dim, hidden_channels)
        
        self.edge_encoder = torch.nn.Linear(7 if args.use_column is None else 1, hidden_channels)
        self.use_edge_attr = args.use_edge_attr
        self.pathway_global_node = args.pathway_global_node
        if self.pathway_global_node:
            self.pathway_num = args.pathway_num
            self.pathway_features_encoder = torch.nn.Linear(2, hidden_channels)
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
            readout_in = (self.pathway_num // 4) * hidden_channels
            self.readout_func = nn.Linear(readout_in, hidden_channels)

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception('Unknown Pool Type')

    def forward(self, input_batch, omix_name):
        x = input_batch.x
        edge_index = input_batch.edge_index
        #816136, 128
        edge_attr = input_batch.edge_attr
        batch = input_batch.batch
        age = input_batch.age
        pathway_node_feature = eval("input_batch.pathway_{}_node_attr".format(omix_name))

        if self.node_embedding:
            h = self.node_features_encoder(torch.cat([x[:, :-1], self.node_embedding_encoder(x[:, -1].to(torch.long))], dim=-1))
        else:
            h = self.node_features_encoder(x)
        if self.use_edge_attr:
            edge_emb = self.edge_encoder(edge_attr)
        else:
            edge_emb = None
        if self.pathway_global_node:
            #pathway_embedding = self.pathway_encoder(input_batch.pathway_global_nodes)
            pathway_embedding = self.pathway_features_encoder(pathway_node_feature)
            for i, node_size in enumerate(torch.cumsum(input_batch.node_size, dim=0).detach().cpu().numpy()):
                #h[node_size-self.pathway_num:node_size] = torch.zeros(self.pathway_num, self.hidden_channels)
                h[node_size-self.pathway_num:node_size] = pathway_embedding[i*self.pathway_num:(i+1)*self.pathway_num]

        pathway_edge_index, pathway_edge_emb, pathway_mask = self.format_pathway_data(input_batch, omix_name)

        if self.block == 'res+':

            h = self.gcns[0](h, edge_index, edge_emb)

            for layer in range(1, self.num_layers):
                h1 = self.norms[layer - 1](h)
                h2 = F.relu(h1)
                h2 = F.dropout(h2, p=self.dropout, training=self.training)
                h = self.gcns[layer](h2, edge_index, edge_emb)
                h = self.pathway_gcns[layer](h, pathway_edge_index, pathway_edge_emb, pathway_mask) + h

            h = self.norms[self.num_layers - 1](h)
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
                h2 = self.norms[layer](h1)
                if layer != (self.num_layers - 1):
                    h = F.relu(h2)
                else:
                    h = h2
                h = F.dropout(h, p=self.dropout, training=self.training)
        else:
            raise Exception('Unknown block Type')
        
        if self.pathway_global_node:
            pathway_results = []
            pathway_batch = []
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
                h_graph = torch.flatten(F.max_pool1d(pathway_results.transpose(1,2), 4), start_dim=1)
                h_graph = self.readout_func(h_graph)
        else:
            h_graph = self.pool(h, batch)
        
        return h_graph

    def format_pathway_data(self, input_batch, omix_name):
        cum_node_num = torch.cumsum(input_batch.node_size, dim=-1)
        cum_edge_num = torch.cumsum(eval("input_batch.pathway_{}_edges_num".format(omix_name)), dim=-1)
        pathway_edge_index = eval("input_batch.pathway_{}_edges".format(omix_name))
        for i, edge_num in enumerate(cum_edge_num[:-1]):
            stop_edge_num = cum_edge_num[i+1]
            pathway_edge_index[edge_num:stop_edge_num] += cum_node_num[i]
        pathway_edge_index = pathway_edge_index.transpose(0,1)
        pathway_edge_emb = eval("input_batch.pathway_{}_edge_attr".format(omix_name))
        pathway_vnode_idx = pathway_edge_index[1]
        pathway_mask = torch.zeros((cum_node_num[-1], 1))
        pathway_mask[pathway_vnode_idx, :] = 1
        return pathway_edge_index, pathway_edge_emb, pathway_mask.to(cum_node_num.device)

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
    