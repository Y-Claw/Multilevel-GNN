import torch
from .gcn_lib.sparse.torch_vertex import GENConv, GATConv, GraphConv
from .gcn_lib.sparse.torch_nn import norm_layer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, TopKPooling
import logging
import math
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import numpy as np

from models.utils import PCA_svd

class MultilevelGNN(nn.Module):

    def __init__(self, args, pca_params=None, pathway_indexs=None):
        super(MultilevelGNN, self).__init__()
        self.pca_compare = args.pca_compare
        self.pca_prelinear = args.pca_prelinear
        self.learnable_pca = args.learnable_pca
        self.pca_loss = args.pca_loss
        self.pca_indep_loss = args.pca_indep_loss
        self.pca_dim = args.pca_dim
        self.pathway_pool_dim = args.pathway_pool_dim
        self.pca_pool_dim = args.pca_pool_dim
        self.pathway_indexs = None
        self.reorder_idxs = None
        self.mutual_info_mask = args.mutual_info_mask
        self.mutual_info_threshold = args.mutual_info_threshold
        self.pca_loss_coef = args.pca_loss_coef
        self.node_select_threshold = args.node_select_threshold
        self.mutual_neighbors = args.mutual_neighbors
        self.args = args
        self.node_num = 5135 #5308
        self.mutual_info_mask_cache = {}
        self.head_dim = args.head_dim
        self.epoch = None
        self.step = None
        self.used_omics = args.used_omics

        if args.input_drop is not None:
            self.input_drop = nn.Dropout(p=args.input_drop)
        else:
            self.input_drop = None

        if args.input_emb_drop is not None:
            self.input_emb_drop = nn.Dropout(p=args.input_emb_drop)
        else:
            self.input_emb_drop = None

        if args.node_embedding:
            self.node_embedding = nn.Parameter(data=torch.rand([self.node_num*3, self.args.node_embedding_dim]), requires_grad=not args.freeze_node_embedding)
            if args.embedding_init_type == "xavier":
                nn.init.xavier_uniform_(self.node_embedding)
            elif args.embedding_init_type == "ones":
                nn.init.constant_(self.node_embedding, 1)
            elif args.embedding_init_type == "constant":
                nn.init.constant_(self.node_embedding, args.emb_val)
            elif args.embedding_init_type == "uniform":
                nn.init.uniform_(self.node_embedding)
            self.node_embedding_dim = self.args.node_embedding_dim
        else:
            self.node_embedding = None
            self.node_embedding_dim = 1

        gnn_blocks = []
        gnn_blocks.append(GraphConv(self.node_embedding_dim, args.hidden_channels, act=args.gnn_act, conv=args.gnn_name, mlp_norm=args.gnn_mlp_norm, drop=args.gnn_dropout))
        for i in range(args.num_layers-2):
            gnn_blocks.append(GraphConv(args.hidden_channels, args.hidden_channels, act=args.gnn_act, conv=args.gnn_name, mlp_norm=args.gnn_mlp_norm, drop=args.gnn_dropout))
        gnn_blocks.append(GraphConv(args.hidden_channels, args.final_channels, heads=args.final_head, act=args.gnn_act, conv=args.gnn_name, norm=args.gnn_last_norm, mlp_norm=args.gnn_mlp_norm, drop=args.gnn_dropout))
        self.gnn_model = nn.ModuleList(gnn_blocks)

        
        self.learnable_pca_params = nn.Parameter(data=torch.rand([25015, self.pca_dim]), requires_grad=(not self.args.freeze_pca_weight)) # 24542
        if pca_params is None:
            if args.pca_init_type == None:
                nn.init.xavier_uniform_(self.learnable_pca_params.data)
            elif args.pca_init_type == "orthogonal":
                nn.init.orthogonal_(self.learnable_pca_params.data)
            elif args.pca_init_type == "normal":
                pass
        else:
            self.learnable_pca_params.data = pca_params

        if self.pca_prelinear:
            self.pre_linear = nn.Sequential(
                nn.Linear(6, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, 6)
            )
        if args.edge_type == 'merge':
            args.final_channels *= 2
        if args.dense_gnn:
            args.final_channels = (args.num_layers-1)*args.hidden_channels+args.final_channels

        conv_blocks = []
        input_channel = args.final_channels
        for output_channel, kernel in zip(args.conv_channel_list, args.conv_kernel_list):
            conv_blocks.append(nn.Conv2d(input_channel, output_channel, kernel, padding=kernel//2))
            conv_blocks.append(nn.ReLU())
            input_channel = output_channel
        self.conv_model = nn.ModuleList(conv_blocks)

        self.pooling = nn.MaxPool2d((self.pathway_pool_dim,self.pca_pool_dim))
        self.drop1 = nn.Dropout(0.25 if args.feature_drop else 0)
        if self.pca_compare:
            self.pre_linear = nn.Sequential(
                nn.Linear(6912, self.head_dim),
                nn.ReLU()
            )
            self.head = nn.Sequential(
                nn.Linear(65, self.head_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.head_dim, 2),
                nn.Softmax()
            )
        else:
            input_dim = args.conv_channel_list[-1]*(146//self.pathway_pool_dim)*((len(self.used_omics)*self.pca_dim)//self.pca_pool_dim)+(1 if self.args.use_age else 0) 
            self.head = nn.Sequential(
                nn.Linear(input_dim, self.head_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.head_dim, 2),
                nn.Softmax()
            )
        self.init_weight()

    
    def forward(self, input_batch, x=None, gene_pca_match=None, raw_indice=None, age=None, require_grad=True):
        #import pdb;pdb.set_trace()
        with torch.no_grad() if not require_grad else torch.enable_grad():
            if x is not None:
                mask_x = x
                x = x.reshape(-1,1)
            else:
                mask_x = input_batch.x
                x = input_batch.x.reshape(-1,1)
                gene_pca_match = input_batch.gene_pca_match
                raw_indice = input_batch.raw_indice
                age = input_batch.age

            if self.input_drop is not None:
                x = self.input_drop(x)
            #if torch.nan in self.node_embedding or (self.epoch == 8 and self.step == 2):
            #    import pdb; pdb.set_trace()

            if self.args.node_embedding:
                x = (x.reshape(-1, self.node_num*3, 1) * self.node_embedding).reshape(-1, self.node_embedding.shape[-1])

            if self.input_emb_drop is not None:
                x = self.input_emb_drop(x)

            if x is not None:
                if isinstance(input_batch.edge_index, list):
                    edge_nums = [tmp_edge_idx.shape[-1] // self.args.device_num for tmp_edge_idx in input_batch.edge_index]
                    edge_index = [edge_index[:, :edge_nums[i]].to(x.device) for i, edge_index in enumerate(input_batch.edge_index)]
                    edge_attr = [edge_attr[:, :edge_nums[i]].to(x.device) for i, edge_attr in enumerate(input_batch.edge_attr)]
                else:
                    total_edge_num = input_batch.edge_index.shape[-1]
                    edge_num = total_edge_num // self.args.device_num
                    edge_index = input_batch.edge_index[:, :edge_num].to(x.device)
                    edge_attr = input_batch.edge_attr[:, :edge_num].to(x.device)
            
            if not self.args.weighted_edge:
                edge_attr = None

            if isinstance(edge_index, list):
                feature_list = []
                raw_x = x
                for single_edge_index in edge_index:
                    x = raw_x
                    for layer in self.gnn_model:
                        try:
                            x = layer(x, single_edge_index)
                        except:
                            import pdb
                            pdb.set_trace()
                    feature_list.append(x)
                x = torch.cat(feature_list, dim=-1)
            else:
                feature_list = []
                for i, layer in enumerate(self.gnn_model):
                    #import pdb;pdb.set_trace()
                    try:
                        if self.args.dense_gnn:
                            x = layer(x, edge_index, edge_attr)
                            feature_list.append(x)
                        elif self.args.resgnn:
                            old_x = x
                            x = layer(x, edge_index, edge_attr) + old_x
                        else:
                            x = layer(x, edge_index, edge_attr)
                        if i+1 != len(self.gnn_model) and self.args.repeat_mask and (i+1) % self.args.repeat_cyclic == 0:
                            if self.args.repeat_norm:
                                x = x / (x**2).sum(1).sqrt()[:, None]
                            x = x * mask_x.reshape(-1,1)
                    except:
                        import pdb
                        pdb.set_trace()
            if self.args.dense_gnn:
                x = torch.cat(feature_list, dim=-1)
            if self.args.value_att_mask:
                if self.args.merge_mode == 'mult':
                    x = x * mask_x.reshape(-1,1)
                elif self.args.merge_mode == 'add':
                    x = self.args.add_coef1*x + self.args.add_coef2*mask_x.reshape(-1,1)
                elif self.args.merge_mode == 'cat':
                    x = self.args.add_coef1*x + self.args.add_coef2*mask_x.reshape(-1,1)
            pca_match_index = gene_pca_match + torch.tensor(range(gene_pca_match.shape[0])).to(x.device)[:, None]*self.node_num*3
            if self.args.pca_match_mask:
                pca_match_mask = torch.where(gene_pca_match >= 0, 1, 0).to(x.device)
                x = x[pca_match_index] * pca_match_mask[:, :, None]
            else:
                x = x[pca_match_index]

            #raw_data = input_batch.raw_data[:,:,None]
            if self.args.reduction_method == "linear_projection":
                raw_data = x
                raw_indice = raw_indice[:,None, :,None].repeat(1, self.args.final_channels,1,self.pca_dim)
                if self.mutual_info_mask:
                    if self.args.final_channels != 1:
                        pca_result = raw_data.unsqueeze(3).repeat(1,1,1,self.pca_dim) * (self.learnable_pca_params * self.info_mask)[:, None,:]
                        pca_result = pca_result.permute(0,2,1,3)
                    else:
                        pca_result = raw_data.unsqueeze(3).repeat(1,1,1,self.pca_dim) * (self.learnable_pca_params * self.info_mask)[:, None,:]
                        pca_result = pca_result.permute(0,2,1,3)
                        #pca_result = raw_data * (self.learnable_pca_params * self.info_mask)
                else:
                    if self.args.final_channels != 1:
                        pca_result = raw_data.unsqueeze(3).repeat(1,1,1,self.pca_dim) * (self.learnable_pca_params * self.info_mask)[:, None,:]
                        pca_result = pca_result.permute(0,2,1,3)
                    else:
                        pca_result = raw_data * self.learnable_pca_params

                b, c, n, pca_dim = pca_result.shape
                x = torch.zeros(b, c, 146*3, self.pca_dim).to(raw_indice.device).scatter_reduce(2, raw_indice, pca_result, reduce="sum").reshape(-1, c, 146, self.pca_dim*3)
                #t=torch.zeros(tb, 1, 146*3, self.pca_dim).to(raw_indice.device).scatter_reduce(2, raw_indice, pca_result.repeat(1,1,1,self.pca_dim), reduce="sum").reshape(-1, 1, 146, self.pca_dim*3)
                if self.args.reorder_pathway and self.reorder_idxs is not None:
                    x = x[:,:,self.reorder_idxs, :]
            elif "pca" in self.args.reduction_method:
                pca_results = []
                x = x.permute(2,0,1)
                for i in range(self.args.pathway_num*3):
                    index = torch.nonzero((raw_indice[0] == i))
                    #feature = PCA_svd(x[:,:,index.min(): index.max()+1], self.pca_dim)
                    if self.args.reduction_method == "pca_lowrank":
                        u, s, v = torch.pca_lowrank(x[:,:,index.min(): index.max()+1], self.pca_dim, center=False, niter=self.args.pca_lowrank_niter)
                        pca_tensor = torch.bmm(x[:,:,index.min(): index.max()+1],v)
                    elif self.args.reduction_method == "pca_svd":
                        pca_tensor = PCA_svd(x[:,:,index.min(): index.max()+1], self.pca_dim, center=False)
                    pca_results.append(pca_tensor) #(pathway,c,b,pca_dim)
                x = torch.stack(pca_results)
                x = x.permute(2,1,0,3).reshape(x.shape[2],x.shape[1],self.args.pathway_num,-1)

        pca_feature = x
        if self.pca_prelinear:
            x = self.pre_linear(x)

        for i, layer in enumerate(self.conv_model):
            x = layer(x)
        """x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)"""
        if len(self.used_omics) == 3:
            x = self.pooling(x)
            if self.pca_compare:
                #error: pre_linear -> drop
                x = torch.flatten(x, start_dim=1)
                x= self.pre_linear(x)
                #x = self.drop1(x)
            else:
                x = self.drop1(x)
                x = torch.flatten(x, start_dim=1)
        else:
            selected_col = []
            for i in self.used_omics:
                int_i = int(i)
                selected_col.extend([col for col in range(int_i*self.pca_dim, (int_i+1)*self.pca_dim)])
            x = x[:,:,:,selected_col]
            x = self.pooling(x)
            x = self.drop1(x)
            x = torch.flatten(x, start_dim=1)
        
        if self.args.use_age:
            x = torch.cat([x, age[:, None]], dim=-1)
        x = self.head(x)
        
        return x, pca_feature

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)

    def set_pca_params(self, pca_params, mutual_info_mask):
        idxs = []
        for i in range(len(mutual_info_mask)):
            if mutual_info_mask[i]>0:
                idxs.append(i)
        self.learnable_pca_params = nn.Parameter(data=torch.zeros([len(mutual_info_mask), self.pca_dim]), requires_grad=(not self.args.freeze_pca_weight))
        #nn.init.constant_(self.learnable_pca_params.data, 0)
        self.learnable_pca_params.data[idxs] =  pca_params[:, :self.pca_dim].to(torch.float32).to(self.learnable_pca_params.data.device)
    
    def set_pathway_indexs(self, pathway_indexs):
        self.pathway_indexs = pathway_indexs
    
    def init_precise_orthogonal(self):
        start = 0
        init_index = []
        t_mean = torch.mean(torch.abs(nn.init.orthogonal_(self.learnable_pca_params.data)))
        for i in range(1, len(self.pathway_indexs)):
            if self.info_mask[i, 0] > 0:
                init_index.append(i)
            if i < len(self.pathway_indexs) and self.pathway_indexs[i-1] != self.pathway_indexs[i]:  
                #nn.init.orthogonal_(self.learnable_pca_params.data[start:i])
                self.learnable_pca_params.data[init_index] = nn.init.orthogonal_(self.learnable_pca_params.data[init_index])
                self.learnable_pca_params.data[init_index] *= (t_mean/torch.mean(torch.abs(self.learnable_pca_params.data[init_index])))
                start = i
                init_index = []
        self.learnable_pca_params.data[init_index] = nn.init.orthogonal_(self.learnable_pca_params.data[init_index])
        #nn.init.orthogonal_(self.learnable_pca_params.data[start:])
    
    def get_feature_loss(self, pca_feature):
        loss = 0
        if self.pca_loss:
            b, c, pn, o = pca_feature.shape
            pca_feature = pca_feature.reshape(b, -1)
            feature_std = torch.std(pca_feature, dim=0)
            loss = loss - self.pca_loss_coef * torch.log(torch.mean(feature_std))
        if self.pca_indep_loss:
            indep_loss = 0
            count = 0
            learnable_pca_params = self.learnable_pca_params * self.info_mask
            for i in range(self.pca_dim-1):
                for j in range(i+1, self.pca_dim):
                    count += 1
                    mul_res = torch.zeros(torch.max(self.pathway_indexs)+1).to(self.pathway_indexs.device).scatter_reduce(0, self.pathway_indexs, learnable_pca_params.data[:, i] * learnable_pca_params.data[:, j], reduce="sum")
                    len_res = torch.sqrt(torch.zeros(torch.max(self.pathway_indexs)+1).to(self.pathway_indexs.device).scatter_reduce(0, self.pathway_indexs, learnable_pca_params.data[:, i]**2, reduce="sum") * torch.zeros(torch.max(self.pathway_indexs)+1).to(self.pathway_indexs.device).scatter_reduce(0, self.pathway_indexs, learnable_pca_params.data[:, j]**2, reduce="sum"))
                indep_loss += torch.mean(torch.abs(mul_res / (len_res+1e-7)))
            loss += (indep_loss / count)
        
        return loss
    
    def set_info_mask(self, info_mask):
        self.info_mask = nn.Parameter(data=info_mask, requires_grad=False)

    def generate_mutual_mask(self, x, y, mutual_classif=True, fold=0, tf_token=None):
        x = torch.tensor(x)
        y = torch.tensor(y)
        if self.args.freeze_mutual_select_init:
            random_state = self.args.random_state
        else:
            random_state = None
        if mutual_classif:
            mutual_info = mutual_info_classif(x, y, n_neighbors=self.mutual_neighbors,random_state=random_state)
        else:
            mutual_info = mutual_info_regression(x, y, n_neighbors=self.mutual_neighbors,random_state=random_state)
        if fold in self.mutual_info_mask_cache:
            res = self.mutual_info_mask_cache[fold]
        elif self.mutual_info_threshold is None:
            res = [torch.where(torch.tensor(mutual_info) < self.node_select_threshold * np.mean(mutual_info), torch.zeros(mutual_info.shape), torch.ones(mutual_info.shape))[:, None], mutual_info]
            self.mutual_info_mask_cache[fold] = res
        else:
            res = [torch.where(torch.tensor(mutual_info) < self.mutual_info_threshold, torch.zeros(mutual_info.shape), torch.ones(mutual_info.shape))[:, None], mutual_info]
            self.mutual_info_mask_cache[fold] = res
        if tf_token is not None:
            tf_token = torch.tensor(tf_token)
            if self.args.remain_all_tf:
                self.mutual_info_mask_cache[fold][0] = self.mutual_info_mask_cache[fold][0].to(torch.int) | tf_token[:, None]
                res[0] = self.mutual_info_mask_cache[fold][0].to(torch.int) | tf_token[:, None]
        return res

    def load_representation(self, representation_path):
        representation = torch.from_numpy(np.load(representation_path)).to(torch.float)
        self.node_embedding.data = representation

    def set_reorder_idxs(self, reorder_idx):
        self.reorder_idxs = torch.tensor(reorder_idx)

    def load_autoencoder_pretrain(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location="cuda:"+str(self.args.device))
        self.learnable_pca_params = nn.Parameter(data=torch.zeros(checkpoint['model_state_dict']['learnable_pca_params'].shape), requires_grad=(not self.args.freeze_pca_weight))
        self.set_info_mask(checkpoint['model_state_dict']['info_mask'])
        layer_names = [name for name, _ in self.named_parameters()]
        partial_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if k in layer_names}
        
        self.load_state_dict(partial_state_dict, strict=False)
        del checkpoint, partial_state_dict

