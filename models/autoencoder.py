import math

import torch
import torch.nn as nn

from models.multilevel_gnn import MultilevelGNN

def findNextPowerOf2(n):
 
    # decrement `n` (to handle cases when `n` itself is a power of 2)
    n = n - 1
 
    # set all bits after the last set bit
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
 
    # increment `n` and return
    return n + 1

class AutoEncoder(MultilevelGNN):

    def __init__(self, args, pca_params=None, pathway_indexs=None):
        super(AutoEncoder, self).__init__(args, pca_params, pathway_indexs)
        self.node_num = 5135
        self.decoder_dim = args.decoder_dim
        self.decoder_type = args.decoder_type
        if self.decoder_type == "flatten":
            self.decoder = nn.ModuleList([
                nn.Linear(self.args.final_channels*146*self.pca_dim*3, self.decoder_dim),
                nn.ReLU(),
                nn.Linear(self.decoder_dim, self.decoder_dim),
                nn.ReLU(),
                nn.Linear(self.decoder_dim, self.node_num*3)
            ])
        elif self.decoder_type == "foreach":
            pathway_nums = pathway_indexs.max().item()+1
            self.decoder = nn.ModuleList([
                nn.Sequential(nn.Linear(self.args.final_channels*self.args.pca_dim, self.decoder_dim),
                nn.ReLU(),
                nn.Linear(self.decoder_dim, (pathway_indexs == i).sum().item())
                ) for i in range(pathway_nums)
            ])
        elif self.decoder_type == "foreach_diffhidden":
            pathway_nums = pathway_indexs.max().item()+1
            self.decoder = []
            for i in range(pathway_nums):
                output_dim = (pathway_indexs == i).sum().item()
                decoder_dim = findNextPowerOf2(int(math.sqrt(output_dim*self.args.final_channels)))
                self.decoder.append(
                    nn.Sequential(nn.Linear(self.args.final_channels*self.args.pca_dim, decoder_dim),
                    nn.ReLU(),
                    nn.Linear(decoder_dim, output_dim)
                    )
                )
            self.decoder = nn.ModuleList(self.decoder)

        self.init_weight()
        
    def forward(self, input_batch, x=None, gene_pca_match=None, raw_indice=None, age=None):
        h = self.encoder(input_batch)
        if self.decoder_type == "flatten":
            output = self.flatten_decoder(h)
        elif "foreach" in self.decoder_type:
            output = self.foreach_decoder(h)
        return output, h, None

    def encoder(self, input_batch):
        x = input_batch.x.reshape(-1,1)
        gene_pca_match = input_batch.gene_pca_match
        raw_indice = input_batch.raw_indice

        if self.args.node_embedding:
            x = (x.reshape(-1, self.node_num*3, 1) * self.node_embedding).reshape(-1, self.node_embedding.shape[-1])

        edge_index = input_batch.edge_index
        edge_attr = input_batch.edge_attr

        for i, layer in enumerate(self.gnn_model):
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
        
        if self.args.dense_gnn:
            x = torch.cat(feature_list, dim=-1)
        # if self.args.value_att_mask:
        #     if self.args.merge_mode == 'mult':
        #         x = x * mask_x.reshape(-1,1)
        #     elif self.args.merge_mode == 'add':
        #         x = self.args.add_coef1*x + self.args.add_coef2*mask_x.reshape(-1,1)
        #     elif self.args.merge_mode == 'cat':
        #         x = self.args.add_coef1*x + self.args.add_coef2*mask_x.reshape(-1,1)
        
        pca_match_index = gene_pca_match + torch.tensor(range(gene_pca_match.shape[0])).to(x.device)[:, None]*self.node_num*3
        if self.args.pca_match_mask:
            pca_match_mask = torch.where(gene_pca_match > 0, 1, 0).to(x.device)
            x = x[pca_match_index] * pca_match_mask[:, :, None]
        else:
            x = x[pca_match_index]
        raw_data = x
        raw_indice = raw_indice[:,None, :,None].repeat(1, self.args.final_channels,1,self.pca_dim)
        if self.mutual_info_mask:
            pca_result = raw_data.unsqueeze(3).repeat(1,1,1,self.pca_dim) * (self.learnable_pca_params * self.info_mask)[:, None,:]
            pca_result = pca_result.permute(0,2,1,3)
        else:
            if self.args.final_channels != 1:
                pca_result = raw_data.unsqueeze(3).repeat(1,1,1,self.pca_dim) * (self.learnable_pca_params * self.info_mask)[:, None,:]
                pca_result = pca_result.permute(0,2,1,3)
            else:
                pca_result = raw_data * self.learnable_pca_params

        b, c, n, pca_dim = pca_result.shape
        if self.decoder_type == "flatten":
            x = torch.zeros(b, c, 146*3, self.pca_dim).to(raw_indice.device).scatter_reduce(2, raw_indice, pca_result, reduce="sum").reshape(-1, c, 146, self.pca_dim*3) # bs, c, 146, pca_dim
        elif "foreach" in self.decoder_type:
            x = torch.zeros(b, c, 146*3, self.pca_dim).to(raw_indice.device).scatter_reduce(2, raw_indice, pca_result, reduce="sum")
        return x
    
    def flatten_decoder(self, h):
        x = h.flatten(1)
        for layer in self.decoder:
            x = layer(x)
        return x

    def foreach_decoder(self, h):
        x = h.permute(0,2,1,3).flatten(2)
        pred_y = []
        for i, block in enumerate(self.decoder):
            pred_y.append(block(x[:, i, :]))
        pred_y = torch.cat(pred_y, dim=-1)
        return pred_y
    
    def set_pca_params(self, pca_params, mutual_info_mask):
        idxs = []
        for i in range(len(mutual_info_mask)):
            if mutual_info_mask[i]>0:
                idxs.append(i)
        self.learnable_pca_params = nn.Parameter(data=torch.zeros([len(mutual_info_mask), self.pca_dim]), requires_grad=(not self.args.freeze_pca_weight))
        #nn.init.constant_(self.learnable_pca_params.data, 0)
        self.learnable_pca_params.data[idxs] =  pca_params.to(torch.float32).to(self.learnable_pca_params.data.device)[:, :self.pca_dim]
    