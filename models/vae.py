import math
import pandas as pd
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F

from models.multilevel_gnn import MultilevelGNN
from models.diff_pooling import DiffPool

from models.utils import PCA_svd

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

def custom_weights_init(block):
    for m in block.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            #nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            #nn.init.constant_(m.bias, 0)

class VAE(MultilevelGNN):

    def __init__(self, args, pca_params=None, pathway_indexs=None):
        super(VAE, self).__init__(args, pca_params, pathway_indexs)
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
        
        if self.args.reorder_type == "diff_pooling":
            if self.args.diff_pooling_location == "pathway":
                self.diff_pooling = DiffPool(self.args.final_channels, 2, self.args.pathway_num, self.args.diff_pooling_layer, self.args.diff_pooling_hidden_dim,
                    self.args.diff_pooling_output_dim, args)
            elif self.args.diff_pooling_location == "head":
                self.diff_pooling = DiffPool(self.args.conv_channel_list[-1], 2, self.args.pathway_num, self.args.diff_pooling_layer, self.args.diff_pooling_hidden_dim,
                    self.args.diff_pooling_output_dim, args)

        H = self.args.final_channels*self.args.pca_dim
        self.enc_mu = torch.nn.Linear(H, H)
        self.enc_log_sigma = torch.nn.Linear(H, H)

        self.init_weight()
    
    def train_step(self, input_batch, require_grad=True):
        with torch.no_grad() if not require_grad else torch.enable_grad():
            q_z, h, loss, gene_feature = self.encoder(input_batch)
            b, _, c = h.shape
            if self.args.vae_generate_train_sample:
                h = q_z.rsample()
            if self.args.channel_one:
                h = h[:,:,:c//2].reshape(b, 1, 146, -1)
            else:
                h = h[:,:,:c//2].permute(0,2,1).reshape(b, c//2, 146, 3)
        if self.args.reorder_pathway and self.reorder_idxs is not None:
            h = h[:,:,self.reorder_idxs, :]
        pred, pca_feature, l, e = self.predict_head(h, input_batch.age)
        return pred, pca_feature, l, e, gene_feature#, [loss]

    def eval_step(self, input_batch, require_grad=True):
        with torch.no_grad() if not require_grad else torch.enable_grad():
            q_z, h, loss, gene_feature = self.encoder(input_batch)
            b, _, c = h.shape
            if self.args.channel_one:
                h = h[:,:,:c//2].reshape(b, 1, 146, -1)
            else:
                h = h[:,:,:c//2].permute(0,2,1).reshape(b, c//2, 146, 3)
        if self.args.reorder_pathway and self.reorder_idxs is not None:
            h = h[:,:,self.reorder_idxs, :]
        
        return self.predict_head(h, input_batch.age)


    def forward(self, input_batch, x=None, gene_pca_match=None, raw_indice=None, age=None):
        q_z, h, loss, gene_feature = self.encoder(input_batch)
        z = q_z.rsample()
        if self.decoder_type == "flatten":
            output = self.flatten_decoder(z)
        elif "foreach" in self.decoder_type:
            output = self.foreach_decoder(z)
        return {"pred_x": output, "embedding":h, "q_z": q_z, "z": z, "loss": loss}

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
            pca_match_mask = torch.where(gene_pca_match >= 0, 1, 0).to(x.device)
            x = x[pca_match_index] * pca_match_mask[:, :, None]
        else:
            x = x[pca_match_index]
        
        gene_feature = x
        if self.args.reduction_method == "linear_projection":
            raw_data = x # b, num_gene, embedding
            raw_indice = raw_indice[:,None, :,None].repeat(1, self.args.final_channels,1,self.pca_dim)
            
            pca_result = raw_data.unsqueeze(3).repeat(1,1,1,self.pca_dim) * (self.learnable_pca_params * self.info_mask)[:, None,:]
            pca_result = pca_result.permute(0,2,1,3) #b, channel, number_gene, 
            
            b, c, n, pca_dim = pca_result.shape
            if self.decoder_type == "flatten":
                x = torch.zeros(b, c, 146*3, self.pca_dim).to(raw_indice.device).scatter_reduce(2, raw_indice, pca_result, reduce="sum").reshape(-1, c, 146, self.pca_dim*3) # bs, c, 146, pca_dim
            elif "foreach" in self.decoder_type:
                x = torch.zeros(b, c, 146*3, self.pca_dim).to(raw_indice.device).scatter_reduce(2, raw_indice, pca_result, reduce="sum")

            x = x.permute(0,2,1,3).flatten(2)
        elif "pca" in self.args.reduction_method:
            pca_results = []
            x = x.permute(2,0,1)
            for i in range(self.args.pathway_num*3):
                index = torch.nonzero((raw_indice[0] == i))
                #feature = PCA_svd(x[:,:,index.min(): index.max()+1], self.pca_dim)
                if self.args.reduction_method == "pca_lowrank":
                    u, s, v = torch.pca_lowrank(x[:,:,index.min(): index.max()+1], self.pca_dim, center=False)
                    pca_tensor = torch.bmm(x[:,:,index.min(): index.max()+1],v)
                elif self.args.reduction_method == "pca_svd":
                    pca_tensor = PCA_svd(x[:,:,index.min(): index.max()+1], self.pca_dim, center=False)
                pca_results.append(pca_tensor) #(pathway,c,b,pca_dim)
            x = torch.stack(pca_results)
            x = x.permute(2,0,1,3).reshape(x.shape[2],x.shape[0],-1)

        mu = self.enc_mu(x)
        log_sigma = self.enc_log_sigma(x)
        sigma = torch.exp(log_sigma)
        
        loss_std = -mu.flatten(1).permute(1,0).std(1).mean()
        loss_idp = 0
        corr_mask = (torch.ones(mu.shape[-1],mu.shape[-1]) - torch.eye(mu.shape[-1])).to(mu.device)
        loss_corr = torch.stack([(torch.corrcoef(pathway_attr_matrix) * corr_mask).abs() for pathway_attr_matrix in mu.permute(1,2,0)]).mean()
        return torch.distributions.Normal(loc=mu, scale=sigma+1e-7), torch.cat([mu, sigma], dim=-1), [loss_std, loss_idp, loss_corr], gene_feature
    
    def flatten_decoder(self, h):
        x = h.flatten(1)
        for layer in self.decoder:
            x = layer(x)
        return x

    def foreach_decoder(self, h):
        x = h
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
    
    def predict_head(self, x, age):
        l = e = 0
        pca_feature = x
        if self.pca_prelinear:
            x = self.pre_linear(x)
        if self.args.reorder_type == "diff_pooling" and self.args.diff_pooling_location == "pathway":
            b, c, pathway_num, d = x.shape
            x = x.permute((0,3,2,1)).reshape(-1,self.args.pathway_num, self.args.final_channels)
            x, l, e = self.diff_pooling(x, self.get_pathway_adj().to(x.device))
            x = x.reshape(b, -1)
            x = self.drop1(x)
            #x = torch.flatten(x, start_dim=1)
        else:
            for i, layer in enumerate(self.conv_model):
                x = layer(x)
            
            if self.args.reorder_type == "diff_pooling" and self.args.diff_pooling_location == "head":
                b, c, pathway_num, d = x.shape
                x = x.permute((0,3,2,1)).reshape(-1,self.args.pathway_num, self.args.conv_channel_list[-1])
                x, l, e = self.diff_pooling(x, self.get_pathway_adj().to(x.device))
                x = x.reshape(b, -1)
                x = self.drop1(x)
            else:
                x = x if self.args.reorder_type == "no_pooling" else self.pooling(x)
                x = self.drop1(x)
                x = torch.flatten(x, start_dim=1)
        
        if self.args.use_age:
            x = torch.cat([x, age[:, None]], dim=-1)
        
        x = self.head(x)

        return x, pca_feature, l, e

    def reconstruct_head(self, args):
        if args.reorder_type == "no_pooling":
            input_dim = args.conv_channel_list[-1]*(146)*(3*self.pca_dim)+(1 if self.args.use_age else 0) 
            self.head = nn.Sequential(
                nn.Linear(input_dim, self.head_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.head_dim, 2),
                nn.Softmax()
            )
        elif args.reorder_type == "diff_pooling":
            pathway_num = self.args.pathway_num
            for i in range(self.args.diff_pooling_layer):
                pathway_num = math.ceil(pathway_num * 0.25)
            
            input_dim = self.args.diff_pooling_output_dim*(pathway_num)*(3*self.pca_dim)+(1 if self.args.use_age else 0) 
            self.head = nn.Sequential(
                nn.Linear(input_dim, self.head_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.head_dim, 2),
                nn.Softmax()
            )
        else:
            input_dim = args.conv_channel_list[-1]*(146//self.pathway_pool_dim)*((3*self.pca_dim)//self.pca_pool_dim)+(1 if self.args.use_age else 0)
            self.head = nn.Sequential(
                nn.Linear(input_dim, self.head_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.head_dim, 2),
                nn.Softmax()
            )
        custom_weights_init(self.head)
    
    def get_pathway_adj(self):
        if self.args.pathway_similarity == "correlation":
            return self.pathway_similarity_matrix

    def set_pathway_similarity_matrix(self, pathway_similarity_matrix):
        self.pathway_similarity_matrix = (torch.tensor(pathway_similarity_matrix) + torch.eye(self.args.pathway_num)).to(torch.float32)

    def get_embedding_similarity(self):
        pathway_matrix = []
        embedding_dir = osp.join('embeddings', 'embeddings_{}'.format(self.args.load_autoencoder_epoch)) if self.args.load_autoencoder_epoch is not None else 'embeddings'
        cut_idx = self.args.autoencoder_ckpt_path.rfind("/")
        for i, file_name in enumerate(['{}/{}/mrna_embeddings.xlsx'.format(self.args.autoencoder_ckpt_path[:cut_idx], embedding_dir), '{}/{}/cnv_embeddings.xlsx'.format(self.args.autoencoder_ckpt_path[:cut_idx], embedding_dir), '{}/{}/mt_embeddings.xlsx'.format(self.args.autoencoder_ckpt_path[:cut_idx], embedding_dir)]):
            embedding = pd.read_excel(file_name, header=None).values
            embedding = embedding.reshape(embedding.shape[0],146,-1).swapaxes(1,0)[:, :, :4]
            pathway_matrix.append(embedding.reshape(146, -1))
        
        cat_pathway_matrix = np.concatenate(pathway_matrix, axis=-1)
        correlation_matrix = np.corrcoef(cat_pathway_matrix) - np.eye(cat_pathway_matrix.shape[0])
        max_idx = correlation_matrix.argmax()
        reorder_idxs = [max_idx//146, max_idx%146]
        remain_idx = set(range(146))
        for elem in reorder_idxs:
            remain_idx.remove(elem)
        sort_matrix = np.argsort(correlation_matrix)
        while len(reorder_idxs) < 146:
            source_idx = reorder_idxs[-1]
            for t in sort_matrix[source_idx][::-1]:
                if t in remain_idx:
                    reorder_idxs.append(t)
                    remain_idx.remove(t)
                    break
        return reorder_idxs, correlation_matrix

    def vae_loss(self, x_predict, x, z, q_z) -> dict:
        recons = x_predict
        input = x

        batch_size = input.size(0)
        bias_corr = batch_size *  (batch_size - 1)
        kld_weight = self.args.kld_weight  # Account for the minibatch samples from the dataset

        recons_loss =F.mse_loss(recons, input)
        mmd_loss_list = []
        for i in range(z.shape[1]):
            mmd_loss_list.append(self.compute_mmd(z[:, i, :]))
        mmd_loss = torch.mean(torch.stack(mmd_loss_list))
        #log_var = log_var.reshape(-1, log_var.shape[-1])
        #mu = mu.reshape(-1, mu.shape[-1])
        #kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        kld_loss = torch.distributions.kl_divergence(
                q_z, 
                torch.distributions.Normal(0, 1.)
                ).sum(-1).mean()
        loss = self.args.mmd_beta * recons_loss + \
               (1. - self.args.mmd_alpha) * kld_weight * kld_loss + \
               (self.args.mmd_alpha + self.args.mmd_reg_weight - 1.)/bias_corr * mmd_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss, 'MMD': mmd_loss, 'KLD':-kld_loss}

    def set_std_weight(self, std_weight):
        self.std_weight = std_weight

    def get_vae_sim_loss(self, pred, ground_y, grad_feat=None):
        loss = 0
        if self.args.std_weight:
            loss += self.args.std_weight_coef * (self.std_weight.to(pred.device) * F.l1_loss(pred, ground_y, reduction="none")).mean()
        else:
            loss += F.l1_loss(pred.to(torch.float32), ground_y.to(torch.float32))
        
        if self.args.grad_weight and grad_feat is not None:
            origin_grad_weight = grad_feat.grad.permute(1,0,2).flatten(1).abs().mean(1)
            grad_att = origin_grad_weight / origin_grad_weight.max()
            loss += self.args.grad_weight_coef * (grad_att * F.l1_loss(pred, ground_y, reduction="none")).mean()

        return loss

    def compute_kernel(self, x1, x2):
        # Convert the tensors into row and column vectors
        D = x1.size(1)
        N = x1.size(0)

        x1 = x1.unsqueeze(-2) # Make it into a column tensor
        x2 = x2.unsqueeze(-3) # Make it into a row tensor

        """
        Usually the below lines are not required, especially in our case,
        but this is useful when x1 and x2 have different sizes
        along the 0th dimension.
        """
        x1 = x1.expand(N, N, D)
        x2 = x2.expand(N, N, D)

        if self.args.mmd_kernel_type == 'rbf':
            result = self.compute_rbf(x1, x2)
        elif self.args.mmd_kernel_type == 'imq':
            result = self.compute_inv_mult_quad(x1, x2)
        else:
            raise ValueError('Undefined kernel type.')

        return result

    def compute_rbf(self, x1, x2, eps = 1e-7):
        """
        Computes the RBF Kernel between x1 and x2.
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        sigma = 2. * z_dim * self.args.z_var

        result = torch.exp(-((x1 - x2).pow(2).mean(-1) / sigma))
        return result

    def compute_inv_mult_quad(self, x1, x2, eps = 1e-7):
        """
        Computes the Inverse Multi-Quadratics Kernel between x1 and x2,
        given by

                k(x_1, x_2) = \sum \frac{C}{C + \|x_1 - x_2 \|^2}
        :param x1: (Tensor)
        :param x2: (Tensor)
        :param eps: (Float)
        :return:
        """
        z_dim = x2.size(-1)
        C = 2 * z_dim * self.args.z_var
        kernel = C / (eps + C + (x1 - x2).pow(2).sum(dim = -1))

        # Exclude diagonal elements
        result = kernel.sum() - kernel.diag().sum()

        return result

    def compute_mmd(self, z):
        # Sample from prior (Gaussian) distribution
        z = z.reshape(-1, z.shape[-1])
        prior_z = torch.randn_like(z)

        prior_z__kernel = self.compute_kernel(prior_z, prior_z)
        z__kernel = self.compute_kernel(z, z)
        priorz_z__kernel = self.compute_kernel(prior_z, z)

        mmd = prior_z__kernel.mean() + \
              z__kernel.mean() - \
              2 * priorz_z__kernel.mean()
        return mmd

    # def permute_pca_dim_and_channel(self):
    #     if args.edge_type == 'merge':
    #         args.final_channels *= 2
    #     if args.dense_gnn:
    #         args.final_channels = (args.num_layers-1)*args.hidden_channels+args.final_channels

    #     conv_blocks = []
    #     input_channel = args.pca_dim
    #     for output_channel, kernel in zip(args.conv_channel_list, args.conv_kernel_list):
    #         conv_blocks.append(nn.Conv2d(input_channel, output_channel, kernel, padding=kernel//2))
    #         conv_blocks.append(nn.ReLU())
    #         input_channel = output_channel
    #     self.conv_model = nn.ModuleList(conv_blocks)
    #     """self.conv1 = nn.Conv2d(args.final_channels, args.first_conv_channel, self.kernel_size, padding=self.kernel_size//2)
    #     if args.more_conv:
    #         self.conv2 = nn.Sequential(
    #             nn.Conv2d(args.first_conv_channel, 64, self.kernel_size, padding=self.kernel_size//2),
    #             nn.ReLU(),
    #             nn.Conv2d(64, 64, self.kernel_size, padding=self.kernel_size//2),
    #             nn.ReLU(),
    #             nn.Conv2d(64, 64, self.kernel_size, padding=self.kernel_size//2),
    #         )
    #     else:
    #         self.conv2 = nn.Conv2d(args.first_conv_channel, 64, self.kernel_size, padding=self.kernel_size//2)"""

    #     self.pooling = nn.MaxPool2d((self.pathway_pool_dim,self.pca_pool_dim))
    #     self.drop1 = nn.Dropout(0.25 if args.feature_drop else 0)
    #     if self.pca_compare:
    #         self.pre_linear = nn.Sequential(
    #             nn.Linear(6912, self.head_dim),
    #             nn.ReLU()
    #         )
    #         self.head = nn.Sequential(
    #             nn.Linear(65, self.head_dim),
    #             nn.ReLU(),
    #             nn.Dropout(0.5),
    #             nn.Linear(self.head_dim, 2),
    #             nn.Softmax()
    #         )
    #     else:
    #         if args.only_mrna_pred:
    #             input_dim = args.conv_channel_list[-1]*(146//self.pathway_pool_dim)*(self.pca_dim//self.pca_pool_dim)+(1 if self.args.use_age else 0) 
    #         else:
    #             input_dim = args.conv_channel_list[-1]*(146//self.pathway_pool_dim)*((3*self.pca_dim)//self.pca_pool_dim)+(1 if self.args.use_age else 0) 
    #         self.head = nn.Sequential(
    #             nn.Linear(input_dim, self.head_dim),
    #             nn.ReLU(),
    #             nn.Dropout(0.5),
    #             nn.Linear(self.head_dim, 2),
    #             nn.Softmax()
    #         )