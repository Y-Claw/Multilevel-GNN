import torch
from .gcn_lib.sparse.torch_vertex import GENConv
from .gcn_lib.sparse.torch_nn import norm_layer
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, TopKPooling
import logging
import math
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import numpy as np
import pandas as pd

class PathCNN(nn.Module):

    def __init__(self, args, pca_params=None, pathway_indexs=None):
        super(PathCNN, self).__init__()
        self.pca_compare = args.pca_compare
        self.pca_prelinear = args.pca_prelinear
        self.kernel_size = args.pathcnn_kernel_size
        self.learnable_pca = args.learnable_pca
        self.pca_loss = args.pca_loss
        self.pca_indep_loss = args.pca_indep_loss
        self.pca_dim = args.pca_dim
        self.pathway_pool_dim = args.pathway_pool_dim
        self.pca_pool_dim = args.pca_pool_dim
        self.pathway_indexs = None
        self.mutual_info_mask = args.mutual_info_mask
        self.mutual_info_threshold = args.mutual_info_threshold
        self.pca_loss_coef = args.pca_loss_coef
        self.node_select_threshold = args.node_select_threshold
        self.mutual_neighbors = args.mutual_neighbors
        self.args = args
        self.head_dim = args.head_dim

        if args.learnable_pca:
            self.learnable_pca_params = nn.Parameter(data=torch.rand([24542, self.pca_dim]), requires_grad=True)
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
        self.conv1 = nn.Conv2d(1, 32, self.kernel_size, padding=self.kernel_size//2)
        if args.more_conv:
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, self.kernel_size, padding=self.kernel_size//2),
                nn.ReLU(),
                nn.Conv2d(64, 64, self.kernel_size, padding=self.kernel_size//2),
                nn.ReLU(),
                nn.Conv2d(64, 64, self.kernel_size, padding=self.kernel_size//2),
            )
        else:
            self.conv2 = nn.Conv2d(32, 64, self.kernel_size, padding=self.kernel_size//2)
        self.pooling = nn.MaxPool2d((self.pathway_pool_dim,self.pca_pool_dim))
        self.drop1 = nn.Dropout(0.25)
        if self.pca_compare:
            self.pre_linear = nn.Sequential(
                nn.Linear(6912, 64),
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
            self.head = nn.Sequential(
                nn.Linear(64*(146//self.pathway_pool_dim)*((3*self.pca_dim)//self.pca_pool_dim)+1, self.head_dim),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(self.head_dim, 2),
                nn.Softmax()
            )
        self.init_weight()

    
    def forward(self, input_batch):
        #self.check_pca_data(input_batch)
        if self.learnable_pca:
            raw_data = input_batch.raw_data[:, :, None]
            raw_indice =input_batch.raw_indice[:,:,None].repeat(1,1,self.pca_dim)
            if self.mutual_info_mask:
                pca_result = raw_data * (self.learnable_pca_params * self.info_mask)
            else:
                pca_result = raw_data * self.learnable_pca_params
            b, n, c = pca_result.shape
            x = torch.zeros(b, 146*3, self.pca_dim).to(raw_indice.device).scatter_reduce(1, raw_indice, pca_result, reduce="sum").reshape(-1, 1, 146, self.pca_dim*3)
            #act_pca_result = raw_data * input_batch.pca_component
            #act_x = torch.zeros(b, 146*3, 2).to(raw_indice.device).scatter_reduce(1, raw_indice, act_pca_result.to(pca_result.dtype), reduce="sum").reshape(-1, 1, 146, 6)
        else:
            x = input_batch.pathway_node_attr.reshape(-1, 1, 146, self.pca_dim*3)
        pca_feature = x
        if self.pca_prelinear:
            x = self.pre_linear(x)
        age = input_batch.age
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pooling(x)
        if self.pca_compare:
            #error: pre_linear -> drop
            x = torch.flatten(x, start_dim=1)
            x= self.pre_linear(x)
            #x = self.drop1(x)
        else:
            x = self.drop1(x)
            x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, age[:, None]], dim=-1)
        x = self.head(x)

        return x, pca_feature

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                torch.nn.init.constant_(m.bias.data, 0.0)

    def set_pca_params(self, pca_params, mutual_info_mask):
        if not self.args.learnable_pca:
            return
        idxs = []
        for i in range(len(mutual_info_mask)):
            if mutual_info_mask[i]>0:
                idxs.append(i)
        nn.init.constant_(self.learnable_pca_params.data, 0)
        self.learnable_pca_params.data = pca_params.to(torch.float32)
    
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
        if self.pca_indep_loss and self.args.learnable_pca:
            indep_loss = 0
            count = 0
            for i in range(self.pca_dim-1):
                for j in range(i+1, self.pca_dim):
                    count += 1
                    mul_res = torch.zeros(torch.max(self.pathway_indexs)+1).to(self.pathway_indexs.device).scatter_reduce(0, self.pathway_indexs, self.learnable_pca_params.data[:, i] * self.learnable_pca_params.data[:, j], reduce="sum")
                    len_res = torch.sqrt(torch.zeros(torch.max(self.pathway_indexs)+1).to(self.pathway_indexs.device).scatter_reduce(0, self.pathway_indexs, self.learnable_pca_params.data[:, i]**2, reduce="sum") * torch.zeros(torch.max(self.pathway_indexs)+1).to(self.pathway_indexs.device).scatter_reduce(0, self.pathway_indexs, self.learnable_pca_params.data[:, j]**2, reduce="sum"))
                indep_loss += torch.mean(torch.abs(mul_res / len_res))
            loss += (indep_loss / count)

        return loss
    
    def set_info_mask(self, info_mask):
        self.info_mask = nn.Parameter(data=info_mask, requires_grad=False)

    def generate_mutual_mask(self, x, y, mutual_classif=None):
        x = torch.tensor(x)
        y = torch.tensor(y)
        #mutual_info = mutual_info_regression(x, y)
        if mutual_classif:
            mutual_info = mutual_info_classif(x, y, n_neighbors=self.mutual_neighbors)
        else:
            mutual_info = mutual_info_regression(x, y, n_neighbors=self.mutual_neighbors)
        if self.mutual_info_threshold is None:
            return torch.where(torch.tensor(mutual_info) < np.mean(mutual_info), torch.zeros(mutual_info.shape), torch.ones(mutual_info.shape))[:, None], mutual_info
        else:
            return torch.where(torch.tensor(mutual_info) < self.mutual_info_threshold, torch.zeros(mutual_info.shape), torch.ones(mutual_info.shape))[:, None], mutual_info
    
    def check_pca_data(self, input_batch):
        x = input_batch.pathway_node_attr.reshape(-1, 146, self.pca_dim*3).detach().cpu().numpy()
        all_data, patient_ids = self.get_pca_data()
        batch_patient_ids = input_batch.patientId
        patient_idx = [np.where(patient_ids == batch_patient_id)[0][0] for batch_patient_id in batch_patient_ids]
        match_pca_data = all_data[patient_idx]
        import pdb; pdb.set_trace()

    def get_pca_data(self):
        root_path = "/gpfsdata/home/buaa_yanhongxi/Code/Desease/2022_11_12/PathCNN/data_23_6_23/gbm_23_symbol_bef_rename_allsample1_latedrop_7_19_450first/"
        pca_exp = pd.read_excel(root_path+"pathcnn_lgg_mrna.xlsx", header=None)
        pca_cnv = pd.read_excel(root_path+"pathcnn_lgg_cnv.xlsx", header=None)
        pca_mt = pd.read_excel(root_path+"pathcnn_lgg_mt.xlsx", header=None)
        origin_pca_exp = pd.read_excel("/gpfsdata/home/buaa_yanhongxi/Code/Desease/2022_11_12/PathCNN/data/PCA_EXP.xlsx", header=None)
        origin_pca_cnv = pd.read_excel("/gpfsdata/home/buaa_yanhongxi/Code/Desease/2022_11_12/PathCNN/data/PCA_CNV.xlsx", header=None)
        origin_pca_mt = pd.read_excel("/gpfsdata/home/buaa_yanhongxi/Code/Desease/2022_11_12/PathCNN/data/PCA_MT.xlsx", header=None)
        patient_ids = np.load("/gpfsdata/home/buaa_yanhongxi/Dataset/desease/data_23_6_24/gbm_23_symbol_bef_rename_allsample1_latedrop_7_19_450first/patient_ids.npy", allow_pickle=True)
        
        n = len(pca_exp)  # sample size: number of Pts
        path_n = 146  # number of pathways
        pc = 3  # number of PCs
        origin_pc = 5

        # data creation-EXP
        pca_exp = pca_exp.to_numpy()
        origin_pca_exp = origin_pca_exp.to_numpy()
        exp_data = np.zeros((n, path_n, pc))
        origin_exp_data = np.zeros((n, path_n, origin_pc))
        for i in range(n):
            for j in range(path_n):
                exp_data[i, j, :] = pca_exp[i, j * pc:(j + 1) * pc]
                origin_exp_data[i, j, :] = origin_pca_exp[i, j * origin_pc:(j + 1) * origin_pc]

        # data creation-CNV
        pca_cnv = pca_cnv.to_numpy()
        origin_pca_cnv = origin_pca_cnv.to_numpy()
        cnv_data = np.zeros((n, path_n, pc))
        origin_cnv_data = np.zeros((n, path_n, origin_pc))
        for i in range(n):
            for j in range(path_n):
                cnv_data[i, j, :] = pca_cnv[i, j * pc:(j + 1) * pc]
                origin_cnv_data[i, j, :] = origin_pca_cnv[i, j * origin_pc:(j + 1) * origin_pc]

        # data creation-MT
        pca_mt = pca_mt.to_numpy()
        origin_pca_mt = origin_pca_mt.to_numpy()
        mt_data = np.zeros((n, path_n, pc))
        origin_mt_data = np.zeros((n, path_n, origin_pc))
        for i in range(n):
            for j in range(path_n):
                mt_data[i, j, :] = pca_mt[i, j * pc:(j + 1) * pc]
                origin_mt_data[i, j, :] = origin_pca_mt[i, j * origin_pc:(j + 1) * origin_pc]

        # data merge: mRNA expression, CNV, and MT with a specific number of PCs
        no_pc = 2  # use the first 2PCs among 5 PCs
        all_data = np.zeros((n, path_n, no_pc * 3))
        origin_all_data = np.zeros((n, path_n, no_pc * 3))
        for i in range(n):
                all_data[i, :, :] = np.concatenate((exp_data[i, :, 0:no_pc], cnv_data[i, :, 0:no_pc], mt_data[i, :, 0:no_pc]),axis=1)
                origin_all_data[i, :, :] = np.concatenate((origin_exp_data[i, :, 0:no_pc], origin_cnv_data[i, :, 0:no_pc], origin_mt_data[i, :, 0:no_pc]),axis=1)
        
        return all_data, patient_ids