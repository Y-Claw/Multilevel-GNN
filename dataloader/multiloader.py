''' 
Repcount data loader from fixed frames file(.npz) which will be uploaded soon.
if you don't pre-process the data file,for example,your raw file is .mp4,
you can use the *RepCountA_raw_Loader.py*(slowly).
or
you can use 'tools.video2npz.py' to transform .mp4 tp .npz
'''
import csv
import os
import os.path as osp
import numpy as np
import math
import random
import time
from multiprocessing import Pool
from tqdm import tqdm
import logging

from torch.utils.data import Dataset, DataLoader
import torch
from concurrent.futures import ThreadPoolExecutor
import torchvision.transforms as transforms
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
import pandas as pd
import pickle
from torch_geometric.data import Data

from utils.knnie import kraskov_mi

class MyData(Dataset):

    def __init__(self, raw_mrna_path, raw_cnv_path, raw_methylation_path, node_path, edge_path, kegg_path, clinical_path, args=None):
        self.raw_mrna_path = raw_mrna_path
        self.raw_cnv_path = raw_cnv_path
        self.raw_methylation_path = raw_methylation_path
        self.node_path = node_path
        self.edge_path = edge_path
        self.grn_edge_path = args.grn_edge_path
        self.kegg_path = kegg_path
        self.clinical_path = clinical_path
        self.make_graph = args.make_graph
        self.neighbor_range =args.neighborhood
        self.pretain_only_pathway_edge = args.pretain_only_pathway_edge
        self.weight_power = args.weight_power
        self.pathway_path = args.pathway_path
        self.soft_label = args.soft_label
        self.args = args
        #'mrna', 'cnv', 'methylation'
        self.omics_types = ['mrna', 'cnv', 'methylation']
        self.data_means = None
        logging.info("omics_types {}".format(', '.join(self.omics_types)))
        self.kegg = pd.read_csv(self.kegg_path)
        with open(self.pathway_path, 'rb') as f:
            self.pathway_codes = pickle.load(f)
        print('init graph')
        if args.edge_type == 'ppi':
            self.edges, self.edge_attrs = self.init_graph()
        elif args.edge_type == 'grnboost2':
            self.edges, self.edge_attrs = self.init_graph_grnboost2()
        elif args.edge_type == 'merge':
            self.edges, self.edge_attrs = self.init_graph()
            self.raw_grn_edges, self.raw_grn_edge_attrs = self.init_graph_grnboost2()

        print('init data')
        self.raw_multi_omics_data = self.init_data()
        if args.edge_type == 'merge':
            self.grn_edges, self.grn_edge_attrs = self.process_grn_edge(self.raw_grn_edges, self.raw_grn_edge_attrs)
        print('predefine_data')
        self.parallel_predefine_data()
        #import pdb; pdb.set_trace()
        #logging.info("used gene number for mrna cnv mt: {}".format((self.__getitem__(0).x.reshape(-1, 4)[:, :-1]!=0).sum(0)))
        self.valid_entrzid = {}

    def __getitem__(self, inx):
        patientId = self.patient_list[inx]
        data = self.data_dict[patientId]
        data.x = data.x.reshape(-1,1)
        raw_data = self.raw_datas[patientId]
        raw_indice = self.raw_indice[patientId]
        data.raw_indice = torch.tensor(raw_indice).unsqueeze(0)
        if "gnn" not in self.args.model:
            data.raw_data = torch.tensor(raw_data).unsqueeze(0)
            data.pca_component = self.pca_components.unsqueeze(0)
            data.data_mean = self.data_means.unsqueeze(0)
        if self.args.edge_type == 'merge' and not isinstance(data.edge_index, list):
            data.edge_index = [data.edge_index, self.grn_edges]
            data.edge_attr = [data.edge_attr, self.grn_edge_attrs]
        if self.args.random_variation_aug:
            data.variation_aug = self.random_variation_aug(data)
        elif self.args.random_mask_aug:
            data.random_mask_aug = self.random_mask_aug(data)
        return data

    def __len__(self):
        """:return the number of video """
        return len(self.patient_list)
    
    def init_data(self):
        self.clinical_data = pickle.load(open(self.clinical_path, 'rb'))
        self.os_month = self.clinical_data["survive_time"]
        self.survive_state = self.clinical_data["survive_state"]
        self.age = self.clinical_data["age"]
        if self.args.global_edge == 'onehot':
            self.args.pathway_edge_num = len(self.kegg)

        self.raw_cnv = raw_cnv = self.drop_na_line(pd.read_csv(self.raw_cnv_path, index_col=0))
        self.raw_mrna = raw_mrna = self.drop_na_line(pd.read_csv(self.raw_mrna_path, index_col=0))
        if self.args.zscore_mrna:
            self.raw_mrna = raw_mrna = (raw_mrna-raw_mrna.mean())/raw_mrna.std()

        self.raw_methylation = raw_methylation = self.drop_na_line(pd.read_csv(self.raw_methylation_path, index_col=0))

        if self.args.add_hat:
            #self.raw_cnv = raw_cnv = self.add_hat(raw_cnv, self.args.add_hat_sigma, self.args.add_hat_percent)
            self.raw_mrna = raw_mrna = self.add_hat(raw_mrna, self.args.add_hat_sigma, self.args.add_hat_percent, direction="higher")
            self.raw_mrna = raw_mrna = self.add_hat(raw_mrna, self.args.add_hat_sigma, 1-self.args.add_hat_percent, direction="lower")
            #self.raw_methylation = raw_methylation = self.add_hat(raw_methylation, self.args.add_hat_sigma, self.args.add_hat_percent)

        if isinstance(self.raw_cnv.keys()[0], str):
            rename_dict = dict([(symbol, str(self.kegg[self.kegg['Symbol'] == symbol]['Entrezid'].unique()[0])) for symbol in self.kegg['Symbol'].unique()])
            self.raw_cnv = raw_cnv = raw_cnv.rename(columns=rename_dict)#.round(5)
            self.raw_mrna = raw_mrna = raw_mrna.rename(columns=rename_dict)#.round(5)
            self.raw_methylation = raw_methylation = raw_methylation.rename(columns=rename_dict)#.round(5)
        

        if self.args.reverse_mt:
            self.raw_methylation = raw_methylation = -raw_methylation

        patient_list = (self.raw_methylation.index &  self.raw_cnv.index & self.raw_mrna.index & self.os_month.keys()).tolist()
        self.os_month = dict([(patient_id, self.os_month[patient_id]) for patient_id in patient_list])

        rename_raw_cnv = raw_cnv.rename(columns=dict(zip(raw_cnv.columns, map(lambda x: "cnv_"+x, raw_cnv.columns))))
        rename_raw_mrna = raw_mrna.rename(columns=dict(zip(raw_mrna.columns, map(lambda x: "mrna_"+x, raw_mrna.columns))))
        rename_raw_methylation = raw_methylation.rename(columns=dict(zip(raw_methylation.columns, map(lambda x: "methylation_"+x, raw_methylation.columns))))
        self.named_raw_cnv = rename_raw_cnv

        raw_multi_omics_data = rename_raw_cnv.join(rename_raw_mrna).join(rename_raw_methylation)
        if self.args.z_score:
            raw_multi_omics_data = (raw_multi_omics_data-raw_multi_omics_data.mean())/raw_multi_omics_data.std()
        elif self.args.z_mean:
            raw_multi_omics_data = (raw_multi_omics_data-raw_multi_omics_data.mean())

        if self.args.debug:
            self.patient_list = [patient_id for patient_id in raw_multi_omics_data.index.unique().tolist() if (patient_id in self.os_month.keys() and patient_id in self.age and not np.isnan(self.age[patient_id]) and (self.os_month[patient_id] > self.args.risk_threshold or (self.survive_state[patient_id] == 1)))][:25]
        else:
            self.patient_list = [patient_id for patient_id in raw_multi_omics_data.index.unique().tolist() if (patient_id in self.os_month.keys() and not np.isnan(self.os_month[patient_id]) and (not self.args.use_age or (patient_id in self.age and not np.isnan(self.age[patient_id]))) and (self.os_month[patient_id] > self.args.risk_threshold or (self.survive_state[patient_id] == 1)))]
        """
            347
            (Pdb) len(self.patient_list)
            289
        """

        if not self.args.lag_pca:
            #patient_list = (self.raw_methylation.index &  self.raw_cnv.index & self.raw_mrna.index & self.os_month.keys()).tolist()
            self.raw_mrna = raw_mrna = raw_mrna.loc[patient_list]
            self.raw_cnv = raw_cnv = raw_cnv.loc[patient_list]
            self.raw_methylation = raw_methylation = raw_methylation.loc[patient_list]
            #self.os_month = dict([(patient_id, self.os_month[patient_id]) for patient_id in patient_list])
            self.pca_result, self.raw_datas, self.raw_indice, self.pca_components, self.all_indice, self.tf_token = self.prepare_pca_result(raw_mrna, raw_cnv, raw_methylation, self.os_month)

        if self.args.lag_pca:
            self.raw_mrna = raw_mrna = raw_mrna.loc[self.patient_list].dropna(axis=1,how='any')
            #self.raw_cnv = raw_cnv = raw_cnv.loc[self.patient_list].replace(np.nan, 0).dropna(axis=1,how='any') # for lgg drop all na will generate a 0 line dataframe
            self.raw_cnv = raw_cnv = raw_cnv.loc[self.patient_list].loc[raw_cnv.loc[self.patient_list].isnull().sum(axis=1) <= 46].dropna(axis=1,how='any')
            self.raw_methylation = raw_methylation = raw_methylation.loc[self.patient_list].dropna(axis=1,how='any')
            if self.args.align_data:
                columns =  self.raw_methylation.columns &  self.raw_cnv.columns & self.raw_mrna.columns
                self.patient_list = (self.raw_methylation.index &  self.raw_cnv.index & self.raw_mrna.index).tolist()
                self.raw_mrna = raw_mrna = raw_mrna[columns].loc[self.patient_list]
                self.raw_cnv = raw_cnv = raw_cnv[columns].loc[self.patient_list]
                self.raw_methylation = raw_methylation = raw_methylation[columns].loc[self.patient_list]
            self.pca_result, self.raw_datas, self.raw_indice, self.pca_components, self.all_indice, self.tf_token = self.prepare_pca_result(raw_mrna, raw_cnv, raw_methylation, self.os_month)
        
        return raw_multi_omics_data

    def init_graph(self):
        raw_node = pd.read_csv(self.node_path)
        raw_edge = pd.read_csv(self.edge_path).fillna(0)
        id_2_entrezid = {}
        edges = {}
        reverse_edges = {}
        edge_attrs = {}
        kegg_symbol = self.kegg['Symbol'].tolist()
        for i, row in raw_node.iterrows():
            stringid = row['@id']
            symbol = row['query term']
            if symbol in kegg_symbol and row['stringdb::node type'] == 'protein':
                id_2_entrezid[stringid.replace('stringdb:', '')] = self.kegg[self.kegg['Symbol'] == symbol]['Entrezid'].unique()[0]
        self.node_size = len(id_2_entrezid)
        self.node_map = dict(zip(id_2_entrezid.values(), range(self.node_size)))

        if not self.args.pca_only:
            edge_count = 0
            for i, row in raw_edge.iterrows():
                if ' (pp) ' not in row['name']:
                    continue
                source_node, end_node = row['name'].split(' (pp) ')
                #value = row['stringdb::coexpression']
                if self.args.use_column is None:
                    value = row[['stringdb::coexpression', 'stringdb::cooccurrence', 'stringdb::databases', 'stringdb::experiments', 'stringdb::fusion','stringdb::neighborhood', 'stringdb::score']]
                else:
                    value = row[self.args.use_column]
                #if source_node not in id_2_entrezid or end_node not in id_2_entrezid or np.isnan(value):
                if source_node not in id_2_entrezid or end_node not in id_2_entrezid or (self.args.use_column is not None and (np.isnan(value) or value == 0)):
                    continue
                source_node_id, end_node_id = id_2_entrezid[source_node], id_2_entrezid[end_node]
                if self.pretain_only_pathway_edge and not self.in_same_pathway(source_node_id, end_node_id, self.kegg):
                    continue
                edge_count += 1
                edges.setdefault(source_node_id, [])
                edge_attrs.setdefault(source_node_id, [])
                edges[source_node_id].append([source_node_id, end_node_id])
                edge_attrs[source_node_id].append(value)
            print("total ppi edge num:", edge_count)
            logging.info("total ppi edge num: {}".format(edge_count))
        return edges, edge_attrs
    
    def init_graph_grnboost2(self):
        raw_node = pd.read_csv(self.node_path)
        if self.args.edge_type == 'merge':
            raw_edge = pd.read_csv(self.grn_edge_path, sep='\t').fillna(0)
        else:
            raw_edge = pd.read_csv(self.edge_path, sep='\t').fillna(0)
        id_2_entrezid = {}
        edges = {}
        reverse_edges = {}
        edge_attrs = {}
        all_pathway = pd.read_excel('./data/ordered_pathway_146_2pc.xlsx', header=None)[0].to_list()
        file = open("./data/c2.cp.kegg.v5.2.symbols.gmt", "r")
        entrez_file = open("./data/c2.cp.kegg.v5.2.entrez.gmt", "r")
        res = file.readlines()
        entrez_res = entrez_file.readlines()
        kegg_symbol = self.kegg['Symbol'].tolist()
        symbols_res = res = [x.strip().split('\t') for x in res]
        pathway_gene_symbols = dict([[x[0].replace('KEGG_', ""), x[2:]] for x in symbols_res])
        query_gene_symbols = set({})
        for pathway_name in all_pathway:
            symbols = pathway_gene_symbols[pathway_name]
            for symbol in symbols:
                query_gene_symbols.add(symbol)
        for symbol in query_gene_symbols:
            if symbol in kegg_symbol:
                id_2_entrezid[symbol] = self.kegg[self.kegg['Symbol'] == symbol]['Entrezid'].unique()[0]
        node_size = len(id_2_entrezid)
        #node_map = dict(zip(id_2_entrezid.values(), range(node_size)))
        #import pdb; pdb.set_trace()
        node_map = dict(zip(sorted(id_2_entrezid.values()), range(node_size)))

        if self.args.edge_type != 'merge':
            self.node_size = node_size
            self.node_map = node_map    

        if not self.args.pca_only:
            edge_count = 0
            for i, (source_node, end_node, value) in raw_edge.iterrows():
                if self.args.grn_edge_select_threshold is not None and value < self.args.grn_edge_select_threshold:
                    continue
                #if source_node not in id_2_entrezid or end_node not in id_2_entrezid or np.isnan(value):
                if source_node not in id_2_entrezid or end_node not in id_2_entrezid or (self.args.use_column is not None and (np.isnan(value) or value == 0)):
                    continue
                source_node_id, end_node_id = id_2_entrezid[source_node], id_2_entrezid[end_node]
                if self.pretain_only_pathway_edge and not self.in_same_pathway(source_node_id, end_node_id, self.kegg):
                    continue
                edge_count += 1
                edges.setdefault(source_node_id, [])
                edge_attrs.setdefault(source_node_id, [])
                edges[source_node_id].append([source_node_id, end_node_id])
                edge_attrs[source_node_id].append(value)
            print("total grn edge num:", edge_count)
            logging.info("total grn edge num:{}".format(edge_count))

        return edges, edge_attrs

    def process_grn_edge(self, edges, edge_attrs):
        idx_map = self.get_index_map()
        processed_edges = []
        processed_edge_attrs = []
        for node_id in edges.keys():
            for p_edge, attr in zip(edges[node_id], edge_attrs[node_id]):
                if p_edge[1] in self.node_map.keys():
                    processed_edges.append([idx_map[self.node_map[p_edge[0]]], idx_map[self.node_map[p_edge[1]]]])
                    processed_edge_attrs.append([attr])
                    processed_edges.append([idx_map[self.node_map[p_edge[1]]], idx_map[self.node_map[p_edge[0]]]])
                    processed_edge_attrs.append([attr])
        processed_edges = torch.tensor(processed_edges).transpose(0,1).type(torch.int64)
        processed_edge_attrs = torch.tensor(processed_edge_attrs).type(torch.float32).reshape((len(processed_edge_attrs), -1))
        return processed_edges, processed_edge_attrs

    def parallel_predefine_data(self, num_workers=8):
        self.data_dict = {}
        pool = Pool()
        res = []
        patient_num_each = math.ceil(len(self.patient_list)/num_workers)
        start = time.time()
        print('get neighbor')
        neighbors = self.get_neighbors(self.neighbor_range)
        print('parrallel predefine data')
        for i in range(num_workers):
            res.append(pool.apply_async(predefine_data, [self, list(range(i*patient_num_each, min((i+1)*patient_num_each, len(self.patient_list)))), self.args.node_embedding, 1000, neighbors]))
        for r in res:
            self.data_dict.update(r.get())
        #predefine_data(self, range(10), self.args.node_embedding, 1000, neighbors)
        pool.close()
        pool.join()
        """import networkx as nx
        G = nx.Graph()
        G.add_edges_from(self.data_dict['TCGA-02-0001'].edge_index.permute(1,0).numpy())
        t = [i for i in nx.connected_components(G)]
        import pdb
        pdb.set_trace()"""
        end = time.time()
        total_time = end - start
        logging.info('avg node num {}'.format(np.mean([len(d.x) for d in self.data_dict.values()])))
        print('avg node num {}'.format(np.mean([len(d.x) for d in self.data_dict.values()])))
        print('Data size: {}'.format(len(self.data_dict)))
        print('Data process time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))
        #logging.info('Data process time: {}'.format(time.strftime('%H:%M:%S', time.gmtime(total_time))))
    
    def get_weight_balance(self, indexs):
        label_nums = [0,0]
        for inx in indexs:
            y = int(self.data_dict[self.patient_list[inx]].y[1] > 0.5)
            label_nums[y] += 1
        return torch.repeat_interleave(((max(label_nums) / torch.tensor(label_nums)).float() ** self.weight_power).unsqueeze(dim=0), repeats=self.args.batch_size, dim=0)
    
    def get_node_num(self):
        return self.node_size
    
    def get_labels(self):
        labels = []
        for data in self.data_dict.values():
            #labels.append(data.y.item())
            labels.append(int(data.y[1].item() > 0.5))
        
        return np.array(labels)
    
    def get_tf_token(self):    
        return self.tf_token
        
    def get_neighbors(self, neighbor_range):
        neighbors = {}
        for node_id in self.edges.keys():
            current_hop = 1
            current_nodes = set([node_id])
            next_current_nodes = set([])
            nodes_in_neighbors = set([node_id])
            while current_hop <= neighbor_range:
                for current_node in current_nodes:
                    if current_node in self.edges:
                        next_current_nodes.update(list(l[1] for l in self.edges[current_node]))
                nodes_in_neighbors.update(next_current_nodes)
                current_nodes = next_current_nodes
                next_current_nodes = set([])
                current_hop += 1
            neighbors[node_id] = nodes_in_neighbors
            for node_in_neighbor in nodes_in_neighbors:
                neighbors.setdefault(node_in_neighbor, set({}))
                neighbors[node_in_neighbor].add(node_id)
        return neighbors
    
    def in_same_pathway(self, source_node_id, end_node_id, kegg):
        if source_node_id not in kegg['Entrezid'] or end_node_id not in kegg['Entrezid']:
            return False
        source_node_pathways = kegg[kegg['Entrezid'] == source_node_id]["PathwayID"].unique()
        end_node_pathways = kegg[kegg['Entrezid'] == end_node_id]["PathwayID"].unique()
        for source_node_pathway in source_node_pathways:
            if source_node_pathway in self.pathway_codes and source_node_pathway in self.pathway_codes and source_node_pathway in end_node_pathways:
                return True
        return False
    
    def get_all_pathways(self):
        return self.pathway_codes
    
    def prepare_pca_result(self, raw_mrna, raw_cnv, raw_methylation, os_month, hop=0, mutual_info_mask=None):
        if self.args.lag_pca:
            raw_mrna = raw_mrna.loc[self.patient_list]
            raw_cnv = raw_cnv.loc[self.patient_list]
            raw_methylation = raw_methylation.loc[self.patient_list]
        pca_result = {}
        raw_datas = {}
        raw_indice = {}
        tf_token = []
        raw_data_list = []
        pca_components = []
        all_pathway = pd.read_excel('./data/ordered_pathway_146_2pc.xlsx', header=None)[0].to_list()
        slice_count = 0
        start = 0
        mask_pathway = 0
        data_means = []
        pathway_matrix = []
        os_month_list = list(os_month.keys())
        #import pdb; pdb.set_trace()
        precise_idx = [os_month_list.index(patient_id) for patient_id in self.patient_list if patient_id in os_month_list]
        for pathway in all_pathway:
            if self.args.lag_pca:
                all_data = np.zeros((len(raw_mrna.index),0))
                #pathway_matrix = np.zeros((len(raw_mrna.index),0))
            else:
                all_data = np.zeros((len(os_month.keys()),0))
                #pathway_matrix = np.zeros((len(os_month.keys()),0))
                
            entrezids = self.kegg[self.kegg['PathwayID'] == pathway].Entrezid.unique()
            if len(entrezids) < 5:
                miss_pathway.append(pathway)
                continue
            #.dropna(axis=1,how='any')
            #mrna_data = raw_mrna[np.intersect1d(entrezids, raw_mrna.keys().map(int)).astype(str)].dropna(axis=1,how='any').loc[list(os_month.keys())]
            #cnv_data =raw_cnv[np.intersect1d(entrezids, raw_cnv.keys().map(int)).astype(str)].dropna(axis=1,how='any').loc[list(os_month.keys())]
            #mt_data = raw_methylation[np.intersect1d(entrezids, raw_methylation.keys().map(int)).astype(str)].dropna(axis=1,how='any').loc[list(os_month.keys())]
            #import pdb;pdb.set_trace()
            if self.args.lag_pca:
                mrna_data = raw_mrna[np.intersect1d(entrezids, raw_mrna.keys().map(int)).astype(str)].dropna(axis=1,how='any')
                cnv_data = raw_cnv[np.intersect1d(entrezids, raw_cnv.keys().map(int)).astype(str)].dropna(axis=1,how='any')
                mt_data = raw_methylation[np.intersect1d(entrezids, raw_methylation.keys().map(int)).astype(str)].dropna(axis=1,how='any')
            else:
                mrna_data = raw_mrna[np.intersect1d(entrezids, raw_mrna.keys().map(int)).astype(str)].loc[list(os_month.keys())].dropna(axis=1,how='any')
                cnv_data = raw_cnv[np.intersect1d(entrezids, raw_cnv.keys().map(int)).astype(str)].loc[list(os_month.keys())].dropna(axis=1,how='any')
                mt_data = raw_methylation[np.intersect1d(entrezids, raw_methylation.keys().map(int)).astype(str)].loc[list(os_month.keys())].dropna(axis=1,how='any')
            if self.args.z_score:
                raw_mrna = (raw_mrna-raw_mrna.mean())/raw_mrna.std()
                raw_cnv = (raw_cnv-raw_cnv.mean())/raw_cnv.std()
                raw_methylation = (raw_methylation-raw_methylation.mean())/raw_methylation.std()

            pca_datas = []          
            
            for num_data, data in enumerate([mrna_data, cnv_data, mt_data]):
                data_mean = data.mean(0)
                data_means.extend(data_mean.to_list())
                selected_entrez = []
                pca_mutual_select_index = []
                for i, patient_id in enumerate(mrna_data.index):
                    raw_datas.setdefault(patient_id, [])
                    raw_indice.setdefault(patient_id, [])
                    raw_data_list.append([])
                    raw_datas[patient_id].extend((data.loc[patient_id] - data_mean).to_list())
                    raw_indice[patient_id].extend([slice_count]*len(data.loc[patient_id]))
                    raw_data_list[-1].extend((data.loc[patient_id] - data_mean).to_list())
                if str(num_data) in self.args.remain_tf_nums:
                    tf_token.extend([int(token) in self.edges for token in data.keys()])
                else:
                    tf_token.extend([False for token in data.keys()])

                if mutual_info_mask is not None:
                    for i, entrezid in enumerate(data.columns):
                        if mutual_info_mask[start + i][0] > 0:
                            selected_entrez.append(entrezid)
                            pca_mutual_select_index.append(i)
                    start += len(data.columns)
                else:
                    selected_entrez = data.columns.to_list()
                slice_count += 1
                
                for entrezid in data.columns:
                    pass

                try:
                    if mutual_info_mask is None or len(selected_entrez) < self.args.pca_sim_dim:
                        tmp_pca_dim = min(len(selected_entrez), self.args.pca_sim_dim)
                        pca = PCA(n_components=tmp_pca_dim, svd_solver='full')
                        pca = pca.fit(data.to_numpy())
                        # np.dot((data.to_numpy()[0]-pca.mean_)[None,:], pca.components_.T)[0]
                        pca_data = pca.transform(data)
                        
                        pad_len = self.args.pca_sim_dim - len(selected_entrez)
                        if pad_len > 0:
                            pca_data = np.concatenate([pca_data, np.zeros((len(pca_data), pad_len))], axis=-1)
                        pca_datas.append(pca_data)
                        if mutual_info_mask is not None:
                            if self.args.drop_irr_pathway:
                                tmp_components = np.zeros(pca.components_[:, pca_mutual_select_index].shape)
                            else:
                                tmp_components = pca.components_[:, pca_mutual_select_index]
                            if len(selected_entrez) < self.args.pca_sim_dim:
                                tmp_components = np.concatenate([tmp_components, np.zeros((pad_len, len(pca_mutual_select_index)))], axis=0)
                            pca_components.append(tmp_components[:self.args.pca_dim, :])
                        else:
                            if len(selected_entrez) < self.args.pca_sim_dim:
                                tmp_components = np.concatenate([pca.components_, np.zeros((pad_len, len(selected_entrez)))], axis=0)
                            else:
                                tmp_components = pca.components_
                            pca_components.append(tmp_components)
                    else:
                        pca = PCA(n_components=self.args.pca_sim_dim, svd_solver='full')
                        pca = pca.fit(data[selected_entrez].to_numpy())
                        # np.dot((data.to_numpy()[0]-pca.mean_)[None,:], pca.components_.T)[0]
                        pca_datas.append(pca.transform(data[selected_entrez]))
                        component = pca.components_ if not self.args.mean_pca_init else (pca.components_ / np.mean(np.abs(pca.components_)) * self.args.pca_mean_value)
                        pca_components.append(pca.components_[:self.args.pca_dim, :] if not self.args.mean_pca_init else pca.components_[:self.args.pca_dim, :])
                        mask_pathway += 1
                except:
                    import pdb
                    pdb.set_trace()

            for i in range(3):
                tmp_matrix = pca_datas[i][:,:self.args.pca_sim_dim]
                if self.args.precise_order:
                    tmp_matrix = tmp_matrix[precise_idx, :]
                pathway_matrix.append(tmp_matrix)    
                all_data = np.concatenate([all_data, pca_datas[i][:,:self.args.pca_dim]], axis=-1)
            
            for i, patient_id in enumerate(mrna_data.index):
                pca_result.setdefault(patient_id, {})
                pca_result[patient_id][pathway] = all_data[i]
        #pdb.set_trace()
        if self.data_means is None:
            self.data_means = torch.tensor(data_means)
        if mutual_info_mask is not None:
            # mask_pathway: 429/438  #347 432 / 438
            print("mask_pathway:", mask_pathway)
        if self.args.reorder_pathway and (self.args.selected_similarity or mutual_info_mask is None):
            cat_pathway_matrix = np.stack(pathway_matrix).reshape(146, -1)
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
            self.reorder_idxs = reorder_idxs

        return pca_result, raw_datas, raw_indice, torch.from_numpy(np.concatenate(pca_components, axis=1)).transpose(0,1), torch.tensor(raw_indice[patient_id]), tf_token
    
    def get_data_by_indice(self, indice):
        all_data = []
        all_y = []
        for inx in indice:
            patientId = self.patient_list[inx]
            data = self.data_dict[patientId]
            raw_data = self.raw_datas[patientId]
            raw_indice = self.raw_indice[patientId]
            all_data.append(raw_data)
            all_y.append(data.y[1])
        return all_data, all_y, 

    def get_survival_data_by_indice(self, indice):
        all_data = []
        all_y = []
        for inx in indice:
            patientId = self.patient_list[inx]
            data = self.data_dict[patientId]
            raw_data = self.raw_datas[patientId]
            raw_indice = self.raw_indice[patientId]
            all_data.append(raw_data)
            all_y.append(data.y[1])
        return all_data, all_y,  

    def generate_mutual_info_feature_mask(self, x, y):
        pass

    def data_analysis(self, data):
        pass
    
    def get_reorder_idxs(self):
        if 'reorder_idxs' in dir(self):
            return self.reorder_idxs
        else:
            return list(range(146))

    def recalculate_pca_bo_selected_gene(self, mutual_info_mask):
        #raw_cnv = pd.read_csv(self.raw_cnv_path, index_col=0)
        #raw_mrna = pd.read_csv(self.raw_mrna_path, index_col=0)
        #raw_methylation = pd.read_csv(self.raw_methylation_path, index_col=0)
        self.pca_result, raw_datas, raw_indice, self.pca_components, all_indice, tf_token= self.prepare_pca_result(self.raw_mrna, self.raw_cnv, self.raw_methylation, self.os_month, mutual_info_mask=mutual_info_mask)
        all_pathway = pd.read_excel('./data/ordered_pathway_146_2pc.xlsx', header=None)[0].to_list()
        for patientId in self.patient_list:
            data = self.data_dict[patientId]
            pathway_node_attr = []
            for pathway in all_pathway:
                pathway_node_attr.append(self.pca_result[patientId][pathway])
            data.pathway_node_attr = torch.tensor(pathway_node_attr).unsqueeze(0).to(torch.float32)
    
    def recalculate_edge_bo_selected_gene(self, mutual_info_mask, indice=None):
        start = 0
        selected_entrzid_sets = [set({}), set({}), set({})]
        gene_pca_match = []
        gene_std_value = []
        all_pathway = pd.read_excel('./data/ordered_pathway_146_2pc.xlsx', header=None)[0].to_list()
        missing_valid_gene_in_graph = []
        total_valid_gene_num = 0
        for pathway in all_pathway:
            entrezids = self.kegg[self.kegg['PathwayID'] == pathway].Entrezid.unique()
            if self.args.lag_pca:
                mrna_data = self.raw_mrna[np.intersect1d(entrezids, self.raw_mrna.keys().map(int)).astype(str)].dropna(axis=1,how='any')
                cnv_data = self.raw_cnv[np.intersect1d(entrezids, self.raw_cnv.keys().map(int)).astype(str)].dropna(axis=1,how='any')
                mt_data = self.raw_methylation[np.intersect1d(entrezids, self.raw_methylation.keys().map(int)).astype(str)].dropna(axis=1,how='any')
            else:
                mrna_data = self.raw_mrna[np.intersect1d(entrezids, self.raw_mrna.keys().map(int)).astype(str)].loc[list(self.os_month.keys())].dropna(axis=1,how='any')
                cnv_data = self.raw_cnv[np.intersect1d(entrezids, self.raw_cnv.keys().map(int)).astype(str)].loc[list(self.os_month.keys())].dropna(axis=1,how='any')
                mt_data = self.raw_methylation[np.intersect1d(entrezids, self.raw_methylation.keys().map(int)).astype(str)].loc[list(self.os_month.keys())].dropna(axis=1,how='any')
            assert self.raw_mrna[np.intersect1d(entrezids, self.raw_mrna.keys().map(int)).astype(str)].dropna(axis=1,how='any').loc[list(self.os_month.keys())].shape[1] == self.raw_mrna[np.intersect1d(entrezids, self.raw_mrna.keys().map(int)).astype(str)].loc[list(self.os_month.keys())].dropna(axis=1,how='any').shape[1]
            assert self.raw_cnv[np.intersect1d(entrezids, self.raw_cnv.keys().map(int)).astype(str)].dropna(axis=1,how='any').loc[list(self.os_month.keys())].shape[1] == self.raw_cnv[np.intersect1d(entrezids, self.raw_cnv.keys().map(int)).astype(str)].loc[list(self.os_month.keys())].dropna(axis=1,how='any').shape[1]
            assert self.raw_methylation[np.intersect1d(entrezids, self.raw_methylation.keys().map(int)).astype(str)].dropna(axis=1,how='any').loc[list(self.os_month.keys())].shape[1] == self.raw_methylation[np.intersect1d(entrezids, self.raw_methylation.keys().map(int)).astype(str)].loc[list(self.os_month.keys())].dropna(axis=1,how='any').shape[1]

            for num_data, data in enumerate([mrna_data, cnv_data, mt_data]):
                if mutual_info_mask is not None:
                    for i, entrezid in enumerate(data.columns):
                        entrezid_int = int(entrezid)
                        if mutual_info_mask[start + i][0] > 0:
                            total_valid_gene_num += 1
                            selected_entrzid_sets[num_data].add(entrezid_int)
                        if entrezid_int in self.node_map:
                            node_idx = self.node_map[entrezid_int]
                            # data.x.reshape(-1,1)[gene_pca_match] -> pca data
                            gene_pca_match.append(3*node_idx+num_data)
                            gene_std_value.append(data[entrezid].std())
                        else:
                            gene_pca_match.append(-1)
                            missing_valid_gene_in_graph.append(entrezid_int)
                    start += len(data.columns)
        self.gene_std_value = torch.tensor(gene_std_value)
        #import pdb;pdb.set_trace()
        print("total {} valid gene, missing {} valid gene in graph".format(total_valid_gene_num, len(missing_valid_gene_in_graph)))
        logging.info("total {} valid gene, missing {} valid gene in graph".format(total_valid_gene_num, len(missing_valid_gene_in_graph)))
        logging.info("unique mrna {} , unique cnv {}, unique mt {}".format(len(selected_entrzid_sets[0]), len(selected_entrzid_sets[1]), len(selected_entrzid_sets[2])))
        logging.info("overlap mrna-cnv {} , overlap mrna-mt {}".format(len(selected_entrzid_sets[0] & selected_entrzid_sets[1]), len(selected_entrzid_sets[0] & selected_entrzid_sets[2])))
        idx_map = self.get_index_map()

        edges = []
        edge_attrs = []
        train_patients = []
        y = []
        if indice is not None:
            for i in indice:
                train_patients.append(self.patient_list[i])
            for patientId in train_patients:
                y.append(self.data_dict[patientId].y[1].item())
        
        remain_edge = 0
        del_edge = 0
        self.mutual_infos = {}
        for num_data, selected_entrzid_set in enumerate(selected_entrzid_sets):
            for entrzid in selected_entrzid_set:
                if str(num_data) in self.args.mute_edge:
                    continue
                if entrzid not in self.edges:
                    continue
                for p_edge, attr in zip(self.edges[entrzid], self.edge_attrs[entrzid]):
                    if p_edge[1] in selected_entrzid_set:
                        if self.valid_pca_mutual_info(p_edge, train_patients, y, num_data):
                            remain_edge += 1
                            edges.append([3*idx_map[self.node_map[p_edge[0]]]+num_data, 3*idx_map[self.node_map[p_edge[1]]]+num_data])
                            edge_attrs.append([attr])
                            if self.args.bidir_edge:
                                edges.append([3*idx_map[self.node_map[p_edge[1]]]+num_data, 3*idx_map[self.node_map[p_edge[0]]]+num_data])
                                edge_attrs.append([attr])
                        else:
                            del_edge += 1
        print("ppi remain edge: {}, del_edge: {}".format(remain_edge, del_edge))
        logging.info("ppi remain edge: {}, del_edge: {}".format(remain_edge, del_edge))
        
        cross_omix_edge_num = 0
        for mrna_entrzid in selected_entrzid_sets[0]:
            if mrna_entrzid not in self.node_map:
                continue
            if self.args.construct_cnv_mrna_edge and mrna_entrzid in selected_entrzid_sets[1]:
                cross_omix_edge_num += 1
                edges.append([3*idx_map[self.node_map[mrna_entrzid]]+1, 3*idx_map[self.node_map[mrna_entrzid]]+0])
                edge_attrs.append([1])
            if self.args.construct_mt_mrna_edge and mrna_entrzid in selected_entrzid_sets[2]:
                cross_omix_edge_num += 1
                edges.append([3*idx_map[self.node_map[mrna_entrzid]]+2, 3*idx_map[self.node_map[mrna_entrzid]]+0])
                edge_attrs.append([1 if not self.args.reverse_mt_attr else -1])
            if self.args.construct_mrna_cnv_edge and mrna_entrzid in selected_entrzid_sets[1]:
                cross_omix_edge_num += 1
                edges.append([3*idx_map[self.node_map[mrna_entrzid]]+0, 3*idx_map[self.node_map[mrna_entrzid]]+1])
                edge_attrs.append([1])
            if self.args.construct_mrna_mt_edge and mrna_entrzid in selected_entrzid_sets[2]:
                cross_omix_edge_num += 1
                edges.append([3*idx_map[self.node_map[mrna_entrzid]]+0, 3*idx_map[self.node_map[mrna_entrzid]]+2])
                edge_attrs.append([1 if not self.args.reverse_mt_attr else -1])

        print("cross omix edge: {}".format(cross_omix_edge_num))
        logging.info("cross omix edge: {}".format(cross_omix_edge_num))

        total_edge_num = len(edges)
        print("total edge num: {}".format(total_edge_num))
        logging.info("total edge num: {}".format(total_edge_num))
        for patientId in self.patient_list:
            data = self.data_dict[patientId]
            if total_edge_num > 0:
                data.edge_index = torch.tensor(edges).transpose(0,1).type(torch.int64)
                data.edge_attr = torch.tensor(edge_attrs).type(torch.float32)
            elif total_edge_num == 0 and self.args.allow_no_edge_pretrain:
                data.edge_index = torch.tensor([[],[]]).type(torch.int64)
                data.edge_attr = torch.tensor([]).type(torch.float32)
            
            data.x = data.x[:, :3]
            data.gene_pca_match = torch.tensor(gene_pca_match).type(torch.int64)[None, :]
            self.data_dict[patientId] = data

        return edges, edge_attrs, gene_pca_match
    
    def recalculate_grn_edge_bo_selected_gene(self, mutual_info_mask, indice=None):
        start = 0
        selected_entrzid_sets = [set({}), set({}), set({})]
        gene_pca_match = []
        all_pathway = pd.read_excel('./data/ordered_pathway_146_2pc.xlsx', header=None)[0].to_list()
        missing_valid_gene_in_graph = []
        total_valid_gene_num = 0
        for pathway in all_pathway:
            entrezids = self.kegg[self.kegg['PathwayID'] == pathway].Entrezid.unique()
            if self.args.lag_pca:
                mrna_data = self.raw_mrna[np.intersect1d(entrezids, self.raw_mrna.keys().map(int)).astype(str)].dropna(axis=1,how='any')
                cnv_data = self.raw_cnv[np.intersect1d(entrezids, self.raw_cnv.keys().map(int)).astype(str)].dropna(axis=1,how='any')
                mt_data = self.raw_methylation[np.intersect1d(entrezids, self.raw_methylation.keys().map(int)).astype(str)].dropna(axis=1,how='any')
            else:
                mrna_data = self.raw_mrna[np.intersect1d(entrezids, self.raw_mrna.keys().map(int)).astype(str)].dropna(axis=1,how='any').loc[list(self.os_month.keys())]
                cnv_data = self.raw_cnv[np.intersect1d(entrezids, self.raw_cnv.keys().map(int)).astype(str)].dropna(axis=1,how='any').loc[list(self.os_month.keys())]
                mt_data = self.raw_methylation[np.intersect1d(entrezids, self.raw_methylation.keys().map(int)).astype(str)].dropna(axis=1,how='any').loc[list(self.os_month.keys())]
            for num_data, data in enumerate([mrna_data, cnv_data, mt_data]):
                if mutual_info_mask is not None:
                    for i, entrezid in enumerate(data.columns):
                        entrezid_int = int(entrezid)
                        if mutual_info_mask[start + i][0] > 0:
                            total_valid_gene_num += 1
                            selected_entrzid_sets[num_data].add(entrezid_int)
                        if entrezid_int in self.node_map:
                            node_idx = self.node_map[entrezid_int]
                            # data.x.reshape(-1,1)[gene_pca_match] -> pca data
                            gene_pca_match.append(3*node_idx+num_data)
                        else:
                            gene_pca_match.append(-1)
                            missing_valid_gene_in_graph.append(entrezid_int)
                    start += len(data.columns)
        print("total {} valid gene, missing {} valid gene in graph".format(total_valid_gene_num, len(missing_valid_gene_in_graph)))
        logging.info("total {} valid gene, missing {} valid gene in graph".format(total_valid_gene_num, len(missing_valid_gene_in_graph)))
        idx_map = self.get_index_map()

        edges = []
        edge_attrs = []
        train_patients = []
        y = []
        if indice is not None:
            for i in indice:
                train_patients.append(self.patient_list[i])
            for patientId in train_patients:
                y.append(self.data_dict[patientId].y[1].item())
        
        remain_edge = 0
        del_edge = 0
        #self.mutual_infos = {}
        for num_data, selected_entrzid_set in enumerate(selected_entrzid_sets):
            if str(num_data) in self.args.mute_edge:
                continue
            for entrzid in selected_entrzid_set:
                if entrzid not in self.raw_grn_edge_attrs:
                    continue
                for p_edge, attr in zip(self.raw_grn_edges[entrzid], self.raw_grn_edge_attrs[entrzid]):
                    if p_edge[1] in selected_entrzid_set:
                        remain_edge += 1
                        edges.append([3*idx_map[self.node_map[p_edge[0]]]+num_data, 3*idx_map[self.node_map[p_edge[1]]]+num_data])
                        edge_attrs.append([attr])
                        if self.args.bidir_edge:
                            edges.append([3*idx_map[self.node_map[p_edge[1]]]+num_data, 3*idx_map[self.node_map[p_edge[0]]]+num_data])
                            edge_attrs.append([attr])
        print("grn remain edge: {}, del_edge: {}".format(remain_edge, del_edge))
        logging.info("grn remain edge: {}, del_edge: {}".format(remain_edge, del_edge))

        self.grn_edges = torch.tensor(edges).transpose(0,1).type(torch.int64)
        self.grn_edge_attrs = torch.tensor(edge_attrs).type(torch.float32)

        return edges, edge_attrs, gene_pca_match

    def set_edge_bo_selected_gene(self, edges, edge_attrs, gene_pca_match):
        edges = torch.tensor(edges).transpose(0,1).type(torch.int64)
        edge_attrs = torch.tensor(edge_attrs).type(torch.float32)
        gene_pca_match = torch.tensor(gene_pca_match).type(torch.int64)[None, :]
        for patientId in self.patient_list:
            data = self.data_dict[patientId]
            if len(edges) > 0:
                data.edge_index = edges
                data.edge_attr = edge_attrs
            data.x = data.x[:, :3]
            data.gene_pca_match = gene_pca_match
            self.data_dict[patientId] = data
        
    
    def get_patient_id(self, indexs):
        return [self.patient_list[idx] for idx in indexs]

    def get_index_map(self):
        patientId = self.patient_list[0]
        raw_multi_omics_data = self.raw_multi_omics_data.loc[patientId]
        node_idxs = []
        node_ids = []
        x = []
        for entrezid in self.node_map.keys():
            value = []
            for omics_type in self.omics_types:
                column_name = '_'.join([omics_type, str(entrezid)])
                if column_name in raw_multi_omics_data.keys() and not np.isnan(raw_multi_omics_data[column_name]):
                    value.append(raw_multi_omics_data[column_name])
                else:
                    value.append(0)
            node_idx = self.node_map[entrezid]

            if len(value) == len(self.omics_types):
                node_ids.append(entrezid)
                node_idxs.append(node_idx)
        idx_map = dict(zip(node_idxs, range(len(node_idxs))))
        return idx_map

    def calculate_all_mutual_info(self, train_idxs):
        self.mutual_info = {}
        train_patients = []
        y = []
        for i in train_idxs:
            train_patients.append(self.patient_list[i])
        for patientId in train_patients:
            y.append(self.data_dict[patientId].y[1])
        datas = [self.raw_mrna, self.raw_cnv, self.raw_methylation]
        for num_data, data in enumerate(datas):
            mutual_info = mutual_info_regression(data.loc[train_patients].numpy(), y)
            for i in len(mutual_info):
                entrezid = data.keys[i]
                self.mutual_info[(num_data, entrezid)] = mutual_info[i]


    def valid_pca_mutual_info(self, edge, train_patients, y, num_data):
        if not self.args.edge_select:
            return True
        datas = [self.raw_mrna, self.raw_cnv, self.raw_methylation]
        data = datas[num_data]
        edge_data = data.loc[train_patients][[str(edge[0]), str(edge[1])]]
        if self.args.freeze_mutual_select_init:
            random_state = self.args.random_state
        else:
            random_state = None
        if self.args.knn_mutual_info and num_data != 1:
            pca_mutual_info = kraskov_mi(edge_data.to_numpy(), np.array(y)[:, None])
        else:
            #from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
            #sklearn_lda = LDA(n_components=1)
            #X_lda_sklearn = sklearn_lda.fit_transform(edge_data.to_numpy(), np.array(y)[:, None])
            pca = PCA(n_components=1)
            pca = pca.fit(edge_data.to_numpy())
            # np.dot((data.to_numpy()[0]-pca.mean_)[None,:], pca.components_.T)[0]
            pca_data = pca.transform(edge_data)
            if self.args.mutual_classif:
                pca_mutual_info = mutual_info_classif(pca_data, y, random_state=random_state)
            else:
                pca_mutual_info = mutual_info_regression(pca_data, y)
        if edge[0] not in self.mutual_infos:
            if self.args.knn_mutual_info and num_data != 1:
                source_mutual_info = kraskov_mi(edge_data[str(edge[0])].to_numpy()[:, None], np.array(y)[:, None])
            elif self.args.mutual_classif:
                source_mutual_info = mutual_info_classif(edge_data[str(edge[0])].to_numpy()[:, None], y, random_state=random_state)
            else:
                source_mutual_info = mutual_info_regression(edge_data[str(edge[0])].to_numpy()[:, None], y, random_state=random_state)
            self.mutual_infos[edge[0]] = source_mutual_info
        else:
            source_mutual_info = self.mutual_infos[edge[0]]
        if edge[1] not in self.mutual_infos:
            if self.args.knn_mutual_info and num_data != 1:
                end_mutual_info = kraskov_mi(edge_data[str(edge[1])].to_numpy()[:, None], np.array(y)[:, None])
            elif self.args.mutual_classif:
                end_mutual_info = mutual_info_classif(edge_data[str(edge[1])].to_numpy()[:, None], y, random_state=random_state)
            else:
                end_mutual_info = mutual_info_regression(edge_data[str(edge[1])].to_numpy()[:, None], y, random_state=random_state)
            self.mutual_infos[edge[1]] = end_mutual_info
        else:
            end_mutual_info = self.mutual_infos[edge[1]]
        if pca_mutual_info > self.args.edge_select_threshold * max(source_mutual_info, end_mutual_info):
            return True
        return False
    
    def random_variation_aug(self, data):
        if random.random() < self.args.random_variation_prob:
            #random_variation_mask = np.random.uniform(low=1-self.args.random_range, high=1+self.args.random_range, size=data.x.shape)
            random_variation_mask = torch.zeros(data.x.shape).uniform_(1-self.args.random_range, 1+self.args.random_range)
            random_variation_mask[:, 1] = 1
        else:
            #random_variation_mask = np.ones(data.x.shape)
            random_variation_mask = torch.ones(data.x.shape)
        return random_variation_mask

    def random_mask_aug(self, data):
        if data.y[1].item() > 0.5: # low risk
            pass
        else: # high risk
            pass
    
    def drop_na_line(self, dataframe, percent=0.9):
        return dataframe.loc[dataframe.isnull().sum(axis=1) <= len(dataframe.columns)*percent]

    def add_hat(self, dataframe, sigma, hat_percent=0.99, direction="higher"):
        for col in dataframe.columns:
            col_mean = dataframe[col].mean()
            col_std = dataframe[col].std()
            hat_percent_value = np.percentile(dataframe[col], hat_percent*100)
            if direction == "higher":
                dataframe[col][dataframe[col] > hat_percent_value] = hat_percent_value
            elif direction == "lower":
                dataframe[col][dataframe[col] < hat_percent_value] = hat_percent_value
        return dataframe

    def get_std_weight(self):
        return self.gene_std_value
    
    def get_explain_data(self):
        x = []
        adj = []
        edge_attr = []
        age = []
        raw_indice = []
        gene_pca_match = []
        patient_ids = []
        patient_states = []
        patient_survival_times = []
        self.os_month = self.clinical_data["survive_time"]
        self.survive_state = self.clinical_data["survive_state"]
        for patient_id in self.patient_list:
            data = self.data_dict[patient_id]
            x.append(data.x)
            adj.append(data.edge_index)
            edge_attr.append(data.edge_attr)
            age.append(data.age)
            raw_indice.append(self.raw_indice[patient_id])
            gene_pca_match.append(data.gene_pca_match)
            patient_ids.append(patient_id)
            patient_states.append(self.survive_state[patient_id])
            patient_survival_times.append(self.os_month[patient_id]*30)
        cli_dict = {"patient_ids":patient_ids, "patient_states":patient_states, "patient_survival_times":patient_survival_times}
        return x, adj, edge_attr, age, raw_indice, gene_pca_match, self.node_map, cli_dict

def predefine_data(data, indexs, node_embedding, num_edge, neighbors):
    mul_attr = data.args.mul_attr
    data_dict = {}
    for inx in indexs:
        patientId = data.patient_list[inx]
        raw_multi_omics_data = data.raw_multi_omics_data.loc[patientId]
        raw_cnv = data.named_raw_cnv.loc[patientId]
        x = []
        x_shape = 3 + (1 if node_embedding else 0) + (2 if mul_attr else 0)
        graph_mask = []
        for i in range(data.node_size):
            x.append([0]*x_shape)
        node_ids = []
        node_idxs = []
        for entrezid in data.node_map.keys():
            value = []
            for omics_type in data.omics_types:
                column_name = '_'.join([omics_type, str(entrezid)])
                if column_name in raw_multi_omics_data.keys() and not np.isnan(raw_multi_omics_data[column_name]):
                    value.append(raw_multi_omics_data[column_name])
                else:
                    value.append(0)
            node_idx = data.node_map[entrezid]
            if entrezid in neighbors:
                neighbor = neighbors[entrezid]
            else:
                neighbor = []
            #print(patientId, entrezid, neighbor, [try_get_value(raw_cnv, '_'.join([data.make_graph, str(v)])) for v in neighbor], sum([try_get_value(raw_cnv, '_'.join([data.make_graph, str(v)]))==0 for v in neighbor]), len(neighbor))
            neighbor_flag = data.make_graph is None or sum([try_get_value(raw_cnv, '_'.join([data.make_graph, str(v)]))==0 for v in neighbor]) != len(neighbor)
            if len(value) == len(data.omics_types) and neighbor_flag:
                x[node_idx][:len(value)] = value
                if mul_attr:
                    x[node_idx][len(value):len(value)+2] = value[0]*value[1], value[0]*value[2]
                if node_embedding:
                    x[node_idx][-1] = node_idx
                node_ids.append(entrezid)
                node_idxs.append(node_idx)
        idx_map = dict(zip(node_idxs, range(len(node_idxs))))
        edges = []
        edge_attrs = []
        for node_id in node_ids:
            if node_id not in data.edges:
                continue
            #num_edge = 10
            for p_edge, attr in zip(data.edges[node_id][-num_edge:], data.edge_attrs[node_id][-num_edge:]):
                if p_edge[1] in node_ids:
                    edges.append([idx_map[data.node_map[p_edge[0]]], idx_map[data.node_map[p_edge[1]]]])
                    edge_attrs.append([attr])
                    edges.append([idx_map[data.node_map[p_edge[1]]], idx_map[data.node_map[p_edge[0]]]])
                    edge_attrs.append([attr])
        if data.args.pathway_global_node:
            pathways = data.get_all_pathways()
            pathway_edges = []
            pathway_attr = []
            pathway_node_attr = []
            graph_size = len(idx_map)
            onehot_count = 0
            for i, pathway_id in enumerate(pathways):
                connect_gene_entrezid = data.kegg[data.kegg["PathwayID"] == pathway_id].Entrezid.unique()
                connect_gene_entrezid = np.intersect1d(connect_gene_entrezid, list(data.node_map.keys()))
                tmp_node_idx = list(map(lambda x: idx_map[data.node_map[x]] if (data.node_map[x] in idx_map and in_neighbors(data, raw_cnv, neighbors, x)) else -1,
                                         connect_gene_entrezid))
                pathway_edges.extend([[n_idx, i+graph_size] for n_idx in tmp_node_idx if n_idx >= 0])
                if data.args.global_edge is None:
                    tmp_edge_attr = [0 for n_idx in tmp_node_idx if n_idx >= 0]
                elif data.args.global_edge == 'onehot':
                    tmp_edge_attr = []
                    for tmp_idx in tmp_node_idx:
                        if tmp_idx >= 0:
                            tmp_edge_attr.append(onehot_count)
                        onehot_count += 1
                
                pathway_attr.extend(tmp_edge_attr)
                if data.args.bi_global_node:
                    pathway_edges.extend([[i+graph_size, n_idx] for n_idx in tmp_node_idx if n_idx >= 0])
                    pathway_attr.extend(tmp_edge_attr)

                pathway_node_attr.append(data.pca_result[patientId][pathway_id])
            #print(pathway_edges)
            if pathway_edges == []:
                pathway_edges = [[0, 0]]
                pathway_attr = [0]
                edges = [[0,0]]
                edge_attrs = [0]
            pathway_edges = torch.tensor(pathway_edges).transpose(0,1).type(torch.int64)
            pathway_attr = torch.tensor(pathway_attr).type(torch.float32).reshape((len(pathway_attr), -1))
            pathway_node_attr = torch.tensor(pathway_node_attr).type(torch.float32).reshape((len(pathway_node_attr), -1))

        x = torch.tensor(x).type(torch.float32)[node_idxs]
        if data.args.pathway_global_node:
            x = torch.cat([x, torch.zeros(len(pathways), x.shape[-1])], dim=0)

        #y = torch.tensor([data.os_month[patientId] > 24]).type(torch.int64)
        if data.soft_label:
            if data.survive_state[patientId] == 0:
                y = torch.tensor([0,1.])#.type(torch.int64)
            else:
                os_month = data.os_month[patientId]
                y = torch.softmax(torch.tensor([1-os_month/data.args.risk_threshold,os_month/data.args.risk_threshold-1]), dim=0)#.type(torch.int64)
        else:
            y = torch.tensor([0,0]).type(torch.int64)
            y[int(data.os_month[patientId] > data.args.risk_threshold)] = 1
        
        edges = torch.tensor(edges).transpose(0,1).type(torch.int64)
        edge_attrs = torch.tensor(edge_attrs).type(torch.float32).reshape((len(edge_attrs), -1))
        if data.args.pathway_global_node:
            edges = torch.cat([edges, pathway_edges], dim=-1)
            edge_attrs = torch.cat([edge_attrs, pathway_attr], dim=0)
        prepare_data = Data(x=x, y=y, edge_index=edges, edge_attr=edge_attrs)
        prepare_data.age = data.age[patientId]
        prepare_data.node_size = len(x)
        if data.args.pathway_global_node:
            prepare_data.pathway_global_nodes = torch.tensor(range(len(pathways)))
            prepare_data.pathway_node_attr = pathway_node_attr
        prepare_data.patientId = patientId
        prepare_data.os_months = data.os_month[patientId]
        prepare_data.survival = data.survive_state[patientId]
        data_dict[patientId] = prepare_data
    return data_dict

def in_neighbors(data, raw_cnv, neighbors, eid):
    if eid in neighbors:
        neighbor = neighbors[eid]
    else:
        neighbor = []
    return sum([try_get_value(raw_cnv, '_'.join(['cnv', str(v)]))==0 for v in neighbor]) != len(neighbor)

def try_get_value(row, require_id):
    try:
        return row[require_id]
    except:
        return 0

