import sys
import os
import os.path as osp
import numpy as np
import pandas as pd
import math
import random
import time
import logging
from tqdm import tqdm
import re

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, random_split, Subset
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DataParallel
from torch_geometric.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedKFold
import statistics
from captum.attr import IntegratedGradients

from opt import parse_opts
from models import get_model
from dataloader.multiloader import MyData
from optimizer import get_optimizer
from utils.cache_data import have_cached_data, cache_data, get_cached_data
import pickle

import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test

def transform_omics_string(input_str):
    replacements = {'mrna': 'mRNA', 'cnv': 'CNV', 'mt': 'MT',
                    'Jak stat': "JAK-STAT", "Mapk": "MAPK"}
    for old, new in replacements.items():
        input_str = input_str.replace(old, new)
    return input_str

def draw_kmfit(df_predict, df_cli, save_res, args):
    save_path = '{}/{}/'.format(args.model_save_path, save_res)
    if not osp.exists(save_path):
        os.mkdir(save_path)
    #import pdb; pdb.set_trace()
    df_predict = df_predict[df_predict.zscore > 1.96]
    smps = df_predict.columns.to_list()[3:]
    pvalue_dict = {
        "mrna":[], 
        "cnv": [],
        "mt": [],
    }
    table_dict = {
        "mrna":[], 
        "cnv": [],
        "mt": [],
    }
    omics_list = ["mrna", "cnv", "mt"]
    pathway_name_list = []
    for idx, row in df_predict.iterrows():
        pathway = row[0]
        score_median = row['median']
        smp_idx = 0
        group1 = []
        group2 = []
        for value in row[3:]:
            if value < score_median:
                group1.append(smps[smp_idx])
            else:
                group2.append(smps[smp_idx])
            smp_idx += 1
        df_cli_group1 = df_cli[df_cli.index.isin(group1)]
        df_cli_group2 = df_cli[df_cli.index.isin(group2)]

        # pvalue: log-rank test
        results=logrank_test(df_cli_group1.survive_time*30, df_cli_group2.survive_time*30,
                            event_observed_A=df_cli_group1.survive_state, 
                            event_observed_B=df_cli_group2.survive_state)
        pvalue = results.p_value

        # plot
        kmf = KaplanMeierFitter()
        #plt.tight_layout()
        ax = plt.subplot(111)
        kmf.fit(df_cli_group1.survive_time*30, event_observed=df_cli_group1.survive_state,label='Group1')
        kmf.plot(ax=ax)
        kmf.fit(df_cli_group2.survive_time*30, event_observed=df_cli_group2.survive_state,label='Group2')
        kmf.plot(ax=ax)
        pvalue_dict[omics_list[idx%3]].append(pvalue)
        pvalue = f"{pvalue:.3e}" if pvalue < 0.001 else f"{pvalue:.3}"
        pathway_name = pathway.replace('_', ' ').capitalize().replace("Ecm ", "ECM-")
        pathway_name = transform_omics_string(pathway_name)
        pathway_name_list.append(pathway_name)
        pathway_pattern = re.compile(r'(.+)\s(\S+)$')
        match = pathway_pattern.match(pathway_name)
        table_dict[match.group(2).lower()].append([match.group(1), match.group(2), f"{row['zscore']:.4}", pvalue])
        plt.title(pathway_name + '\n log-rank test: {}'.format(pvalue))
        plt.xlabel('Time (Days)')
        pathway = pathway.replace(' ', '')
        pathway = pathway.replace('/', '_')
        plt.savefig(save_path + f"{row['zscore']:.4}-" + pathway_name  + '.pdf')
        plt.clf()
    with open(save_path+'statistic.txt', 'w') as file:
        file.write("total pathway num:{}\n".format(sum([len(pvalue_dict[omics]) for omics in omics_list])))
        for omics_name in omics_list:
            file.write("{} pathway num:{}\n".format(omics_name, len(pvalue_dict[omics_name])))
            file.write("{} hit rate: {}/{}\n".format(omics_name, sum(np.array(pvalue_dict[omics_name]) < 0.05), len(pvalue_dict[omics_name])))       
        for data_list in table_dict.values():
            sorted_list = sorted(data_list, key=lambda x: x[2], reverse=True)
            for line_data in sorted_list:
                file.write(" & ".join(line_data)+"\\\\\n")


def draw_kmfit_by_group(df_cli, group1, group2, save_res, save_name, args):
    save_path = '{}/{}/'.format(args.model_save_path, save_res)
    if not osp.exists(save_path):
        os.mkdir(save_path)

    df_cli_group1 = df_cli[df_cli.index.isin(group1)]
    df_cli_group2 = df_cli[df_cli.index.isin(group2)]

    # pvalue: log-rank test
    results=logrank_test(df_cli_group1.survive_time*30, df_cli_group2.survive_time*30,
                        event_observed_A=df_cli_group1.survive_state, 
                        event_observed_B=df_cli_group2.survive_state)
    pvalue = results.p_value

    # plot
    kmf = KaplanMeierFitter()
    ax = plt.subplot(111)
    kmf.fit(df_cli_group1.survive_time*30, event_observed=df_cli_group1.survive_state,label='Group1')
    kmf.plot(ax=ax)
    kmf.fit(df_cli_group2.survive_time*30, event_observed=df_cli_group2.survive_state,label='Group2')
    kmf.plot(ax=ax)
    pvalue = f"{pvalue:.3e}" if pvalue < 0.001 else pvalue
    plt.title(save_name + ' , log-rank test: {}'.format(pvalue))
    plt.xlabel('Time (Days)')
    plt.savefig(save_path + save_name + '.pdf')
    plt.clf()