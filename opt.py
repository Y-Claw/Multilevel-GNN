import argparse
import time
import os
import yaml

def load_config(file_path):
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)

parser = argparse.ArgumentParser()

# paths
parser.add_argument(
    '--cancer_type',
    default= 'gbm',
    type=str,
    help='raw mrna path')
parser.add_argument(
    '--raw_mrna_path',
    default= './data/{}/pathcnn_raw_mrna_pid_u133.csv',
    type=str,
    help='raw mrna path')
parser.add_argument(
    '--raw_cnv_path',
    default= './data/{}/pathcnn_raw_cnv_pid.csv',
    type=str,
    help='raw cnv path')
parser.add_argument(
    '--raw_methylation_path',
    default= './data/{}/pathcnn_raw_methylation_pid.csv',
    type=str,
    help='raw methylation path')
parser.add_argument(
    '--clinical_path',
    default= './data/{}/pathcnn_clinical_data.pkl',
    type=str,
    help='node path')
parser.add_argument(
    '--node_path',
    default= './data/pathcnn_node.csv',
    type=str,
    help='node path')
parser.add_argument(
    '--edge_path',
    default= './data/{}/pyscenic_adj_nes1.0_auc0.01_weighted_l1.tsv',
    type=str,
    help='edge path')
parser.add_argument(
    '--grn_edge_path',
    default= './data/adjacencies.csv',
    type=str,
    help='grn edge path')
parser.add_argument(
    '--kegg_path',
    default= './data/kegg_52.csv', #from 2023.4.6
    type=str,
    help='node path')
parser.add_argument(
    '--pathway_path',
    default= './data/pathcnn_pathway_codes.pkl',
    type=str,
    help='node path')
parser.add_argument(
    '--z_score',
    default= False,
    type=bool,
    help='node path')
parser.add_argument(
    '--use_column',
    default=None,
    type=str,
    help='node path')
parser.add_argument(
    '--make_graph',
    default= None,
    type=str,
    help='node path')
parser.add_argument(
    '--neighborhood',
    default=0,
    type=int,
    help='node path')
parser.add_argument(
    '--pretain_only_pathway_edge',
    default=False,
    type=bool,
    help='node path')
parser.add_argument('--pathway_global_node', action='store_true')
parser.add_argument('--pathway_num',
    default=146,
    type=int,
    help='node path')
parser.add_argument('--risk_threshold',
    default=24,
    type=int,
    help='node path')


#dataset
parser.add_argument(
    '--position_embedding',
    default= None,
    type=str,
    help='node path')
parser.add_argument('--mul_attr', action='store_true')
parser.add_argument('--soft_label', action='store_true')
parser.add_argument(
    '--edge_type',
    default='grnboost2',
    type=str,
    help='edge type')
parser.add_argument('--bidir_edge', action='store_true')
parser.add_argument('--resgnn', action='store_true')
parser.add_argument('--pca_match_mask', action='store_true')
parser.add_argument(
    '--mute_edge',
    default='',
    type=str,
    help='mute edge for mrna cnv or mt')
parser.add_argument('--add_hat', action='store_true')
parser.add_argument('--add_hat_sigma', type=float, default=3)
parser.add_argument('--add_hat_percent', type=float, default=0.99)

#model
parser.add_argument(
    '--model',
    default= 'deepergcn',
    type=str,
    help='node path')
parser.add_argument('--num_layers', type=int, default=3,
                    help='the number of layers of the networks')
parser.add_argument('--mlp_layers', type=int, default=2,
                    help='the number of layers of mlp in conv')
parser.add_argument('--hidden_channels', type=int, default=128,
                    help='the dimension of embeddings of nodes and edges')
parser.add_argument('--final_channels', type=int, default=1,
                    help='the dimension of embeddings of nodes and edges')
parser.add_argument('--final_head', type=int, default=1,
                    help='the dimension of embeddings of nodes and edges')
parser.add_argument('--block', default='res+', type=str,
                    help='graph backbone block type {res+, res, dense, plain}')
parser.add_argument('--conv', type=str, default='gen',
                    help='the type of GCNs')
parser.add_argument('--gcn_aggr', type=str, default='max',
                    help='the aggregator of GENConv [mean, max, add, softmax, softmax_sg, power]')
parser.add_argument('--norm', type=str, default='layer',
                    help='the type of normalization layer')
parser.add_argument('--num_tasks', type=int, default=2,
                    help='the number of prediction tasks')
# learnable parameters
parser.add_argument('--t', type=float, default=1.0,
                    help='the temperature of SoftMax')
parser.add_argument('--p', type=float, default=1.0,
                    help='the power of PowerMean')
parser.add_argument('--learn_t', action='store_true')
parser.add_argument('--learn_p', action='store_true')
# message norm
parser.add_argument('--msg_norm', action='store_true')
parser.add_argument('--learn_msg_scale', action='store_true')
# encode edge in conv
parser.add_argument('--conv_encode_edge', action='store_true')
# graph pooling type
parser.add_argument('--graph_pooling', type=str, default='mean',
                    help='graph pooling method')
parser.add_argument('--node_embedding', type=bool, default=False,
                    help='graph pooling method')
parser.add_argument('--node_num', type=int, default=5606,
                    help='graph pooling method')
parser.add_argument('--omics_num', type=int, default=3,
                    help='graph pooling method')
parser.add_argument('--used_omics', type=str, default="012",
                    help='graph pooling method')
parser.add_argument('--node_embedding_dim', type=int, default=32,
                    help='graph pooling method')
parser.add_argument('--num_layer_head', type=int, default=1,
                    help='graph pooling method')
parser.add_argument('--use_age', type=bool, default=False,
                    help='graph pooling method')
parser.add_argument('--head_dropout', type=bool, default=False,
                    help='graph pooling method')
parser.add_argument('--use_edge_attr', action='store_true')
parser.add_argument('--pathway_readout', type=str, default="maxpool",
                    help='graph readout method')         
parser.add_argument('--gnn_encoder', type=str, default='linear',
                    help='graph readout method')
parser.add_argument('--pca_only', action='store_true')
parser.add_argument('--pca_compare', action='store_true')
parser.add_argument('--no_inter_drop', action='store_true')
parser.add_argument('--no_inter_norm', action='store_true')
parser.add_argument('--head_init', action='store_true')
parser.add_argument('--all_init', type=bool, default=True)
parser.add_argument('--pre_readout_drop', action='store_true')
parser.add_argument('--pre_concat_age', action='store_true')
parser.add_argument('--bi_global_node', action='store_true')
parser.add_argument('--global_edge', type=str, default="onehot",
                    help='graph pooling method')
parser.add_argument('--init_emb', action='store_true')
parser.add_argument('--feature_drop', action='store_true')
parser.add_argument('--pca_prelinear', action='store_true')
parser.add_argument('--more_conv', action='store_true')
parser.add_argument('--pathcnn_kernel_size', type=int, default=3,
                    help='graph pooling method')
parser.add_argument('--learnable_pca', action='store_true')
parser.add_argument('--init_with_pca', action='store_true')
parser.add_argument('--pca_loss', action='store_true')
parser.add_argument('--pca_loss_coef', type=float, default=1,
                    help='graph pooling method')
parser.add_argument('--pca_indep_loss', action='store_true')
parser.add_argument('--pca_init_type', type=str, default=None,
                    help='graph pooling method')
parser.add_argument('--pca_sim_dim', type=int, default=5,
                    help='graph pooling method')
parser.add_argument('--pca_dim', type=int, default=2,
                    help='graph pooling method')
parser.add_argument('--pca_pool_dim', type=int, default=2,
                    help='graph pooling method')
parser.add_argument('--mutual_info_mask', action='store_true')
parser.add_argument('--mutual_info_threshold', type=float, default=None,
                    help='graph pooling method')
parser.add_argument('--mutual_info_pca', action='store_true')
parser.add_argument('--pathway_pool_dim', type=int, default=4, help='graph pooling method')
parser.add_argument('--step', type=int, default=0, help='graph pooling method')
parser.add_argument('--gamma', type=float, default=0.25, help='graph pooling method')
parser.add_argument('--gnn_pathcnn', action='store_true')
parser.add_argument('--z_mean', action='store_true')
parser.add_argument('--freeze_pca_weight', action='store_true')
parser.add_argument('--value_att_mask', action='store_true')
parser.add_argument('--edge_select', action='store_true')
parser.add_argument('--edge_select_threshold', type=float, default=1, help='graph pooling method')
parser.add_argument('--grn_edge_select_threshold', type=float, default=None, help='graph pooling method')
parser.add_argument('--node_select_threshold', type=float, default=1, help='graph pooling method')
parser.add_argument('--mutual_neighbors', type=int, default=3, help='graph pooling method')
parser.add_argument('--mutual_classif', action='store_true')
parser.add_argument('--drop_irr_pathway', action='store_true')
parser.add_argument('--mean_pca_init', action='store_true')
parser.add_argument('--pca_mean_value', type=float, default=0.006, help='graph pooling method')
parser.add_argument('--random_variation_aug', action='store_true')
parser.add_argument('--random_mask_aug', action='store_true')
parser.add_argument('--random_range', type=float, default=0.05, help='graph pooling method')
parser.add_argument('--random_variation_prob', type=float, default=0.5, help='graph pooling method')
parser.add_argument('--random_state', type=int, default=1, help='graph pooling method')
parser.add_argument('--freeze_node_embedding', action='store_true')
parser.add_argument('--freeze_mutual_select_init', action='store_true')
parser.add_argument('--freeze_dataloader_init', action='store_true')
parser.add_argument('--freeze_net_params_init', action='store_true')
parser.add_argument('--knn_mutual_info', action='store_true')
parser.add_argument('--set_all_seed', action='store_true')
parser.add_argument('--seed', type=int, default=1, help='graph pooling method')
parser.add_argument('--split_seed', type=int, default=1, help='graph pooling method')
parser.add_argument('--split_shaffle', action='store_true')
parser.add_argument('--lag_pca', action='store_true')
parser.add_argument('--align_data', action='store_true')
parser.add_argument('--class_sample', action='store_true')
parser.add_argument('--weighted_loss', action='store_true')
parser.add_argument('--batch_weighted_loss', action='store_true')
parser.add_argument('--pca_all', action='store_true')
parser.add_argument('--head_dim', type=int, default=64, help='head dim')
parser.add_argument('--hidden_head', type=int, default=8, help='hidden dim')
parser.add_argument('--gnn_name', type=str, default='gat',
                    help='graph model name')
parser.add_argument('--dense_gnn', action='store_true')
parser.add_argument('--first_conv_channel', type=int, default=32, help='conv dim')
parser.add_argument('--construct_cnv_mrna_edge', action='store_true')
parser.add_argument('--construct_mt_mrna_edge', action='store_true')
parser.add_argument('--construct_mrna_cnv_edge', action='store_true')
parser.add_argument('--construct_mrna_mt_edge', action='store_true')
parser.add_argument('--only_mrna_pred', action='store_true')
parser.add_argument('--reverse_mt', action='store_true')
parser.add_argument('--weighted_edge', action='store_true')
parser.add_argument('--reverse_mt_attr', action='store_true')
parser.add_argument('--gnn_act', type=str, default='leakyrelu',
                    help='graph model name')
parser.add_argument('--remain_all_tf', action='store_true')
parser.add_argument('--remain_tf_nums', type=str, default="012", help='graph pooling method')
parser.add_argument('--zscore_mrna', action='store_true')
parser.add_argument('--reorder_pathway', action='store_true')
parser.add_argument('--reorder_type', type=str, default="pca", help='reorder type')
parser.add_argument('--pathway_similarity', type=str, default="correlation", help='pathway adj construction method')
parser.add_argument('--precise_order', action='store_true')
parser.add_argument('--selected_similarity', action='store_true')
parser.add_argument('--gnn_last_norm', action='store_true')
parser.add_argument('--gnn_mlp_norm', type=str, default="none", help='GNN mlp normal')
parser.add_argument('--merge_mode', type=str, default="mult", help='GNN mlp normal')
parser.add_argument('--add_coef1', type=float, default=0.5, help='GNN mlp normal')
parser.add_argument('--add_coef2', type=float, default=0.5, help='GNN mlp normal')

parser.add_argument('--repeat_mask', action='store_true')
parser.add_argument('--repeat_cyclic', type=int, default=2)
parser.add_argument('--repeat_norm', action='store_true')
parser.add_argument('--conv_channel_list', nargs='+', type=int, default=[32,64], help='A list of integers')
parser.add_argument('--conv_kernel_list', nargs='+', type=int, default=[1, 1], help='A list of integers')

parser.add_argument('--embedding_init_type', type=str, default="xavier", help='GNN mlp normal')
parser.add_argument('--emb_val', type=float, default=0.01)
parser.add_argument('--input_drop', type=float, default=None)
parser.add_argument('--input_emb_drop', type=float, default=None)

#train args
parser.add_argument(
    '--epochs',
    default=200,
    type=int,
    help='node path')
parser.add_argument(
    '--batch_size',
    default=4,
    type=int,
    help='node path')
parser.add_argument(
    '--num_workers',
    default=8,
    type=int,
    help='node path')
parser.add_argument(
    '--optimizer',
    default='adamw',
    type=str,
    help='node path')
parser.add_argument(
    '--lr',
    default= 1e-4,
    type=float,
    help='node path')
parser.add_argument(
    '--wd',
    default=0,
    type=float,
    help='node path')
parser.add_argument(
    '--beta1',
    default= 0.9,
    type=float,
    help='node path')
parser.add_argument(
    '--beta2',
    default= 0.999,
    type=float,
    help='node path')
parser.add_argument(
    '--weight_balance',
    default=False,
    type=bool,
    help='node path')
parser.add_argument(
    '--weight_power',
    default=1,
    type=float,
    help='node path')
parser.add_argument('--clip_grad', action='store_true')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gnn_dropout', type=float, default=0.0)
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--device_num', type=int, default=1)

parser.add_argument('--num_run', type=int, default=1)
parser.add_argument('--metrics', type=str, default='auc')

parser.add_argument('--model_save_path', type=str, default='./checkpoint')
parser.add_argument('--name_pre', type=str, default='test')

parser.add_argument('--debug', type=bool, default=False)

parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--save_tag', type=str, default='')
parser.add_argument('--use_cache', action='store_true')

parser.add_argument('--time', type=str, default='')

#autoencoder/vae base
parser.add_argument('--decoder_dim', type=int, default=4096)
parser.add_argument('--decoder_type', type=str, default='flatten')
parser.add_argument('--autoencoder_save_path', type=str, default='./AutoEncoder_checkpoint')
parser.add_argument('--autoencoder_save_dir', type=str, default='flatten')

parser.add_argument('--load_autoencoder_ckpt', action='store_true')
parser.add_argument('--autoencoder_ckpt_path', type=str, default='2023-09-05-14-02-09-test/test_mae_best.pth')
parser.add_argument('--load_autoencoder_epoch', type=str, default=None)
parser.add_argument('--warmup_epochs', type=int, default=0)
parser.add_argument('--warmup_lr', type=float, default=0.00005)

parser.add_argument('--channel_one', action='store_true')
parser.add_argument('--vae_generate_train_sample', action='store_true')

parser.add_argument('--save_method', type=str, default="best_p_value")
parser.add_argument('--reconstruct_head', action='store_true')
parser.add_argument('--allow_no_edge_pretrain', action='store_true')
parser.add_argument('--train_with_vae_loss', action='store_true')

parser.add_argument('--pretrain_std_loss', action='store_true')
parser.add_argument('--pretrain_std_coef', type=float, default=1)
parser.add_argument('--pretrain_idp_loss', action='store_true')
parser.add_argument('--pretrain_idp_coef', type=float, default=1)
parser.add_argument('--pretrain_corr_loss', action='store_true')
parser.add_argument('--pretrain_corr_coef', type=float, default=1)
parser.add_argument('--kl_beta', type=float, default=1)
parser.add_argument('--std_weight', action='store_true')
parser.add_argument('--grad_weight', action='store_true')

#mmd vae
parser.add_argument('--mmd_kernel_type', type=str, default='imq')
parser.add_argument('--mmd_alpha', type=float, default=-9.0)
parser.add_argument('--mmd_beta', type=float, default=10.5)
parser.add_argument('--kld_weight', type=float, default=0.2)
parser.add_argument('--mmd_reg_weight', type=float, default=110)
parser.add_argument('--z_var', type=float, default=2)
parser.add_argument('--std_weight_coef', type=float, default=1)
parser.add_argument('--grad_weight_coef', type=float, default=1)

#vqvae
parser.add_argument('--vqvae_num_embeddings', type=int, default=512)
parser.add_argument('--vqvae_beta', type=float, default=0.25)

#dynamic pooling
parser.add_argument('--diff_pooling_location', type=str, default="pathway")
parser.add_argument('--diff_pooling_layer', type=int, default=2)
parser.add_argument('--diff_pooling_hidden_dim', type=int, default=32)
parser.add_argument('--diff_pooling_output_dim', type=int, default=64)
parser.add_argument('--after_pooling_layer', type=int, default=1)
parser.add_argument('--pooling_type', type=str, default="correlation")

#active learning
parser.add_argument('--active_learning', action='store_true')
parser.add_argument('--active_type', type=str, default='loss')
parser.add_argument('--active_percent', type=float, default=0.25)

#reduction
parser.add_argument('--reduction_method', type=str, default='linear_projection')
parser.add_argument('--pca_lowrank_niter', type=int, default=2)

#igscore
parser.add_argument('--ckpt_path', type=str, default="")
parser.add_argument('--igscore_epoch', type=int, default=0)

parser.add_argument('--config', type=str, help='Path to the YAML configuration file')

def parse_opts():
    args = parser.parse_args()
    if args.config:
        config_data = load_config(args.config)
        for key, value in config_data.items():
            setattr(args, key, value)

    return args

if __name__ == "__main__":
    args = parse_opts()
    import pdb; pdb.set_trace()