from .gcn import *
from .deepergcn import *
from .deepergcn_virtual_node import *
from .pathcnn import *
from .multilevel_gnn import *
from .multilevel_gnn_seq import *
from .autoencoder import *
from .vae import *
from .vq_vae import *

MODELS = {
    'deepergcn':DeeperGCN,
    'multiomix': MultiOmixGCN,
    'pathcnn': PathCNN, 
    'multilevel_gnn': MultilevelGNN,
    'autoencoder': AutoEncoder,
    'vae': VAE,
    'mmd_vae': VAE,
    'vq_vae': VQ_VAE,
    'multilevel_gnn_seq': MultilevelGNNSeq,
}

def get_model(model_name):
    return MODELS[model_name]