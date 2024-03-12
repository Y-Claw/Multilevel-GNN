import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric as tg
from .torch_nn import MLP, act_layer, norm_layer, BondEncoder
from .torch_edge import DilatedKnnGraph
from .torch_message import GenMessagePassing, MsgNorm, PathwayMessagePassing
from utils.pyg_util import scatter_
from torch_geometric.utils import remove_self_loops, add_self_loops


class GENConv(GenMessagePassing):
    """
     GENeralized Graph Convolution (GENConv): https://arxiv.org/pdf/2006.07739.pdf
     SoftMax  &  PowerMean Aggregation
    """
    def __init__(self, in_dim, emb_dim,
                 aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 y=0.0, learn_y=False,
                 gnn_encoder='linear',
                 msg_norm=False, learn_msg_scale=True,
                 encode_edge=False, bond_encoder=False,
                 edge_feat_dim=None,
                 norm='batch', mlp_layers=2,
                 eps=1e-7, pca_only=False):

        super(GENConv, self).__init__(aggr=aggr,
                                      t=t, learn_t=learn_t,
                                      p=p, learn_p=learn_p,
                                      y=y, learn_y=learn_y)

        self.gnn_encoder = gnn_encoder
        if gnn_encoder == 'linear':
            channels_list = [in_dim]

            for i in range(mlp_layers-1):
                channels_list.append(in_dim*2)

            channels_list.append(emb_dim)

            self.feature_encoder = MLP(channels=channels_list,
                        norm=norm,
                        last_lin=True)
        elif gnn_encoder == 'conv1x1':
            self.feature_encoder = nn.Sequential(
                nn.Conv1d(in_dim, emb_dim, 1),
                nn.ReLU()
            )

        self.msg_encoder = torch.nn.ReLU()
        self.eps = eps

        self.msg_norm = msg_norm
        self.encode_edge = encode_edge
        self.bond_encoder = bond_encoder

        if msg_norm:
            self.msg_norm = MsgNorm(learn_msg_scale=learn_msg_scale)
        else:
            self.msg_norm = None

        if self.encode_edge:
            if self.bond_encoder:
                self.edge_encoder = BondEncoder(emb_dim=in_dim)
            else:
                self.edge_encoder = torch.nn.Linear(edge_feat_dim, in_dim)

        self.pca_only = pca_only

    def forward(self, x, edge_index, edge_attr=None):
        if self.pca_only:
            out = self.feature_encoder(x)
        else:
            if self.encode_edge and edge_attr is not None:
                edge_emb = self.edge_encoder(edge_attr)
            else:
                edge_emb = edge_attr

            tx, te =  x.flatten(1), edge_emb.flatten(1)
            m = self.propagate(edge_index, x=tx, edge_attr=te)
            import pdb 
            #pdb.set_trace()
            #torch.cosine_similarity(x[5606], x[5606+5748], dim=0)
            if self.msg_norm is not None:
                m = self.msg_norm(x, m)

            h = x + m.reshape(x.shape)
            out = self.feature_encoder(h)

        return out

    def message(self, x_j, edge_attr=None):

        if edge_attr is not None:
            msg = x_j + edge_attr
        else:
            msg = x_j

        return self.msg_encoder(msg) + self.eps

    def update(self, aggr_out):
        return aggr_out


class PathwayConv(PathwayMessagePassing):
    """
     GENeralized Graph Convolution (GENConv): https://arxiv.org/pdf/2006.07739.pdf
     SoftMax  &  PowerMean Aggregation
    """
    def __init__(self, in_dim, emb_dim,
                 aggr='softmax',
                 t=1.0, learn_t=False,
                 p=1.0, learn_p=False,
                 y=0.0, learn_y=False,
                 msg_norm=False, learn_msg_scale=True,
                 encode_edge=False, bond_encoder=False,
                 edge_feat_dim=None,
                 norm='batch', mlp_layers=2,
                 eps=1e-7):

        super(PathwayConv, self).__init__(aggr=aggr,
                                      t=t, learn_t=learn_t,
                                      p=p, learn_p=learn_p,
                                      y=y, learn_y=learn_y)

        channels_list = [in_dim]

        for i in range(mlp_layers-1):
            channels_list.append(in_dim*2)

        channels_list.append(emb_dim)

        self.mlp = MLP(channels=channels_list,
                       norm=norm,
                       last_lin=True)

        self.msg_encoder = torch.nn.Linear(2*in_dim, in_dim)
        self.eps = eps

        self.msg_norm = msg_norm
        self.encode_edge = encode_edge
        self.bond_encoder = bond_encoder

        if msg_norm:
            self.msg_norm = MsgNorm(learn_msg_scale=learn_msg_scale)
        else:
            self.msg_norm = None

        if self.encode_edge:
            if self.bond_encoder:
                self.edge_encoder = BondEncoder(emb_dim=in_dim)
            else:
                self.edge_encoder = torch.nn.Linear(edge_feat_dim, in_dim)

    def forward(self, x, edge_index, edge_attr=None, mask=None):

        m = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        h = x + m
        out = self.mlp(h)
        out = F.relu(out)
        if mask is not None:
            out = out * mask
        return out

    def message(self, x_j, edge_attr=None):
        if edge_attr is not None:
            msg = torch.matmul(x_j[:, :, None], edge_attr[:, None, :]).flatten(1)
        else:
            msg = x_j

        #return self.msg_encoder(msg) + self.eps
        return self.msg_encoder(msg)

    def update(self, aggr_out):
        return aggr_out


class MRConv(nn.Module):
    """
    Max-Relative Graph Convolution (Paper: https://arxiv.org/abs/1904.03751)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        super(MRConv, self).__init__()
        self.nn = MLP([in_channels*2, out_channels], act, norm, bias)
        self.aggr = aggr

    def forward(self, x, edge_index):
        """"""
        x_j = scatter_(self.aggr, torch.index_select(x, 0, edge_index[0]) - torch.index_select(x, 0, edge_index[1]), edge_index[1], dim_size=x.shape[0])
        return self.nn(torch.cat([x, x_j], dim=1))


class EdgConv(tg.nn.EdgeConv):
    """
    Edge convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='max'):
        super(EdgConv, self).__init__(MLP([in_channels*2, out_channels], act, norm, bias), aggr)

    def forward(self, x, edge_index):
        return super(EdgConv, self).forward(x, edge_index)


class GATConv(nn.Module):
    """
    Graph Attention Convolution layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels,  act='relu', norm=None, bias=True, heads=8):
        super(GATConv, self).__init__()
        self.gconv = tg.nn.GATConv(in_channels, out_channels, heads, bias=bias)
        m =[]
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, edge_index, edge_attr=None):
        out = self.unlinear(self.gconv(x, edge_index))
        return out


class SAGEConv(tg.nn.SAGEConv):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`False`, output features
            will not be :math:`\ell_2`-normalized. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 nn,
                 norm=True,
                 bias=True,
                 relative=False,
                 **kwargs):
        self.relative = relative
        if norm is not None:
            super(SAGEConv, self).__init__(in_channels, out_channels, normalize=True, bias=bias, **kwargs)
        else:
            super(SAGEConv, self).__init__(in_channels, out_channels, normalize=False, bias=bias, **kwargs)
        self.nn = nn
        self.weight = self.lin_r.weight.T
        self.normalize = norm
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x, edge_index, size=None, edge_attr=None):
        """"""
        if size is None:
            edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
            edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))

        x = x.unsqueeze(-1) if x.dim() == 1 else x
        edge_attr = edge_attr.unsqueeze(-1) if edge_attr.dim() == 1 else edge_attr
        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr=None):
        if edge_attr is not None:
            x_j = x_j * edge_attr
        if self.relative:
            x = torch.matmul(x_j - x_i, self.lin_r.weight.T)
        else:
            x = torch.matmul(x_j, self.lin_r.weight.T)
        return x

    def update(self, aggr_out, x):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        out = self.nn(torch.cat((x, aggr_out), dim=1))
        if self.normalize:
            out = F.normalize(out, p=2, dim=-1)
        return out


class RSAGEConv(SAGEConv):
    """
    Residual SAGE convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=False, mlp_norm=None, bias=True, relative=False, drop=0.0):
        nn = MLP([out_channels + in_channels, out_channels], act, mlp_norm, bias, drop=drop)
        super(RSAGEConv, self).__init__(in_channels, out_channels, nn, norm, False, relative)


class SemiGCNConv(nn.Module):
    """
    SemiGCN convolution layer (with activation, batch normalization)
    """

    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True):
        super(SemiGCNConv, self).__init__()
        self.gconv = tg.nn.GCNConv(in_channels, out_channels, bias=bias)
        m = []
        if act:
            m.append(act_layer(act))
        if norm:
            m.append(norm_layer(norm, out_channels))
        self.unlinear = nn.Sequential(*m)

    def forward(self, x, edge_index):
        out = self.unlinear(self.gconv(x, edge_index))
        return out


class GinConv(tg.nn.GINConv):
    """
    GINConv layer (with activation, batch normalization)
    """
    def __init__(self, in_channels, out_channels, act='relu', norm=None, bias=True, aggr='add'):
        super(GinConv, self).__init__(MLP([in_channels, out_channels], act, norm, bias))

    def forward(self, x, edge_index):
        return super(GinConv, self).forward(x, edge_index)


class GraphConv(nn.Module):
    """
    Static graph convolution layer
    """
    def __init__(self, in_channels, out_channels, conv='edge',
                 act='relu', norm=None, bias=True, heads=8, mlp_norm=None, drop=0.0):
        super(GraphConv, self).__init__()
        if conv.lower() == 'edge':
            self.gconv = EdgConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'mr':
            self.gconv = MRConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gat':
            self.gconv = GATConv(in_channels, out_channels//heads, act, norm, bias, heads)
        elif conv.lower() == 'gcn':
            self.gconv = SemiGCNConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'gin':
            self.gconv = GinConv(in_channels, out_channels, act, norm, bias)
        elif conv.lower() == 'sage':
            self.gconv = RSAGEConv(in_channels, out_channels, act, norm, mlp_norm, bias, False, drop)
        elif conv.lower() == 'rsage':
            self.gconv = RSAGEConv(in_channels, out_channels, act, norm, mlp_norm, bias, True, drop)
        else:
            raise NotImplementedError('conv {} is not implemented'.format(conv))

    def forward(self, x, edge_index, edge_attr=None):
        return self.gconv(x, edge_index, edge_attr=edge_attr)


class DynConv(GraphConv):
    """
    Dynamic graph convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size=9, dilation=1, conv='edge', act='relu',
                 norm=None, bias=True, heads=8, **kwargs):
        super(DynConv, self).__init__(in_channels, out_channels, conv, act, norm, bias, heads)
        self.k = kernel_size
        self.d = dilation
        self.dilated_knn_graph = DilatedKnnGraph(kernel_size, dilation, **kwargs)

    def forward(self, x, batch=None, edge_index=None):
        if edge_index is None:
            edge_index = self.dilated_knn_graph(x, batch)
        return super(DynConv, self).forward(x, edge_index)


class PlainDynBlock(nn.Module):
    """
    Plain Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, **kwargs):
        super(PlainDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None, edge_index=None):
        return self.body(x, batch, edge_index), batch


class ResDynBlock(nn.Module):
    """
    Residual Dynamic graph convolution block
    """
    def __init__(self, channels,  kernel_size=9, dilation=1, conv='edge', act='relu', norm=None,
                 bias=True, res_scale=1, **kwargs):
        super(ResDynBlock, self).__init__()
        self.body = DynConv(channels, channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)
        self.res_scale = res_scale

    def forward(self, x, batch=None, edge_index=None):
        return self.body(x, batch, edge_index) + x*self.res_scale, batch


class DenseDynBlock(nn.Module):
    """
    Dense Dynamic graph convolution block
    """
    def __init__(self, in_channels, out_channels=64, kernel_size=9, dilation=1, conv='edge', act='relu', norm=None, bias=True, **kwargs):
        super(DenseDynBlock, self).__init__()
        self.body = DynConv(in_channels, out_channels, kernel_size, dilation, conv,
                            act, norm, bias, **kwargs)

    def forward(self, x, batch=None, edge_index=None):
        dense = self.body(x, batch, edge_index)
        return torch.cat((x, dense), 1), batch


class ResGraphBlock(nn.Module):
    """
    Residual Static graph convolution block
    """
    def __init__(self, channels,  conv='edge', act='relu', norm=None, bias=True, heads=8,  res_scale=1):
        super(ResGraphBlock, self).__init__()
        self.body = GraphConv(channels, channels, conv, act, norm, bias, heads)
        self.res_scale = res_scale

    def forward(self, x, edge_index):
        return self.body(x, edge_index) + x*self.res_scale, edge_index


class DenseGraphBlock(nn.Module):
    """
    Dense Static graph convolution block
    """
    def __init__(self, in_channels,  out_channels, conv='edge', act='relu', norm=None, bias=True, heads=8):
        super(DenseGraphBlock, self).__init__()
        self.body = GraphConv(in_channels, out_channels, conv, act, norm, bias, heads)

    def forward(self, x, edge_index):
        dense = self.body(x, edge_index)
        return torch.cat((x, dense), 1), edge_index

