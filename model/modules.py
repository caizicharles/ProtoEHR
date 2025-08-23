import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing as MP
from torch_geometric.typing import (Adj, OptPairTensor, OptTensor, Size)
from torch import Tensor
from typing import Optional, Union
import dgl
import dgl.function as fn


########## Deepr ##########


class TimeGapEmbedding(nn.Module):

    def __init__(self, embed_dim, device):
        super().__init__()

        self.boundary = torch.tensor([1, 3, 6, 12], dtype=torch.float32, device=device)
        self.time_embed = nn.Embedding(5, embed_dim)

    def forward(self, visit_rel_times):

        visit_rel_times = visit_rel_times / 4
        indices = torch.bucketize(visit_rel_times, self.boundary, right=True)

        return self.time_embed(indices)


########## KerPrint ##########


class Aggregator(nn.Module):

    def __init__(self, in_dim, out_dim, dropout):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout

        self.message_dropout = nn.Dropout(dropout)

        self.W = nn.Linear(self.in_dim, self.out_dim)

        self.activation = nn.LeakyReLU()
        self._init_weight()

    def _init_weight(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, g, entity_embed):
        g = g.local_var()
        g.ndata['node'] = entity_embed

        g.update_all(dgl.function.u_mul_e('node', 'att', 'side'),
                         dgl.function.sum('side', 'N_h'))

        out = self.activation(self.W(g.ndata['node'] + g.ndata['N_h']))

        out = self.message_dropout(out)

        return out


########## GraphCare ##########


class BiAttentionGNNConv(MP):
    def __init__(self,
                 nn: torch.nn.Module,
                 eps: float = 0.,
                 train_eps: bool = False,
                 edge_dim: Optional[int] = None,
                 edge_attn=True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.nn = nn
        self.initial_eps = eps
        self.edge_attn = edge_attn
        if edge_attn:
            self.W_R = torch.nn.Linear(edge_dim, 1)
        else:
            self.W_R = None

        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        self.reset_parameters()

    def reset_parameters(self):
        self.nn.reset_parameters()
        self.eps.data.fill_(self.initial_eps)
        if self.W_R is not None:
            self.W_R.reset_parameters()

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                edge_attr: OptTensor = None,
                size: Size = None,
                attn: Tensor = None) -> Tensor:

        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size, attn=attn)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        if self.W_R is not None:
            w_rel = self.W_R(edge_attr)
        else:
            w_rel = None

        return self.nn(out), w_rel

    def message(self, x_j: Tensor, edge_attr: Tensor, attn: Tensor) -> Tensor:
        if self.edge_attn:
            w_rel = self.W_R(edge_attr)
            out = (x_j * attn + w_rel * edge_attr).relu()
        else:
            out = (x_j * attn).relu()
        return out

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'


def masked_softmax(src: Tensor, mask: Tensor, dim: int = -1) -> Tensor:
    out = src.masked_fill(~mask, float('-inf'))
    out = torch.softmax(out, dim=dim)
    out = out.masked_fill(~mask, 0)

    return out + 1e-8  # Add small constant to avoid numerical issues


########## ProtoEHR ##########


class CompGCNConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 act=lambda x: x,
                 bias=True,
                 drop_rate=0.,
                 opn='rotate',
                 num_base=-1,
                 num_rel=None):

        super(CompGCNConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.act = act  # activation function
        self.device = None
        self.rel = None
        self.opn = opn

        # relation-type specific parameter
        self.in_w = self.get_param([in_channels, out_channels])
        self.out_w = self.get_param([in_channels, out_channels])
        self.loop_w = self.get_param([in_channels, out_channels])
        self.w_rel = self.get_param([in_channels, out_channels])  # transform embedding of relations to next layer
        self.loop_rel = self.get_param([1, in_channels])  # self-loop embedding

        self.drop = nn.Dropout(drop_rate)
        self.bn = torch.nn.BatchNorm1d(out_channels)
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        if num_base > 0:
            self.rel_wt = self.get_param([num_rel * 2, num_base])
        else:
            self.rel_wt = None

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param

    def message_func(self, edges):
        edge_type = edges.data['type']  # [E, 1]
        edge_num = edge_type.shape[0]
        edge_data = self.comp(edges.src['h'], self.rel[edge_type])  # [E, in_channel]
        # NOTE: first half edges are all in-directions, last half edges are out-directions.
        msg = torch.cat([torch.matmul(edge_data[:edge_num // 2, :], self.in_w),
                         torch.matmul(edge_data[edge_num // 2:, :], self.out_w)])
        msg = msg * edges.data['norm'].reshape(-1, 1)  # [E, D] * [E, 1]
        return {'msg': msg}

    def reduce_func(self, nodes):
        return {'h': self.drop(nodes.data['h']) / 3}

    def comp(self, h, edge_data):

        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        def ccorr(a, b):
            # Compute the Fourier transforms
            a_fft = torch.fft.rfft(a, dim=-1)
            b_fft = torch.fft.rfft(b, dim=-1)

            # Compute the conjugate of the Fourier transform of `a`
            a_fft_conj = torch.conj(a_fft)

            # Perform element-wise multiplication in the Fourier domain
            result_fft = a_fft_conj * b_fft

            # Inverse Fourier transform to get the result in the time domain
            result = torch.fft.irfft(result_fft, n=a.shape[-1], dim=-1)

            return result
        
        def rotate(h, r):
            d = h.shape[-1]
            h_re, h_im = torch.split(h, d // 2, -1)
            r_re, r_im = torch.split(r, d // 2, -1)
            return torch.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)

        if self.opn == 'mult':
            return h * edge_data
        elif self.opn == 'sub':
            return h - edge_data
        elif self.opn == 'corr':
            return ccorr(h, edge_data.expand_as(h))
        elif self.opn == 'rotate':
            return rotate(h, edge_data.expand_as(h))
        else:
            raise KeyError(f'composition operator {self.opn} not recognized.')

    def forward(self, g: dgl.DGLGraph, x, rel_repr, edge_type, edge_norm):
        """
        :param g: dgl Graph, a graph without self-loop
        :param x: input node features, [V, in_channel]
        :param rel_repr: input relation features: 1. not using bases: [num_rel*2, in_channel]
                                                  2. using bases: [num_base, in_channel]
        :param edge_type: edge type, [E]
        :param edge_norm: edge normalization, [E]
        :return: x: output node features: [V, out_channel]
                 rel: output relation features: [num_rel*2, out_channel]
        """

        self.device = x.device
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = edge_type
        g.edata['norm'] = edge_norm

        if self.rel_wt is None:
            self.rel = rel_repr
        else:
            self.rel = torch.mm(self.rel_wt, rel_repr)  # [num_rel*2, num_base] @ [num_base, in_c]

        g.update_all(self.message_func, fn.sum(msg='msg', out='h'), self.reduce_func)
        x = g.ndata.pop('h') + torch.mm(self.comp(x, self.loop_rel), self.loop_w) / 3

        if self.bias is not None:
            x = x + self.bias

        x = self.bn(x)

        return self.act(x), torch.matmul(self.rel, self.w_rel)
    

class PrototypeLearning(nn.Module):
    def __init__(self,
                 prototype_num,
                 hidden_dim,
                 attn_iter=None):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.prototype_num = prototype_num
        
        self.prototypes = nn.Parameter(torch.Tensor(self.prototype_num, self.hidden_dim))
        self.prototypes.data.uniform_(-1 / self.prototype_num, 1 / self.prototype_num)

        self.attn_iter = attn_iter 
        self.attn = nn.MultiheadAttention(embed_dim=self.hidden_dim,
                                            num_heads=1,
                                            dropout=0.,
                                            batch_first=True)

    def forward(self, x, mask=None):
        prototypes = self.prototypes

        for _ in range(self.attn_iter):
            prototypes, weights = self.attn(prototypes, x, x, key_padding_mask=mask)

        return None, None, prototypes, weights


class PrototypePrediction(nn.Module):
    def __init__(self, prototype_num, hidden_dim, out_dim):
        super().__init__()

        self.prototype_num = prototype_num
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        
        self.projection = nn.Linear(self.hidden_dim, self.hidden_dim)

    def get_simC_prob(self, x, prototypes):
        def pairwise_cos_sim(x1: torch.Tensor, x2:torch.Tensor):
            x1 = F.normalize(x1,dim=-1)
            x2 = F.normalize(x2,dim=-1)
            sim = torch.matmul(x1, x2.transpose(-2, -1))
            return sim
        
        prob = pairwise_cos_sim(x, prototypes)
        prob = torch.softmax(prob, dim=-1)
        return prob

    def forward(self, x, prototypes):
        prob = self.get_simC_prob(x, prototypes)    # (B, K)
        x_prototype = torch.matmul(prob, prototypes) / self.prototype_num
        x = self.projection(x_prototype)

        return x
