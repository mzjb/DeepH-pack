import os
from typing import Union, Tuple
from math import ceil, sqrt

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import LayerNorm, PairNorm, InstanceNorm
from torch_geometric.typing import PairTensor, Adj, OptTensor, Size
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import softmax
from torch_geometric.nn.models.dimenet import BesselBasisLayer
from torch_scatter import scatter_add, scatter
import numpy as np
from scipy.special import comb

from .from_se3_transformer import SphericalHarmonics
from .from_schnetpack import GaussianBasis
from .from_PyG_future import GraphNorm, DiffGroupNorm
from .from_HermNet import RBF, cosine_cutoff, ShiftedSoftplus, _eps


class ExpBernsteinBasis(nn.Module):
    def __init__(self, K, gamma, cutoff, trainable=True):
        super(ExpBernsteinBasis, self).__init__()
        self.K = K
        if trainable:
            self.gamma = nn.Parameter(torch.tensor(gamma))
        else:
            self.gamma = torch.tensor(gamma)
        self.register_buffer('cutoff', torch.tensor(cutoff))
        self.register_buffer('comb_k', torch.Tensor(comb(K - 1, np.arange(K))))

    def forward(self, distances):
        f_zero = torch.zeros_like(distances)
        f_cut = torch.where(distances < self.cutoff, torch.exp(
            -(distances ** 2) / (self.cutoff ** 2 - distances ** 2)), f_zero)
        x = torch.exp(-self.gamma * distances)
        out = []
        for k in range(self.K):
            out.append((x ** k) * ((1 - x) ** (self.K - 1 - k)))
        out = torch.stack(out, dim=-1)
        out = out * self.comb_k[None, :] * f_cut[:, None]
        return out


def get_spherical_from_cartesian(cartesian, cartesian_x=1, cartesian_y=2, cartesian_z=0):
    spherical = torch.zeros_like(cartesian[..., 0:2])
    r_xy = cartesian[..., cartesian_x] ** 2 + cartesian[..., cartesian_y] ** 2
    spherical[..., 0] = torch.atan2(torch.sqrt(r_xy), cartesian[..., cartesian_z])
    spherical[..., 1] = torch.atan2(cartesian[..., cartesian_y], cartesian[..., cartesian_x])
    return spherical


class SphericalHarmonicsBasis(nn.Module):
    def __init__(self, num_l=5):
        super(SphericalHarmonicsBasis, self).__init__()
        self.num_l = num_l

    def forward(self, edge_attr):
        r_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]
        r_vec_sp = get_spherical_from_cartesian(r_vec)
        sph_harm_func = SphericalHarmonics()

        angular_expansion = []
        for l in range(self.num_l):
            angular_expansion.append(sph_harm_func.get(l, r_vec_sp[:, 0], r_vec_sp[:, 1]))
        angular_expansion = torch.cat(angular_expansion, dim=-1)

        return angular_expansion


"""
The class CGConv below is extended from "https://github.com/rusty1s/pytorch_geometric", which has the MIT License below

---------------------------------------------------------------------------
Copyright (c) 2020 Matthias Fey <matthias.fey@tu-dortmund.de>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
class CGConv(MessagePassing):
    def __init__(self, channels: Union[int, Tuple[int, int]], dim: int = 0,
                 aggr: str = 'add', normalization: str = None,
                 bias: bool = True, if_exp: bool = False, **kwargs):
        super(CGConv, self).__init__(aggr=aggr, flow="source_to_target", **kwargs)
        self.channels = channels
        self.dim = dim
        self.normalization = normalization
        self.if_exp = if_exp

        if isinstance(channels, int):
            channels = (channels, channels)

        self.lin_f = nn.Linear(sum(channels) + dim, channels[1], bias=bias)
        self.lin_s = nn.Linear(sum(channels) + dim, channels[1], bias=bias)
        if self.normalization == 'BatchNorm':
            self.bn = nn.BatchNorm1d(channels[1], track_running_stats=True)
        elif self.normalization == 'LayerNorm':
            self.ln = LayerNorm(channels[1])
        elif self.normalization == 'PairNorm':
            self.pn = PairNorm(channels[1])
        elif self.normalization == 'InstanceNorm':
            self.instance_norm = InstanceNorm(channels[1])
        elif self.normalization == 'GraphNorm':
            self.gn = GraphNorm(channels[1])
        elif self.normalization == 'DiffGroupNorm':
            self.group_norm = DiffGroupNorm(channels[1], 128)
        elif self.normalization is None:
            pass
        else:
            raise ValueError('Unknown normalization function: {}'.format(normalization))

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.normalization == 'BatchNorm':
            self.bn.reset_parameters()

    def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor, batch, distance, size: Size = None) -> torch.Tensor:
        """"""
        if isinstance(x, torch.Tensor):
            x: PairTensor = (x, x)

        # propagate_type: (x: PairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, distance=distance, size=size)
        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        elif self.normalization == 'LayerNorm':
            out = self.ln(out, batch)
        elif self.normalization == 'PairNorm':
            out = self.pn(out, batch)
        elif self.normalization == 'InstanceNorm':
            out = self.instance_norm(out, batch)
        elif self.normalization == 'GraphNorm':
            out = self.gn(out, batch)
        elif self.normalization == 'DiffGroupNorm':
            out = self.group_norm(out)
        out += x[1]
        return out

    def message(self, x_i, x_j, edge_attr: OptTensor, distance) -> torch.Tensor:
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))
        if self.if_exp:
            sigma = 3
            n = 2
            out = out * torch.exp(-distance ** n / sigma ** n / 2).view(-1, 1)
        return out

    def __repr__(self):
        return '{}({}, dim={})'.format(self.__class__.__name__, self.channels, self.dim)


class GAT_Crystal(MessagePassing):
    def __init__(self, in_features, out_features, edge_dim, heads, concat=False, normalization: str = None,
                 dropout=0, bias=True, **kwargs):
        super(GAT_Crystal, self).__init__(node_dim=0, aggr='add', flow='target_to_source', **kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.neg_slope = 0.2
        self.prelu = nn.PReLU()
        self.bn1 = nn.BatchNorm1d(heads)
        self.W = nn.Parameter(torch.Tensor(in_features + edge_dim, heads * out_features))
        self.att = nn.Parameter(torch.Tensor(1, heads, 2 * out_features))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_features))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.normalization = normalization
        if self.normalization == 'BatchNorm':
            self.bn = nn.BatchNorm1d(out_features, track_running_stats=True)
        elif self.normalization == 'LayerNorm':
            self.ln = LayerNorm(out_features)
        elif self.normalization == 'PairNorm':
            self.pn = PairNorm(out_features)
        elif self.normalization == 'InstanceNorm':
            self.instance_norm = InstanceNorm(out_features)
        elif self.normalization == 'GraphNorm':
            self.gn = GraphNorm(out_features)
        elif self.normalization == 'DiffGroupNorm':
            self.group_norm = DiffGroupNorm(out_features, 128)
        elif self.normalization is None:
            pass
        else:
            raise ValueError('Unknown normalization function: {}'.format(normalization))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.W)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, batch, distance):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        elif self.normalization == 'LayerNorm':
            out = self.ln(out, batch)
        elif self.normalization == 'PairNorm':
            out = self.pn(out, batch)
        elif self.normalization == 'InstanceNorm':
            out = self.instance_norm(out, batch)
        elif self.normalization == 'GraphNorm':
            out = self.gn(out, batch)
        elif self.normalization == 'DiffGroupNorm':
            out = self.group_norm(out)
        return out

    def message(self, edge_index_i, x_i, x_j, size_i, index, ptr: OptTensor, edge_attr):
        x_i = torch.cat([x_i, edge_attr], dim=-1)
        x_j = torch.cat([x_j, edge_attr], dim=-1)

        x_i = F.softplus(torch.matmul(x_i, self.W))
        x_j = F.softplus(torch.matmul(x_j, self.W))
        x_i = x_i.view(-1, self.heads, self.out_features)
        x_j = x_j.view(-1, self.heads, self.out_features)

        alpha = F.softplus((torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1))
        alpha = F.softplus(self.bn1(alpha))

        alpha = softmax(alpha, index, ptr, size_i)

        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out, x):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_features)
        else:
            aggr_out = aggr_out.mean(dim=1)
        if self.bias is not None:  aggr_out = aggr_out + self.bias
        return aggr_out


class PaninnNodeFea():
    def __init__(self, node_fea_s, node_fea_v=None):
        self.node_fea_s = node_fea_s
        if node_fea_v == None:
            self.node_fea_v = torch.zeros(node_fea_s.shape[0], node_fea_s.shape[1], 3, dtype=node_fea_s.dtype,
                                          device=node_fea_s.device)
        else:
            self.node_fea_v = node_fea_v

    def __add__(self, other):
        return PaninnNodeFea(self.node_fea_s + other.node_fea_s, self.node_fea_v + other.node_fea_v)


class PAINN(nn.Module):
    def __init__(self, in_features, edge_dim, rc: float, l: int, normalization):
        super(PAINN, self).__init__()
        self.ms1 = nn.Linear(in_features, in_features)
        self.ssp = ShiftedSoftplus()
        self.ms2 = nn.Linear(in_features, in_features * 3)

        self.rbf = RBF(rc, l)
        self.mv = nn.Linear(l, in_features * 3)
        self.fc = cosine_cutoff(rc)

        self.us1 = nn.Linear(in_features * 2, in_features)
        self.us2 = nn.Linear(in_features, in_features * 3)

        self.normalization = normalization
        if self.normalization == 'BatchNorm':
            self.bn = nn.BatchNorm1d(in_features, track_running_stats=True)
        elif self.normalization == 'LayerNorm':
            self.ln = LayerNorm(in_features)
        elif self.normalization == 'PairNorm':
            self.pn = PairNorm(in_features)
        elif self.normalization == 'InstanceNorm':
            self.instance_norm = InstanceNorm(in_features)
        elif self.normalization == 'GraphNorm':
            self.gn = GraphNorm(in_features)
        elif self.normalization == 'DiffGroupNorm':
            self.group_norm = DiffGroupNorm(in_features, 128)
        elif self.normalization is None or self.normalization == 'None':
            pass
        else:
            raise ValueError('Unknown normalization function: {}'.format(normalization))

    def forward(self, x: Union[torch.Tensor, PairTensor], edge_index: Adj,
                edge_attr: OptTensor, batch, edge_vec) -> torch.Tensor:
        r = torch.sqrt((edge_vec ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
        sj = x.node_fea_s[edge_index[1, :]]
        vj = x.node_fea_v[edge_index[1, :]]

        phi = self.ms2(self.ssp(self.ms1(sj)))
        w = self.fc(r) * self.mv(self.rbf(r))
        v_, s_, r_ = torch.chunk(phi * w, 3, dim=-1)

        ds_update = s_
        dv_update = vj * v_.unsqueeze(-1) + r_.unsqueeze(-1) * (edge_vec / r).unsqueeze(1)

        ds = scatter(ds_update, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce='mean')
        dv = scatter(dv_update, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce='mean')
        x = x + PaninnNodeFea(ds, dv)

        sj = x.node_fea_s[edge_index[1, :]]
        vj = x.node_fea_v[edge_index[1, :]]
        norm = torch.sqrt((vj ** 2).sum(dim=-1) + _eps)
        s = torch.cat([norm, sj], dim=-1)
        sj = self.us2(self.ssp(self.us1(s)))

        uv = scatter(vj, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce='mean')
        norm = torch.sqrt((uv ** 2).sum(dim=-1) + _eps).unsqueeze(-1)
        s_ = scatter(sj, edge_index[0], dim=0, dim_size=x.node_fea_s.shape[0], reduce='mean')
        avv, asv, ass = torch.chunk(s_, 3, dim=-1)

        ds = ((uv / norm) ** 2).sum(dim=-1) * asv + ass
        dv = uv * avv.unsqueeze(-1)

        if self.normalization == 'BatchNorm':
            ds = self.bn(ds)
        elif self.normalization == 'LayerNorm':
            ds = self.ln(ds, batch)
        elif self.normalization == 'PairNorm':
            ds = self.pn(ds, batch)
        elif self.normalization == 'InstanceNorm':
            ds = self.instance_norm(ds, batch)
        elif self.normalization == 'GraphNorm':
            ds = self.gn(ds, batch)
        elif self.normalization == 'DiffGroupNorm':
            ds = self.group_norm(ds)

        x = x + PaninnNodeFea(ds, dv)

        return x


class MPLayer(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, out_edge_fea_len, if_exp, if_edge_update, normalization,
                 atom_update_net, gauss_stop, output_layer=False):
        super(MPLayer, self).__init__()
        if atom_update_net == 'CGConv':
            self.cgconv = CGConv(channels=in_atom_fea_len,
                                 dim=in_edge_fea_len,
                                 aggr='add',
                                 normalization=normalization,
                                 if_exp=if_exp)
        elif atom_update_net == 'GAT':
            self.cgconv = GAT_Crystal(
                in_features=in_atom_fea_len,
                out_features=in_atom_fea_len,
                edge_dim=in_edge_fea_len,
                heads=3,
                normalization=normalization
            )
        elif atom_update_net == 'PAINN':
            self.cgconv = PAINN(
                in_features=in_atom_fea_len,
                edge_dim=in_edge_fea_len,
                rc=gauss_stop,
                l=64,
                normalization=normalization
            )

        self.if_edge_update = if_edge_update
        self.atom_update_net = atom_update_net
        if if_edge_update:
            if output_layer:
                self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 128),
                                           nn.SiLU(),
                                           nn.Linear(128, out_edge_fea_len),
                                           )
            else:
                self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2, 128),
                                           nn.SiLU(),
                                           nn.Linear(128, out_edge_fea_len),
                                           nn.SiLU(),
                                           )

    def forward(self, atom_fea, edge_idx, edge_fea, batch, distance, edge_vec):
        if self.atom_update_net == 'PAINN':
            atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, edge_vec)
            atom_fea_s = atom_fea.node_fea_s
        else:
            atom_fea = self.cgconv(atom_fea, edge_idx, edge_fea, batch, distance)
            atom_fea_s = atom_fea
        if self.if_edge_update:
            row, col = edge_idx
            edge_fea = self.e_lin(torch.cat([atom_fea_s[row], atom_fea_s[col], edge_fea], dim=-1))
            return atom_fea, edge_fea
        else:
            return atom_fea


class LCMPLayer(nn.Module):
    def __init__(self, in_atom_fea_len, in_edge_fea_len, out_edge_fea_len, num_l,
                 normalization: str = None, bias: bool = True, if_exp: bool = False):
        super(LCMPLayer, self).__init__()
        self.in_atom_fea_len = in_atom_fea_len
        self.normalization = normalization
        self.if_exp = if_exp

        self.lin_f = nn.Linear(in_atom_fea_len * 2 + in_edge_fea_len, in_atom_fea_len, bias=bias)
        self.lin_s = nn.Linear(in_atom_fea_len * 2 + in_edge_fea_len, in_atom_fea_len, bias=bias)
        self.bn = nn.BatchNorm1d(in_atom_fea_len, track_running_stats=True)

        self.e_lin = nn.Sequential(nn.Linear(in_edge_fea_len + in_atom_fea_len * 2 - num_l ** 2, 128),
                                   nn.SiLU(),
                                   nn.Linear(128, out_edge_fea_len)
                                   )
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_f.reset_parameters()
        self.lin_s.reset_parameters()
        if self.normalization == 'BatchNorm':
            self.bn.reset_parameters()

    def forward(self, atom_fea, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                huge_structure, output_final_layer_neuron):
        if huge_structure:
            sub_graph_batch_num = 8

            sub_graph_num = sub_atom_idx.shape[0]
            sub_graph_batch_size = ceil(sub_graph_num / sub_graph_batch_num)

            num_edge = edge_fea.shape[0]
            vf_update = torch.zeros((num_edge * 2, self.in_atom_fea_len)).type(torch.get_default_dtype()).to(atom_fea.device)
            for sub_graph_batch_index in range(sub_graph_batch_num):
                if sub_graph_batch_index == sub_graph_batch_num - 1:
                    sub_graph_idx = slice(sub_graph_batch_size * sub_graph_batch_index, sub_graph_num)
                else:
                    sub_graph_idx = slice(sub_graph_batch_size * sub_graph_batch_index,
                                          sub_graph_batch_size * (sub_graph_batch_index + 1))

                sub_atom_idx_batch = sub_atom_idx[sub_graph_idx]
                sub_edge_idx_batch = sub_edge_idx[sub_graph_idx]
                sub_edge_ang_batch = sub_edge_ang[sub_graph_idx]
                sub_index_batch = sub_index[sub_graph_idx]

                z = torch.cat([atom_fea[sub_atom_idx_batch][:, 0, :], atom_fea[sub_atom_idx_batch][:, 1, :],
                               edge_fea[sub_edge_idx_batch], sub_edge_ang_batch], dim=-1)
                out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

                if self.if_exp:
                    sigma = 3
                    n = 2
                    out = out * torch.exp(-distance[sub_edge_idx_batch] ** n / sigma ** n / 2).view(-1, 1)

                vf_update += scatter_add(out, sub_index_batch, dim=0, dim_size=num_edge * 2)

            if self.normalization == 'BatchNorm':
                vf_update = self.bn(vf_update)
            vf_update = vf_update.reshape(num_edge, 2, -1)
            if output_final_layer_neuron != '':
                final_layer_neuron = torch.cat([vf_update[:, 0, :], vf_update[:, 1, :], edge_fea],
                                               dim=-1).detach().cpu().numpy()
                np.save(os.path.join(output_final_layer_neuron, 'final_layer_neuron.npy'), final_layer_neuron)
            out = self.e_lin(torch.cat([vf_update[:, 0, :], vf_update[:, 1, :], edge_fea], dim=-1))

            return out

        num_edge = edge_fea.shape[0]
        z = torch.cat(
            [atom_fea[sub_atom_idx][:, 0, :], atom_fea[sub_atom_idx][:, 1, :], edge_fea[sub_edge_idx], sub_edge_ang],
            dim=-1)
        out = self.lin_f(z).sigmoid() * F.softplus(self.lin_s(z))

        if self.if_exp:
            sigma = 3
            n = 2
            out = out * torch.exp(-distance[sub_edge_idx] ** n / sigma ** n / 2).view(-1, 1)

        out = scatter_add(out, sub_index, dim=0)
        if self.normalization == 'BatchNorm':
            out = self.bn(out)
        out = out.reshape(num_edge, 2, -1)
        if output_final_layer_neuron != '':
            final_layer_neuron = torch.cat([out[:, 0, :], out[:, 1, :], edge_fea], dim=-1).detach().cpu().numpy()
            np.save(os.path.join(output_final_layer_neuron, 'final_layer_neuron.npy'), final_layer_neuron)
        out = self.e_lin(torch.cat([out[:, 0, :], out[:, 1, :], edge_fea], dim=-1))
        return out


class MultipleLinear(nn.Module):
    def __init__(self, num_linear: int, in_fea_len: int, out_fea_len: int, bias: bool = True) -> None:
        super(MultipleLinear, self).__init__()
        self.num_linear = num_linear
        self.out_fea_len = out_fea_len
        self.weight = nn.Parameter(torch.Tensor(num_linear, in_fea_len, out_fea_len))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_linear, out_fea_len))
        else:
            self.register_parameter('bias', None)
        # self.ln = LayerNorm(num_linear * out_fea_len)
        # self.gn = GraphNorm(out_fea_len)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor, batch_edge: torch.Tensor) -> torch.Tensor:
        output = torch.matmul(input, self.weight)

        if self.bias is not None:
            output += self.bias[:, None, :]
        return output


class HGNN(nn.Module):
    def __init__(self, num_species, in_atom_fea_len, in_edge_fea_len, num_orbital,
                 distance_expansion, gauss_stop, if_exp, if_MultipleLinear, if_edge_update, if_lcmp,
                 normalization, atom_update_net, separate_onsite,
                 trainable_gaussians, type_affine, num_l=5):
        super(HGNN, self).__init__()
        self.num_species = num_species
        self.embed = nn.Embedding(num_species + 5, in_atom_fea_len)

        # pair-type aware affine
        if type_affine:
            self.type_affine = nn.Embedding(
                num_species ** 2, 2,
                _weight=torch.stack([torch.ones(num_species ** 2), torch.zeros(num_species ** 2)], dim=-1)
            )
        else:
            self.type_affine = None

        if if_edge_update or (if_edge_update is False and if_lcmp is False):
            distance_expansion_len = in_edge_fea_len
        else:
            distance_expansion_len = in_edge_fea_len - num_l ** 2
        if distance_expansion == 'GaussianBasis':
            self.distance_expansion = GaussianBasis(
                0.0, gauss_stop, distance_expansion_len, trainable=trainable_gaussians
            )
        elif distance_expansion == 'BesselBasis':
            self.distance_expansion = BesselBasisLayer(distance_expansion_len, gauss_stop, envelope_exponent=5)
        elif distance_expansion == 'ExpBernsteinBasis':
            self.distance_expansion = ExpBernsteinBasis(K=distance_expansion_len, gamma=0.5, cutoff=gauss_stop,
                                                        trainable=True)
        else:
            raise ValueError('Unknown distance expansion function: {}'.format(distance_expansion))

        self.if_MultipleLinear = if_MultipleLinear
        self.if_edge_update = if_edge_update
        self.if_lcmp = if_lcmp
        self.atom_update_net = atom_update_net
        self.separate_onsite = separate_onsite

        if if_lcmp == True:
            mp_output_edge_fea_len = in_edge_fea_len - num_l ** 2
        else:
            assert if_MultipleLinear == False
            mp_output_edge_fea_len = in_edge_fea_len

        if if_edge_update == True:
            self.mp1 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp2 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp3 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp4 = MPLayer(in_atom_fea_len, in_edge_fea_len, in_edge_fea_len, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp5 = MPLayer(in_atom_fea_len, in_edge_fea_len, mp_output_edge_fea_len, if_exp, if_edge_update,
                               normalization, atom_update_net, gauss_stop)
        else:
            self.mp1 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp2 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp3 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp4 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)
            self.mp5 = MPLayer(in_atom_fea_len, distance_expansion_len, None, if_exp, if_edge_update, normalization,
                               atom_update_net, gauss_stop)

        if if_lcmp == True:
            if self.if_MultipleLinear == True:
                self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, 32, num_l, if_exp=if_exp)
                self.multiple_linear1 = MultipleLinear(num_orbital, 32, 16)
                self.multiple_linear2 = MultipleLinear(num_orbital, 16, 1)
            else:
                self.lcmp = LCMPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, num_l, if_exp=if_exp)
        else:
            self.mp_output = MPLayer(in_atom_fea_len, in_edge_fea_len, num_orbital, if_exp, if_edge_update=True,
                                     normalization=normalization, atom_update_net=atom_update_net,
                                     gauss_stop=gauss_stop, output_layer=True)


    def forward(self, atom_attr, edge_idx, edge_attr, batch,
                sub_atom_idx=None, sub_edge_idx=None, sub_edge_ang=None, sub_index=None,
                huge_structure=False, output_final_layer_neuron=''):
        batch_edge = batch[edge_idx[0]]
        atom_fea0 = self.embed(atom_attr)
        distance = edge_attr[:, 0]
        edge_vec = edge_attr[:, 1:4] - edge_attr[:, 4:7]
        if self.type_affine is None:
            edge_fea0 = self.distance_expansion(distance)
        else:
            affine_coeff = self.type_affine(self.num_species * atom_attr[edge_idx[0]] + atom_attr[edge_idx[1]])
            edge_fea0 = self.distance_expansion(distance * affine_coeff[:, 0] + affine_coeff[:, 1])
        if self.atom_update_net == "PAINN":
            atom_fea0 = PaninnNodeFea(atom_fea0)

        if self.if_edge_update == True:
            atom_fea, edge_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea, edge_fea = self.mp2(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea, edge_fea = self.mp4(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
            atom_fea0, edge_fea0 = atom_fea0 + atom_fea, edge_fea0 + edge_fea
            atom_fea, edge_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)

            if self.if_lcmp == True:
                if self.atom_update_net == 'PAINN':
                    atom_fea_s = atom_fea.node_fea_s
                else:
                    atom_fea_s = atom_fea
                out = self.lcmp(atom_fea_s, edge_fea, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron)
            else:
                atom_fea, edge_fea = self.mp_output(atom_fea, edge_idx, edge_fea, batch, distance, edge_vec)
                out = edge_fea
        else:
            atom_fea = self.mp1(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea = self.mp2(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea0 = atom_fea0 + atom_fea
            atom_fea = self.mp3(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea = self.mp4(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
            atom_fea0 = atom_fea0 + atom_fea
            atom_fea = self.mp5(atom_fea0, edge_idx, edge_fea0, batch, distance, edge_vec)

            if self.atom_update_net == 'PAINN':
                atom_fea_s = atom_fea.node_fea_s
            else:
                atom_fea_s = atom_fea
            if self.if_lcmp == True:
                out = self.lcmp(atom_fea_s, edge_fea0, sub_atom_idx, sub_edge_idx, sub_edge_ang, sub_index, distance,
                                huge_structure, output_final_layer_neuron)
            else:
                atom_fea, edge_fea = self.mp_output(atom_fea, edge_idx, edge_fea0, batch, distance, edge_vec)
                out = edge_fea

        if self.if_MultipleLinear == True:
            out = self.multiple_linear1(F.silu(out), batch_edge)
            out = self.multiple_linear2(F.silu(out), batch_edge)
            out = out.T

        return out
