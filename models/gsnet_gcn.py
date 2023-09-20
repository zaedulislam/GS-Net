from __future__ import absolute_import
import torch.nn as nn
from models.gsnet_gcn_conv import GSNetGraphConv
from models.graph_non_local import GraphNonLocal
from nets.non_local_embedded_gaussian import NONLocalBlock2D


class GraphConvBlockI(nn.Module):
    def __init__(self, adj, input_dim, output_dim, beta, p_dropout=None):
        super(GraphConvBlockI, self).__init__()

        self.gconv = GSNetGraphConv(input_dim, output_dim, adj, beta)
        self.ln = nn.LayerNorm(output_dim)

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, x0):
        x = self.gconv(x, x0)
        x = self.ln(x)
        
        if self.dropout is not None:
            x = self.dropout(x)

        return x


class GraphConvBlockII(nn.Module):
    def __init__(self, adj, input_dim, output_dim, beta, p_dropout=None):
        super(GraphConvBlockII, self).__init__()

        self.gconv = GSNetGraphConv(input_dim, output_dim, adj, beta)
        self.act = nn.GELU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, x0):
        x = self.gconv(x, x0)

        if self.dropout is not None:
            x = self.dropout(self.act(x))
        
        x = self.act(x)
        
        return x


class ResGraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, beta, p_dropout):
        super(ResGraphConv, self).__init__()

        self.gconv1 = GraphConvBlockI(adj, input_dim, hid_dim, beta, p_dropout)
        self.gconv2 = GraphConvBlockII(adj, hid_dim, output_dim, beta, p_dropout)

    def forward(self, x):
        residual = x
        out = self.gconv1(x, residual)
        out = self.gconv2(out, residual)
        return residual + out


class GraphNonLocal(nn.Module):
    def __init__(self, hid_dim, grouped_order, restored_order, group_size):
        super(GraphNonLocal, self).__init__()

        self.non_local = GraphNonLocal(hid_dim, sub_sample=group_size)
        self.grouped_order = grouped_order
        self.restored_order = restored_order

    def forward(self, x):
        out = x[:, self.grouped_order, :]
        out = self.non_local(out.transpose(1, 2)).transpose(1, 2)
        out = out[:, self.restored_order, :]
        return out


class GSNetGCN(nn.Module):
    def __init__(self, adj, hid_dim, beta, coords_dim=(2, 3), num_layers=4, p_dropout=None):
        super(GSNetGCN, self).__init__()

        self.gconv_input = GraphConvBlockII(adj, coords_dim[0], hid_dim, beta, p_dropout=p_dropout)

        _gconv_layers = []

        for i in range(num_layers):
            _gconv_layers.append(ResGraphConv(adj, hid_dim, hid_dim, hid_dim, beta, p_dropout=p_dropout))

        self.gconv_layers = nn.Sequential(*_gconv_layers)
        self.gconv_output = GSNetGraphConv(hid_dim, coords_dim[1], adj, beta) 
        self.non_local = NONLocalBlock2D(in_channels=hid_dim, sub_sample=False)

    def forward(self, x):
        x = x.squeeze() 
        x = x.permute(0,2,1)
        out = self.gconv_input(x, x)
        x_out = out
        out = self.gconv_layers(out)
        out = out.unsqueeze(2)
        out = out.permute(0,3,2,1)
        out = self.non_local(out)
        out = out.permute(0,3,1,2)
        out = out.squeeze()
        out = self.gconv_output(out, x_out)
        out = out.permute(0,2,1)
        out = out.unsqueeze(2)
        out = out.unsqueeze(4)
        
        return out
