from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import softmax
from torch import Tensor

import torch_geometric
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_scatter as scatter

import math
import numpy as np



class TimeEncoder(torch.nn.Module):
    def __init__(self, dimension):
        super(TimeEncoder, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 1.5, dimension)))
            .float()
            .reshape(dimension, -1)
        )
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension))

    def reset_parameters(self):
        self.w.reset_parameters()

    def forward(self, t):
        t = torch.log(t + 1)
        t = t.unsqueeze(dim=1)
        output = torch.cos(self.w(t))
        return output


class Trans2GraphConv(nn.Module):
    def __init__(
        self, in_dim, out_dim, time_dim, num_node_types=19, n_heads=1,
        use_norm=True, use_dropout=True, use_time_encoder=True, use_backward=True, use_attention=True, use_time_interval=True, p=0.2, **kwargs
    ):
        super(Trans2GraphConv, self).__init__()
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.time_dim = time_dim
        self.num_node_types = num_node_types
        self.n_heads = n_heads
        
        if use_time_encoder:
            if use_time_interval:
                self.dk = (out_dim - self.time_dim) // n_heads
            else:
                self.dk = (out_dim) // n_heads
        else:
            self.dk = (out_dim) // n_heads
        self.sqrt_dk = math.sqrt(self.dk)
        
        self.use_norm = use_norm
        self.use_time_encoder = use_time_encoder
        self.use_backward = use_backward
        self.use_attention = use_attention
        self.use_time_interval = use_time_interval
        
        if use_attention == False:
            self.use_backward = False
            use_backward = False

        self.use_dropout = use_dropout
        if self.use_dropout:
            self.drop = nn.Dropout(p)
            self.att_drop = nn.Dropout(p)

        # K, Q, V
        self.k_linears = nn.ModuleList()
        self.q_linears = nn.ModuleList()
        self.v_linears = nn.ModuleList()
        # A
        self.a_linears = nn.ModuleList()
        # norm
        self.norms = nn.ModuleList()
        
        # TimeEncoder
        if self.use_time_encoder:
            self.emb = TimeEncoder(self.time_dim)
        
        # each node type corresponds to a K, Q, V, A, norm
        for t in range(self.num_node_types):
            if self.use_time_encoder:
                if self.use_time_interval:
                    self.k_linears.append(nn.Linear(self.in_dim, self.out_dim - self.time_dim * 2))
                    self.v_linears.append(nn.Linear(self.in_dim, self.out_dim - self.time_dim * 2))
                    self.q_linears.append(nn.Linear(self.in_dim, self.out_dim - self.time_dim * 2))
                else:
                    self.k_linears.append(nn.Linear(self.in_dim, self.out_dim - self.time_dim))
                    self.v_linears.append(nn.Linear(self.in_dim, self.out_dim - self.time_dim))
                    self.q_linears.append(nn.Linear(self.in_dim, self.out_dim - self.time_dim))
            else:
                self.k_linears.append(nn.Linear(self.in_dim, self.out_dim))
                self.v_linears.append(nn.Linear(self.in_dim, self.out_dim))
                self.q_linears.append(nn.Linear(self.in_dim, self.out_dim))               

            if self.use_backward:
                self.a_linears.append(nn.Linear(self.out_dim * 2, self.in_dim))
            else:
                self.a_linears.append(nn.Linear(self.out_dim, self.in_dim))

            if self.use_norm:
                self.norms.append(nn.LayerNorm(self.in_dim))
        
    def reset_parameters(self):
        for t in range(self.num_node_types):
            self.k_linears[t].reset_parameters()
            self.v_linears[t].reset_parameters()
            self.q_linears[t].reset_parameters()
            self.a_linears[t].reset_parameters()
            
            if self.use_norm:
                self.norms[t].reset_parameters()
            
        if self.use_time_encoder:
            self.emb.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def propagate(self, edge_index, x, edge_attr):
        row, col = edge_index
        x_j = x[row]
        x_i = x[col]
        out1, out2 = self.message(x_i, x_j, edge_index, edge_attr)
        out = torch.zeros(x.size(0), self.out_dim).to(out1.device)
        out1, out2 = self.aggregate(out1, out2, out, edge_index)
        out = self.update(out1, out2, x, edge_index, edge_attr)
        return out
    
    def aggregate(self, x_j, x_i, out, edge_index):
        row, col = edge_index
        aggr_out1, aggr_out2 = None, None
        if self.use_attention:
            reduce_type = 'sum'
        else:
            reduce_type = 'mean'
        aggr_out1 = scatter.scatter(src=x_j, index=col, dim=-2, out=out, reduce=reduce_type)
        if self.use_backward:
            aggr_out2 = scatter.scatter(src=x_i, index=row, dim=-2, out=out, reduce=reduce_type)
        return aggr_out1, aggr_out2

    def message(self, x_i, x_j, edge_index, edge_attr):
        """
        process messages
        source: j => target: i
        """
        res_att = torch.zeros(edge_index.size(1), self.n_heads).to(x_i.device)  # attention [num_edges, n_heads]
        res_msg = torch.zeros(edge_index.size(1), self.n_heads, self.dk).to(x_i.device)  # messages [num_edges, n_heads, dk]

        res_att_inverse = torch.zeros(edge_index.size(1), self.n_heads).to(x_i.device)  # attention [num_edges, n_heads]
        res_msg_inverse = torch.zeros(edge_index.size(1), self.n_heads, self.dk).to(x_i.device)  # messages [num_edges, n_heads, dk]

        for src_node_type in range(self.num_node_types):
            k_linear = self.k_linears[src_node_type]
            v_linear = self.v_linears[src_node_type]
            q_linear_inverse = self.q_linears[src_node_type]
            
            for trg_node_type in range(self.num_node_types):
                q_linear = self.q_linears[trg_node_type]
                k_linear_inverse = self.k_linears[trg_node_type]
                v_linear_inverse = self.v_linears[trg_node_type]
                
                # find the edge idx that matches the src_node_type and trg_node_type
                idx = (edge_attr[:, 3] == src_node_type) & (edge_attr[:, 4] == trg_node_type)

                if idx.sum() != 0:
                    src_x = x_j[idx]
                    trg_x = x_i[idx]
                    src_x = src_x.float()
                    trg_x = trg_x.float()

                    # calculate matrix Q and K
                    q_mat = q_linear(trg_x)
                    k_mat = k_linear(src_x)
                    
                    if self.use_time_encoder:
                        k_mat = torch.cat([k_mat, self.emb(edge_attr[idx, 1])], dim=1)
                        q_mat = torch.cat([q_mat, self.emb(edge_attr[idx, 2])], dim=1)
    
                    q_mat = q_mat.view(-1, self.n_heads, self.dk).transpose(-1, -2)
                    k_mat = k_mat.view(-1, self.n_heads, self.dk)

                    # multiply Q and K
                    res_att[idx] = (torch.bmm(k_mat, q_mat) / self.sqrt_dk).sum(dim=-1)
                    
                    # calculate matrix V
                    if self.use_time_encoder:
                        v_mat = torch.cat([v_linear(src_x), self.emb(edge_attr[idx, 1])], dim=1).view(-1, self.n_heads, self.dk)
                    else:
                        v_mat = v_linear(src_x).view(-1, self.n_heads, self.dk)
                    res_msg[idx] = v_mat
                    
                    # ================ inverse ================
                    if self.use_backward:
                        src_x = x_i[idx]
                        trg_x = x_j[idx]
                        src_x = src_x.float()
                        trg_x = trg_x.float()
                        
                        # calculate matrix Q and K
                        q_mat = q_linear_inverse(trg_x)
                        k_mat = k_linear_inverse(src_x)
                        
                        if self.use_time_encoder:
                            k_mat = torch.cat([k_mat, self.emb(edge_attr[idx, 2])], dim=1)
                            q_mat = torch.cat([q_mat, self.emb(edge_attr[idx, 1])], dim=1)

                        q_mat = q_mat.view(-1, self.n_heads, self.dk).transpose(-1, -2)
                        k_mat = k_mat.view(-1, self.n_heads, self.dk)

                        # multiply Q and K
                        res_att_inverse[idx] = (torch.bmm(k_mat, q_mat) / self.sqrt_dk).sum(dim=-1)
                        
                        # calculate matrix V
                        if self.use_time_encoder:
                            v_mat = torch.cat([v_linear_inverse(src_x), self.emb(edge_attr[idx, 2])], dim=1).view(-1, self.n_heads, self.dk)
                        else:
                            v_mat = v_linear_inverse(src_x).view(-1, self.n_heads, self.dk)
                        res_msg_inverse[idx] = v_mat
        
        # perform softmax operation
        res_att = softmax(res_att, edge_index[1])
        res_att_inverse = softmax(res_att_inverse, edge_index[0])

        # attention dropout
        if self.use_dropout:
            res_att = self.att_drop(res_att)
            res_att_inverse = self.att_drop(res_att_inverse)
        
        # attention * message
        res, res_inverse = None, None

        if self.use_time_interval:
            res_dim =  self.out_dim - self.time_dim
        else:
            res_dim =  self.out_dim

        if self.use_attention:
            if self.use_time_encoder:
                res = res_msg * res_att.view(-1, self.n_heads, 1)
                res = res.view(-1, res_dim)
                res_inverse = res_msg_inverse * res_att_inverse.view(-1, self.n_heads, 1)
                res_inverse = res_inverse.view(-1, res_dim)
                
                # concatenate time interval information
                if self.use_time_interval:
                    res = torch.cat([res, self.emb(edge_attr[:, 0])], dim=1)
                    res_inverse = torch.cat([res_inverse, self.emb(edge_attr[:, 0])], dim=1)
            else:
                res = res_msg * res_att.view(-1, self.n_heads, 1)
                res = res.view(-1, self.out_dim)
                res_inverse = res_msg_inverse * res_att_inverse.view(-1, self.n_heads, 1)
                res_inverse = res_inverse.view(-1, self.out_dim)
        
        else:
            if self.use_time_encoder:
                res = res_msg
                res = res.view(-1, res_dim)
                res_inverse = res_msg_inverse
                res_inverse = res_inverse.view(-1, res_dim)
                
                # concatenate time interval information
                if self.use_time_interval:
                    res = torch.cat([res, self.emb(edge_attr[:, 0])], dim=1)
                    res_inverse = torch.cat([res_inverse, self.emb(edge_attr[:, 0])], dim=1)
            else:
                res = res_msg
                res = res.view(-1, self.out_dim)
                res_inverse = res_msg_inverse
                res_inverse = res_inverse.view(-1, self.out_dim)
        
        return res, res_inverse

    def update(self, aggr_out1, aggr_out2, x, edge_index, edge_attr):
        """
        update node representation
        """
        out = torch.zeros(aggr_out1.size(0), self.in_dim).to(x.device)  # [num_nodes, in_dim]
        idx_list = torch.tensor(np.arange(int(aggr_out1.size(0))))  # [0, 1, 2, ..., num_nodes-1]
        
        for trg_node_type in range(self.num_node_types):
            a_linear = self.a_linears[trg_node_type]
            
            # find the node idx of current node type
            idx = (idx_list % self.num_node_types == trg_node_type)
        
            # aggregate self and neighbor information
            if self.use_backward:
                if self.use_dropout:
                    out[idx] = self.drop(F.relu(a_linear(torch.cat([aggr_out1[idx], aggr_out2[idx]], dim=1)))) + x[idx].float()
                else:
                    out[idx] = F.relu(a_linear(torch.cat([aggr_out1[idx], aggr_out2[idx]], dim=1))) + x[idx].float()
            else:
                if self.use_dropout:
                    out[idx] = self.drop(F.relu(a_linear(aggr_out1[idx]))) + x[idx].float()
                else:
                    out[idx] = F.relu(a_linear(aggr_out1[idx])) + x[idx].float()
                
            if self.use_norm:
                out[idx] = self.norms[trg_node_type](out[idx])
                
        return out


class Trans2Graph(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_channels, hidden_channels, num_layers, num_node_types=19, n_heads=1,
        use_norm=True, use_dropout=True, use_time_encoder=True, use_backward=True, use_attention=True, use_time_interval=True, p=0.2, ppr_emb=True
    ):
        super(Trans2Graph, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_channels = time_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.num_node_types = num_node_types
        self.ppr_emb = ppr_emb
        self.use_dropout = use_dropout
        
        # dropout
        if self.use_dropout: self.drop = nn.Dropout(p)
        # conv + linear
        self.convs = torch.nn.ModuleList()
        self.convs.append(Trans2GraphConv(in_channels, hidden_channels, time_channels, num_node_types, n_heads, use_norm, use_dropout, use_time_encoder, use_backward, use_attention, use_time_interval, p))
        for i in range(num_layers - 1):
            self.convs.append(Trans2GraphConv(hidden_channels, hidden_channels, time_channels, num_node_types, n_heads, use_norm, use_dropout, use_time_encoder, use_backward, use_attention, use_time_interval, p))
        self.lin = nn.Linear(in_channels, out_channels)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr, batch):
        if not self.ppr_emb:
            x = torch.zeros([x.size(0), 19]).to(x.device)
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
            
        x = global_mean_pool(x, batch)
        x = self.lin(x.float())
        return x.log_softmax(dim=-1)
        
    def get_embedding(self, x, edge_index, edge_attr, batch):
        if not self.ppr_emb:
            x = torch.zeros([x.size(0), 19]).to(x.device)
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return x
