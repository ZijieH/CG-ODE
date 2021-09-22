import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
import math
import lib.utils as utils
from torch_geometric.nn.inits import glorot
from torch_scatter import scatter_add





def normalize_graph_asymmetric(edge_index,edge_weight, num_nodes):
    '''

    :param edge_index: [num_edges,2], torch.LongTensor
    :param edge_weight: [num_edge]
    :param num_nodes:
    :return:
    '''
    assert (not torch.isnan(edge_weight).any())

    row, col = edge_index[0],edge_index[1]
    deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)  # [K*N]
    deg_inv_sqrt = deg.pow_(-1)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float("inf"), 0)
    if torch.isnan(deg_inv_sqrt).any():
        assert (torch.sum(deg == 0) == 0)
        assert (torch.sum(deg < 0) == 0)


    edge_weight_normalized = deg_inv_sqrt[row] * edge_weight  # [num_edge]

    # Reshape back
    assert (not torch.isnan(edge_weight_normalized).any())

    return edge_weight_normalized


class TemporalEncoding(nn.Module):

    def __init__(self, d_hid):
        super(TemporalEncoding, self).__init__()
        self.d_hid = d_hid
        self.div_term = torch.FloatTensor([1 / np.power(10000, 2 * (hid_j // 2) / self.d_hid) for hid_j in range(self.d_hid)]) #[20]
        self.div_term = torch.reshape(self.div_term,(1,-1))
        self.div_term = nn.Parameter(self.div_term,requires_grad=False)

    def forward(self, t):
        '''

        :param t: [n,1]
        :return:
        '''
        t = t.view(-1,1)
        t = t *200  # scale from [0,1] --> [0,200], align with 'attention is all you need'
        position_term = torch.matmul(t,self.div_term)
        position_term[:,0::2] = torch.sin(position_term[:,0::2])
        position_term[:,1::2] = torch.cos(position_term[:,1::2])

        return position_term


class GTrans(MessagePassing):
    '''
    Multiply attention by edgeweight
    '''

    def __init__(self, n_heads=1,d_input=6, d_output=6,dropout = 0.1,**kwargs):
        super(GTrans, self).__init__(aggr='add', **kwargs)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        self.d_input = d_input
        self.d_k = d_output//n_heads
        self.d_q = d_output//n_heads
        self.d_e = d_output//n_heads
        self.d_sqrt = math.sqrt(d_output//n_heads)


        #Attention Layer Initialization
        self.w_k_list = nn.ModuleList([nn.Linear(self.d_input, self.d_k, bias=True) for _ in range(self.n_heads)])
        self.w_q_list = nn.ModuleList([nn.Linear(self.d_input, self.d_q, bias=True) for _ in range(self.n_heads)])
        self.w_v_list = nn.ModuleList([nn.Linear(self.d_input, self.d_e, bias=True) for _ in range(self.n_heads)])

        #initiallization
        utils.init_network_weights(self.w_k_list)
        utils.init_network_weights(self.w_q_list)
        utils.init_network_weights(self.w_v_list)


        #Temporal Layer
        self.temporal_net = TemporalEncoding(d_input)

        #Normalization
        self.layer_norm = nn.LayerNorm(d_input,elementwise_affine = False)

    def forward(self, x, edge_index, edge_weight,time_nodes,edge_time):
        '''

        :param x:
        :param edge_index:
        :param edge_wight: edge_weight
        :param time_nodes:
        :param edge_time: edge_time_attr
        :return:
        '''

        residual = x
        x = self.layer_norm(x)

        # Edge normalization if using multiplication
        edge_weight = normalize_graph_asymmetric(edge_index,edge_weight,time_nodes.shape[0])
        assert (torch.sum(edge_weight<0)==0) and (torch.sum(edge_weight>1) == 0)

        return self.propagate(edge_index, x=x, edges_weight=edge_weight, edge_time=edge_time, residual=residual)

    def message(self, x_j,x_i,edge_index_i, edges_weight,edge_time):
        '''

           :param x_j: [num_edge, d] sender
           :param x_i: [num_edge,d]  receiver
           :param edge_index_i:  receiver node list [num_edge]
           :param edges_temporal: [num_edge,d]
           :param edge_time: [num_edge,d]
           :return:
        '''


        messages = []
        for i in range(self.n_heads):
            k_linear = self.w_k_list[i]
            q_linear = self.w_q_list[i]
            v_linear = self.w_v_list[i]

            edge_temporal_vector = self.temporal_net(edge_time) #[num_edge,d]
            edges_weight = edges_weight.view(-1, 1)
            x_j_transfer = x_j + edge_temporal_vector

            attention = self.each_head_attention(x_j_transfer,k_linear,q_linear,x_i) #[N_edge,1]
            attention = torch.div(attention,self.d_sqrt)

            # Need to multiply by original edge weight
            attention = attention * edges_weight

            attention_norm = softmax(attention,edge_index_i) #[N_neighbors_,1]
            sender = v_linear(x_j_transfer)

            message  = attention_norm * sender #[N_nodes,d]
            messages.append(message)

        message_all_head  = torch.cat(messages,1) #[N_nodes, k*d] ,assuming K head

        return message_all_head

    def each_head_attention(self,x_j_transfer,w_k,w_q,x_i):
        '''

        :param x_j_transfer: sender [N_edge,d]
        :param w_k:
        :param w_q:
        :param x_i: receiver
        :return:
        '''

        # Receiver #[num_edge,d*heads]
        x_i = w_q(x_i)
        # Sender
        sender = w_k(x_j_transfer)
        # Calculate attention score
        attention = torch.bmm(torch.unsqueeze(sender,1),torch.unsqueeze(x_i,2)) #[N,1]

        return torch.squeeze(attention,1)


    def update(self, aggr_out,residual):
        x_new = residual + F.gelu(aggr_out)

        return self.dropout(x_new)

    def __repr__(self):
        return '{}'.format(self.__class__.__name__)


class Node_GCN(nn.Module):
    """Node ODE function."""

    def __init__(self, in_dims, out_dims, num_atoms,dropout=0.):
        super(Node_GCN, self).__init__()

        self.w_node = nn.Parameter(torch.FloatTensor(in_dims, out_dims), requires_grad=True) #[D,D]
        self.num_atoms = num_atoms
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_dims,elementwise_affine = False)

        glorot(self.w_node)


    def forward(self, inputs, edges,z_0):

        '''
        :param inputs: [K*N,D] (node attributes, H)
        :param edges: [K,N*N], after normalize
        :param z_0: [K*N,D],
        :return:
        '''
        inputs = self.layer_norm(inputs)

        num_feature = inputs.shape[-1]

        edges = edges.view(-1,self.num_atoms,self.num_atoms) #[K,N,N]
        inputs_transform = torch.matmul(inputs,self.w_node) #[K*N,D]
        inputs_transform = inputs_transform.view(-1,self.num_atoms,num_feature) #[K,N,D]

        x_hidden = torch.bmm(edges,inputs_transform) #[K,N,D]
        x_hidden = x_hidden.view(-1,num_feature) #[K*N,D]

        x_new = F.gelu(x_hidden) - inputs + z_0

        return self.dropout(x_new)


class GeneralConv(nn.Module):
    '''
    general layers
    '''
    def __init__(self, conv_name, in_hid, out_hid, n_heads, dropout,args):
        super(GeneralConv, self).__init__()
        self.conv_name = conv_name
        if self.conv_name == 'GTrans':
            self.base_conv = GTrans(n_heads, in_hid, out_hid, dropout)
        elif self.conv_name == "Node":
            self.base_conv = Node_GCN(in_hid,out_hid,args.num_atoms,dropout)


    def forward(self, x, edge_index, edge_weight, x_time,edge_time):

        return self.base_conv(x, edge_index, edge_weight, x_time,edge_time)


class GNN(nn.Module):
    '''
    wrap up multiple layers
    '''
    def __init__(self, in_dim, n_hid,out_dim, n_heads, n_layers,args, dropout = 0.2, conv_name = 'GTrans', is_encoder = False):
        super(GNN, self).__init__()
        self.gcs = nn.ModuleList()
        self.in_dim = in_dim
        self.n_hid = n_hid
        self.out_dim = out_dim
        self.drop = nn.Dropout(dropout)
        self.is_encoder = is_encoder


        if is_encoder:
            # If encoder, adding 1.) sequence_W 2.)transform_W ( to 2*z_dim).
            self.sequence_w = nn.Linear(n_hid,n_hid) # for encoder
            self.hidden_to_z0 = nn.Sequential(
		        nn.Linear(n_hid, n_hid//2),
		        nn.Tanh(),
		        nn.Linear(n_hid//2, out_dim))
            self.adapt_w = nn.Linear(in_dim,n_hid)
            utils.init_network_weights(self.sequence_w)
            utils.init_network_weights(self.hidden_to_z0)
            utils.init_network_weights(self.adapt_w)
        else: # ODE GNN
            assert self.in_dim == self.n_hid

        # first layer is input layer
        for l in range(0,n_layers):
            self.gcs.append(GeneralConv(conv_name, self.n_hid, self.n_hid,  n_heads, dropout,args))

        if conv_name in  ['GTrans'] :
            self.temporal_net = TemporalEncoding(n_hid)  #// Encoder, needs positional encoding for sequence aggregation.

    def forward(self, x, edge_weight=None, edge_index=None, x_time=None, edge_time=None,batch= None, batch_y = None):  #aggregation part

        if not self.is_encoder: #Encoder initial input node feature
            h_t = self.drop(x)
        else:
            h_t = self.drop(F.gelu(self.adapt_w(x)))  #initial input for encoder


        for gc in self.gcs:
            h_t = gc(h_t, edge_index, edge_weight, x_time,edge_time)  #[num_nodes,d]

        ### Output
        if batch!= None:  ## for encoder
            batch_new = self.rewrite_batch(batch,batch_y) #group by balls

            h_t += self.temporal_net(x_time)
            attention_vector = F.gelu(
                self.sequence_w(global_mean_pool(h_t, batch_new)))  # [num_ball,d] ,graph vector with activation Relu
            attention_vector_expanded = self.attention_expand(attention_vector, batch, batch_y)  # [num_nodes,d]
            attention_nodes = torch.sigmoid(
                torch.squeeze(torch.bmm(torch.unsqueeze(attention_vector_expanded, 1), torch.unsqueeze(h_t, 2)))).view(
                -1, 1)  # [num_nodes]
            nodes_attention = attention_nodes * h_t  # [num_nodes,d]
            h_ball = global_mean_pool(nodes_attention, batch_new)  # [num_ball,d] without activation

            h_out = self.hidden_to_z0(h_ball) #[num_ball,2*z_dim] Must ganrantee NO 0 ENTRIES!
            mean,mu = self.split_mean_mu(h_out)
            mu = mu.abs()
            return mean,mu

        else:  # for ODE
            h_out = h_t
            return h_out

    def rewrite_batch(self,batch, batch_y):
        assert (torch.sum(batch_y).item() == list(batch.size())[0])
        batch_new = torch.zeros_like(batch)
        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            batch_new[current_index:current_index + ball_time] = group_num
            group_num += 1
            current_index += ball_time.item()

        return batch_new

    def attention_expand(self,attention_ball, batch,batch_y):
        '''

        :param attention_ball: [num_ball, d]
        :param batch: [num_nodes,d]
        :param batch_new: [num_ball,d]
        :return:
        '''
        node_size = batch.size()[0]
        dim = attention_ball.size()[1]
        new_attention = torch.ones(node_size, dim)
        if attention_ball.device != torch.device("cpu"):
            new_attention = new_attention.cuda()

        group_num = 0
        current_index = 0
        for ball_time in batch_y:
            new_attention[current_index:current_index+ball_time] = attention_ball[group_num]
            group_num +=1
            current_index += ball_time.item()

        return new_attention

    def split_mean_mu(self,h):
        last_dim = h.size()[-1] //2
        res = h[:,:last_dim], h[:,last_dim:]
        return res





