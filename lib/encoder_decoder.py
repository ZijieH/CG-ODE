import torch.nn as nn
import lib.utils as utils
import numpy as np
import torch
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim,decoder_network = None):
        super(Decoder, self).__init__()
        # decode data from latent space where we are solving an ODE back to the data space
        if decoder_network == None:
            decoder = nn.Sequential(
                nn.Linear(latent_dim, latent_dim//2),
                nn.ReLU(),
                nn.Linear(latent_dim//2,output_dim),
            )
            utils.init_network_weights(decoder)
        else:
            decoder = decoder_network

        self.decoder = decoder

    def forward(self, data):
        return self.decoder(data)


class Edge_NRI(nn.Module):

    def __init__(self, in_channels, w_node2edge, num_atoms,device,dropout=0.):
        super(Edge_NRI, self).__init__()

        self.dropout = nn.Dropout(dropout)
        self.w_node2edge = w_node2edge  #[2*in_channel, in_channel]

        self.w_edge2value = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(),
            nn.Linear(in_channels//2, 1))   # No negative weight!
        self.edge_self_evolve = nn.Sequential(
            nn.Linear(in_channels, in_channels//2),
            nn.ReLU(),
            nn.Linear(in_channels//2, in_channels))

        self.num_atoms = num_atoms
        self.device = device
        self.layer_norm = nn.LayerNorm(in_channels, elementwise_affine=False)

        utils.init_network_weights(self.w_edge2value)
        utils.init_network_weights(self.edge_self_evolve)

        self.rel_send,self.rel_rec = self.rel_rec_compute()


    def rel_rec_compute(self):
        fully_connected = np.ones([self.num_atoms, self.num_atoms])
        rel_send = np.array(utils.encode_onehot(np.where(fully_connected)[0]),
                            dtype=np.float32)  # every node as one-hot[10000], (N*N,N)
        rel_rec = np.array(utils.encode_onehot(np.where(fully_connected)[1]),
                           dtype=np.float32)  # every node as one-hot[10000], (N*N,N)
        rel_send = torch.FloatTensor(rel_send).to(self.device)
        rel_rec = torch.FloatTensor(rel_rec).to(self.device)

        return rel_send,rel_rec


    def forward(self, node_inputs, edges_input, num_atoms):
        # NOTE: Assumes that we have the same graph across all samples.
        '''

        :param node_inputs: [K*N,D]
        :param edges: [K*N*N,D], after normalize
        :return:
        '''
        node_feature_num = node_inputs.shape[1]
        edge_feature_num = edges_input.shape[-1]

        node_inputs = node_inputs.view(-1, num_atoms, node_feature_num)  # [K,N,D]

        senders = torch.matmul(self.rel_send, node_inputs)  # [K,N*N,D]
        receivers = torch.matmul(self.rel_rec, node_inputs)  # [K,N*N,D]
        edges = torch.cat([senders, receivers], dim=-1)  # [K,N*N,2D]

        # Compute z for edges
        edges_from_node = F.gelu(self.w_node2edge(edges))
        edges_input = self.layer_norm(edges_input)
        edges_self = self.edge_self_evolve(edges_input) #[K*N*N,D]
        edges_self = edges_self.view(-1,num_atoms*num_atoms,edge_feature_num) #[K,N*N,D]
        edges_z = self.dropout(edges_from_node + edges_self) #[K,N*N,D]

        # edge2value
        edge_2_value = torch.squeeze(F.relu(self.w_edge2value(edges_z)),dim=-1) #[K,N*N]
        edges_z = edges_z.view(-1,node_feature_num) #[K*N*N,D]

        return edges_z,  edge_2_value



