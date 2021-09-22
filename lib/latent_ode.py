from lib.base_models import VAE_Baseline
import torch
import numpy as np
import lib.utils as utils
import torch.nn.functional as F

class CoupledODE(VAE_Baseline):
	def __init__(self, w_node_to_edge_initial,ode_hidden_dim, encoder_z0, decoder_node,decoder_edge, diffeq_solver,
				 z0_prior, device, obsrv_std=None):

		super(CoupledODE, self).__init__(
			z0_prior=z0_prior,
			device=device, obsrv_std=obsrv_std
		)



		self.encoder_z0 = encoder_z0
		self.diffeq_solver = diffeq_solver
		self.decoder_node = decoder_node
		self.decoder_edge = decoder_edge
		self.ode_hidden_dim =ode_hidden_dim


		# Shared with edge ODE
		self.w_node_to_edge_initial = w_node_to_edge_initial #h_ij = W([h_i||h_j])



	def get_reconstruction(self, batch_en,batch_de,num_atoms):

        #Encoder:
		first_point_mu, first_point_std = self.encoder_z0(batch_en.x, batch_en.edge_weight,
														  batch_en.edge_index, batch_en.pos, batch_en.edge_time,
														  batch_en.batch, batch_en.y)  # [K*N,D]

		first_point_enc = utils.sample_standard_gaussian(first_point_mu, first_point_std) #[K*N,D]



		first_point_std = first_point_std.abs()

		time_steps_to_predict = batch_de["time_steps"]


		assert (torch.sum(first_point_std < 0) == 0.)
		assert (not torch.isnan(time_steps_to_predict).any())
		assert (not torch.isnan(first_point_enc).any())

		assert (not torch.isnan(first_point_std).any())
		assert (not torch.isnan(first_point_mu).any())



		# ODE:Shape of sol_y #[ K*N + K*N*N, time_length, d], concat of node and edge.
		# K_N is the index for node.
		sol_y,K_N = self.diffeq_solver(first_point_enc,time_steps_to_predict,self.w_node_to_edge_initial)

		assert(not torch.isnan(sol_y).any())

        # Decoder:
		pred_node = self.decoder_node(sol_y[:K_N,:,:])
		pred_edge = self.decoder_edge(sol_y[K_N:, :, :])


		all_extra_info = {
			"first_point": (first_point_mu, first_point_std, first_point_enc),
			"latent_traj": sol_y.detach()
		}

		return pred_node,pred_edge, all_extra_info, None


	def compute_edge_initials(self,first_point_enc,num_atoms):
		'''

		:param first_point_enc: [K*N,D]
		:return: [K*N*N,D']
		'''
		node_feature_num = first_point_enc.shape[1]
		fully_connected = np.ones([num_atoms, num_atoms])
		rel_send = np.array(utils.encode_onehot(np.where(fully_connected)[0]),
							dtype=np.float32)  # every node as one-hot[10000], (N*N,N)
		rel_rec = np.array(utils.encode_onehot(np.where(fully_connected)[1]),
						   dtype=np.float32)  # every node as one-hot[10000], (N*N,N)

		rel_send = torch.FloatTensor(rel_send).to(first_point_enc.device)
		rel_rec = torch.FloatTensor(rel_rec).to(first_point_enc.device)

		first_point_enc = first_point_enc.view(-1,num_atoms,node_feature_num) #[K,N,D]

		senders = torch.matmul(rel_send,first_point_enc) #[K,N*N,D]
		receivers = torch.matmul(rel_rec,first_point_enc) #[K,N*N,D]

		edge_initials = torch.cat([senders,receivers],dim=-1)  #[K,N*N,2D]
		edge_initials = F.relu(self.w_node_to_edge_initial(edge_initials)) #[K,N*N,D_edge]
		edge_initials = edge_initials.view(-1,edge_initials.shape[2]) #[K*N*N,D_edge]

		return edge_initials










