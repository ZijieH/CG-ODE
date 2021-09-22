from lib.likelihood_eval import *
import torch





def gaussian_log_likelihood(mu, data, obsrv_std):
	log_p = ((mu - data) ** 2) / (2 * obsrv_std * obsrv_std)
	neg_log_p = -1*log_p
	return neg_log_p

def generate_time_weight(n_timepoints,n_dims):
	value_min = 1
	value_max = 2
	interval = (value_max - value_min)/(n_timepoints-1)

	value_list = [value_min + i*interval for i in range(n_timepoints)]
	value_list= torch.FloatTensor(value_list).view(-1,1)

	value_matrix= torch.cat([value_list for _ in range(n_dims)],dim = 1)

	return value_matrix


def compute_masked_likelihood(mu, data ,mask ,mu_gt = None, likelihood_func = None,temporal_weights=None):
	# Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
	n_traj, n_timepoints, n_dims = mu.size()
	if mu_gt != None:
		log_prob = likelihood_func(mu, data,mu_gt)  # [n_traj, n_timepoints, n_dims]
	else:
		log_prob = likelihood_func(mu, data)  # MSE
	if mask != None:
		log_prob_masked = torch.sum(log_prob * mask, dim=1)  # [n_traj, n_dims]
		timelength_per_nodes = torch.sum(mask.permute(0, 2, 1), dim=2)  # [n_traj, n_dims]
		assert (not torch.isnan(timelength_per_nodes).any())
		log_prob_masked_normalized = torch.div(log_prob_masked,
											   timelength_per_nodes)  # 【n_traj, feature], average each feature by dividing time length
		# Take mean over the number of dimensions
		res = torch.mean(log_prob_masked_normalized, -1)  # 【n_traj], average among features.
	else:
		res = torch.sum(log_prob , dim=1)  # [n_traj,n_dims]
		time_length = log_prob.shape[1]
		res = torch.div(res,time_length)
		res = torch.mean(res,-1)


	return res



def masked_gaussian_log_density(mu, data, obsrv_std, mask=None,temporal_weights=None):

	n_traj, n_timepoints, n_dims = mu.size()  #n_traj = K*N

	assert(data.size()[-1] == n_dims)

	# Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
	func = lambda mu, data: gaussian_log_likelihood(mu, data, obsrv_std = obsrv_std)
	res = compute_masked_likelihood(mu, data,mask, likelihood_func=func)  #[n_traj = K*N]
	return res


def mse(mu,data):
	return  (mu - data) ** 2

def mape(mu,data,mu_gt = None):
	# [M,T,D]
	if mu_gt == None:
		output = torch.abs(mu-data)/torch.abs(data)
		output = torch.where(output == float("inf"),torch.Tensor([0]).to(mu.device),output)
	else:
		output = torch.abs(mu - data) / torch.abs(mu_gt)
		output = torch.where(output == float("inf"), torch.Tensor([0]).to(mu.device), output)

	return output



def compute_loss(mu, data,mu_gt=None, mask=None,method=None):
	# mu is prediction; data is groud truth

	n_traj, n_timepoints, n_dims = mu.size()
	assert(data.size()[-1] == n_dims)

	if method in ["MSE","RMSE"]:
		res = compute_masked_likelihood(mu, data, mask,likelihood_func = mse)
	elif method == "MAPE":
		res = compute_masked_likelihood(mu, data,mask, mu_gt = mu_gt,likelihood_func=mape)


	return res

	

