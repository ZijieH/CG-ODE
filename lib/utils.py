import os
import logging
from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp




def transfer_index(date_string):
    init_date = datetime(2020,4,12)
    date_spec = date_string.split("-")
    d1 = datetime(int(date_spec[0]),int(date_spec[1]),int(date_spec[2]))
    interval = d1-init_date
    return interval.days

def print_parameters(model):
	for name, param in model.named_parameters():
		if param.requires_grad:
			print(name)


def makedirs(dirname):
	if not os.path.exists(dirname):
		os.makedirs(dirname)

def save_checkpoint(state, save, epoch):
	if not os.path.exists(save):
		os.makedirs(save)
	filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
	torch.save(state, filename)

	
def get_logger(logpath, filepath, package_files=[],
			   displaying=True, saving=True, debug=False):
	logger = logging.getLogger()
	if debug:
		level = logging.DEBUG
	else:
		level = logging.INFO
	logger.setLevel(level)
	if saving:
		info_file_handler = logging.FileHandler(logpath, mode='w')
		info_file_handler.setLevel(level)
		logger.addHandler(info_file_handler)
	if displaying:
		console_handler = logging.StreamHandler()
		console_handler.setLevel(level)
		logger.addHandler(console_handler)
	logger.info(filepath)

	for f in package_files:
		logger.info(f)
		with open(f, 'r') as package_f:
			logger.info(package_f.read())

	return logger


def inf_generator(iterable):
	"""Allows training with DataLoaders in a single infinite loop:
		for i, (x, y) in enumerate(inf_generator(train_loader)):
	"""
	iterator = iterable.__iter__()
	while True:
		try:
			yield iterator.__next__()
		except StopIteration:
			iterator = iterable.__iter__()


def init_network_weights(net, std = 0.1):
	for m in net.modules():
		if isinstance(m, nn.Linear):
			nn.init.normal_(m.weight, mean=0, std=std)
			nn.init.constant_(m.bias, val=0)



def get_device(tensor):
	device = torch.device("cpu")
	if tensor.is_cuda:
		device = tensor.get_device()
	return device

def sample_standard_gaussian(mu, sigma):
	device = get_device(mu)

	d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))
	r = d.sample(mu.size()).squeeze(-1)
	return r * sigma.float() + mu.float()

def get_dict_template():
	return {"data": None,
			"time_setps": None,
			"mask": None
			}
def get_next_batch_new(dataloader,device):
	data_dict = dataloader.__next__()
	#device_now = data_dict.batch.device
	return data_dict.to(device)

def get_next_batch(dataloader,device):
	# Make the union of all time points and perform normalization across the whole dataset
	data_dict = dataloader.__next__()

	batch_dict = get_dict_template()


	batch_dict["data"] = data_dict["data"].to(device)
	batch_dict["time_steps"] = data_dict["time_steps"].to(device)
	batch_dict["data_gt"] = data_dict["data_gt"].to(device)

	return batch_dict


def get_next_batch_test(dataloader,device):
	# Make the union of all time points and perform normalization across the whole dataset
	data_dict = dataloader.__next__()

	batch_dict = get_dict_template()


	batch_dict["data"] = data_dict["data"].to(device)
	batch_dict["time_steps"] = data_dict["time_steps"].to(device)
	batch_dict["masks"] = data_dict["masks"].to(device)
	batch_dict["data_gt"] = data_dict["data_gt"].to(device)

	return batch_dict

def get_ckpt_model(ckpt_path, model, device):
	if not os.path.exists(ckpt_path):
		raise Exception("Checkpoint " + ckpt_path + " does not exist.")
	# Load checkpoint.
	checkpt = torch.load(ckpt_path)
	ckpt_args = checkpt['args']
	state_dict = checkpt['state_dict']
	model_dict = model.state_dict()

	# 1. filter out unnecessary keys
	state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
	# 2. overwrite entries in the existing state dict
	model_dict.update(state_dict) 
	# 3. load the new state dict
	model.load_state_dict(state_dict)
	model.to(device)


def update_learning_rate(optimizer, decay_rate = 0.999, lowest = 1e-3):
	for param_group in optimizer.param_groups:
		lr = param_group['lr']
		lr = max(lr * decay_rate, lowest)
		param_group['lr'] = lr




def reverse(tensor):
	idx = [i for i in range(tensor.size(0)-1, -1, -1)]
	return tensor[idx]

def create_net(n_inputs, n_outputs, n_layers = 1,
	n_units = 100, nonlinear = nn.Tanh):
	layers = [nn.Linear(n_inputs, n_units)]
	for i in range(n_layers):
		layers.append(nonlinear())
		layers.append(nn.Linear(n_units, n_units))

	layers.append(nonlinear())
	layers.append(nn.Linear(n_units, n_outputs))
	return nn.Sequential(*layers)

def encode_onehot(labels):
	classes = set(labels)
	classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
					enumerate(classes)}
	labels_onehot = np.array(list(map(classes_dict.get, labels)),
							 dtype=np.int32)
	return labels_onehot



def convert_sparse(graph):
	graph_sparse = sp.coo_matrix(graph)
	edge_index = np.vstack((graph_sparse.row, graph_sparse.col))
	edge_attr = graph_sparse.data
	return edge_index, edge_attr


def print_MAPE(MAPE_each):
	str_list = [str(i) for i in MAPE_each]
	str_print = ",".join(str_list)

	return str_print


def inc_to_cum(pred_inc):
	'''

	:param pred_inc: [ K*N , time_length, d]
	:return:
	'''
	num_samples, num_time, num_feature = pred_inc.size()
	pred_cum = torch.ones_like(pred_inc)
	for i in range(num_time):
		pred_cum[:, i, :] = torch.sum(pred_inc[:, :i + 1, :], dim=1)


	return pred_cum.to(pred_inc.device)


def test_data_covid(model, pred_length, condition_length, dataloader,device,args,kl_coef):


	encoder, decoder, graph, num_batch = dataloader.load_test_data(pred_length=pred_length,
	 															   condition_length=condition_length)


	total = {}
	total["loss"] = 0
	total["likelihood"] = 0
	total["MAPE"] = 0
	total["RMSE"] = 0
	total["MSE"] = 0
	total["kl_first_p"] = 0
	total["std_first_p"] = 0
	MAPE_each = []
	RMSE_each = []

	n_test_batches = 0

	model.eval()
	print("Computing loss... ")
	with torch.no_grad():
		for _ in tqdm(range(num_batch)):
			batch_dict_encoder = get_next_batch_new(encoder, device)
			batch_dict_graph = get_next_batch_new(graph, device)
			batch_dict_decoder = get_next_batch_test(decoder, device)

			results = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph, args.num_atoms,
											   edge_lamda=args.edge_lamda, kl_coef=kl_coef, istest=True)

			for key in total.keys():
				if key in results:
					var = results[key]
					if isinstance(var, torch.Tensor):
						var = var.detach().item()
					if key =="MAPE":
						MAPE_each.append(var)
					elif key == "MSE": # assign value for both MSE and RMSE
						RMSE_each.append(np.sqrt(var))
						total["RMSE"] += np.sqrt(var)
					total[key] += var

			n_test_batches += 1

			del batch_dict_encoder, batch_dict_graph, batch_dict_decoder, results

		if n_test_batches > 0:
			for key, value in total.items():
				total[key] = total[key] / n_test_batches




	return total,print_MAPE(MAPE_each),print_MAPE(RMSE_each)


def test_data_social(model, pred_length, condition_length, dataloader, device, args, kl_coef):
	encoder, decoder, graph, num_batch = dataloader.load_test_data(pred_length=pred_length,
																   condition_length=condition_length)

	total = {}
	total["loss"] = 0
	total["likelihood"] = 0
	total["MAPE"] = 0
	total["RMSE"] = 0
	total["MSE"] = 0
	total["kl_first_p"] = 0
	total["std_first_p"] = 0
	MAPE_each = []
	RMSE_each = []

	n_test_batches = 0

	model.eval()
	print("Computing loss... ")
	with torch.no_grad():
		for _ in tqdm(range(num_batch)):
			batch_dict_encoder = get_next_batch_new(encoder, device)
			batch_dict_graph = get_next_batch_new(graph, device)
			batch_dict_decoder = get_next_batch_test(decoder, device)

			results = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph, args.num_atoms,
											   edge_lamda=args.edge_lamda, kl_coef=kl_coef, istest=True)

			for key in total.keys():
				if key in results:
					var = results[key]
					if isinstance(var, torch.Tensor):
						var = var.detach().item()
					if key == "MAPE":
						MAPE_each.append(var)
					elif key == "MSE":  # assign value for both MSE and RMSE
						RMSE_each.append(np.sqrt(var))
						total["RMSE"] += np.sqrt(var)
					total[key] += var

			n_test_batches += 1

			del batch_dict_encoder, batch_dict_graph, batch_dict_decoder, results

		if n_test_batches > 0:
			for key, value in total.items():
				total[key] = total[key] / n_test_batches

	return total