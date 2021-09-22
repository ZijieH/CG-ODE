import os
import sys
from lib.load_data_covid import ParseData
from tqdm import tqdm
import argparse
import numpy as np
from random import SystemRandom
import torch
import torch.optim as optim
import lib.utils as utils
from torch.distributions.normal import Normal
from lib.create_coupled_ode_model import create_CoupledODE_model
from lib.utils import test_data_covid


parser = argparse.ArgumentParser('Coupled ODE')

parser.add_argument('--save', type=str, default='experiments/', help="Path for save checkpoints")
parser.add_argument('--load', type=str, default=None, help="name of ckpt. If None, run a new experiment.")
parser.add_argument('--dataset', type=str, default='Dec', help="Dec")
parser.add_argument('--datapath', type=str, default='data/', help="default data path")
parser.add_argument('--pred_length', type=int, default=14, help="Number of days to predict ")
parser.add_argument('--condition_length', type=int, default=21, help="Number days to condition on")
parser.add_argument('--features', type=str,
                    default="Confirmed,Deaths,Recovered,Mortality_Rate,Testing_Rate,Population,Mobility",
                    help="selected features")
parser.add_argument('--split_interval', type=int, default=3,
                    help="number of days between two adjacent starting date of two series.")
parser.add_argument('--feature_out', type=str, default='Deaths',
                    help="Confirmed, Deaths, or Confirmed and deaths")

parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-3, help="Starting learning rate.")
parser.add_argument('-b', '--batch-size', type=int, default=8)
parser.add_argument('-r', '--random-seed', type=int, default=1991, help="Random_seed")
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 regulazer')
parser.add_argument('--optimizer', type=str, default="AdamW", help='Adam, AdamW')
parser.add_argument('--clip', type=float, default=10, help='Gradient Norm Clipping')
parser.add_argument('--edge_lamda', type=float, default=0.5, help='edge weight')

parser.add_argument('--z0-encoder', type=str, default='GTrans', help="GTrans")
parser.add_argument('--rec-dims', type=int, default= 64, help="Dimensionality of the recognition model .")
parser.add_argument('--ode-dims', type=int, default= 20, help="Dimensionality of the ODE func for edge and node (must be the same)")
parser.add_argument('--rec-layers', type=int, default=1, help="Number of layers in recognition model ")
parser.add_argument('--gen-layers', type=int, default=1, help="Number of layers  ODE func ")

parser.add_argument('--augment_dim', type=int, default=0, help='augmented dimension')
parser.add_argument('--solver', type=str, default="rk4", help='dopri5,rk4,euler')

parser.add_argument('--alias', type=str, default="run")
args = parser.parse_args()


############ CPU AND GPU related
if torch.cuda.is_available():
	print("Using GPU" + "-"*80)
	device = torch.device("cuda:0")
else:
	print("Using CPU" + "-" * 80)
	device = torch.device("cpu")

###########  feature related:
if args.feature_out == "Confirmed":
    args.output_dim = 1
    args.feature_out_index = [0]
elif args.feature_out == "Deaths":
    args.output_dim = 1
    args.feature_out_index = [1]
else:
    args.output_dim = 2
    args.feature_out_index = [0, 1]


#####################################################################################################

if __name__ == '__main__':
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    #Saving Path
    file_name = os.path.basename(__file__)[:-3]  # run_models
    utils.makedirs(args.save)
    experimentID = int(SystemRandom().random() * 100000)

    #Command Log
    input_command = sys.argv
    ind = [i for i in range(len(input_command)) if input_command[i] == "--load"]
    if len(ind) == 1:
        ind = ind[0]
        input_command = input_command[:ind] + input_command[(ind + 2):]
    input_command = " ".join(input_command)


    #Loading Data
    print("predicting data at: %s" % args.dataset)
    dataloader = ParseData(args =args)
    train_encoder, train_decoder, train_graph, train_batch, num_atoms = dataloader.load_train_data(is_train=True)
    val_encoder, val_decoder, val_graph, val_batch, _ = dataloader.load_train_data(is_train=False)
    args.num_atoms = num_atoms
    input_dim = dataloader.num_features

    # Model Setup
    # Create the model
    obsrv_std = 0.01
    obsrv_std = torch.Tensor([obsrv_std]).to(device)
    z0_prior = Normal(torch.Tensor([0.0]).to(device), torch.Tensor([1.]).to(device))
    model = create_CoupledODE_model(args, input_dim, z0_prior, obsrv_std, device)

    # Load checkpoint for saved model
    if args.load is not None:
        ckpt_path = os.path.join(args.save, args.load)
        utils.get_ckpt_model(ckpt_path, model, device)
        print("loaded saved ckpt!")
        #exit()


    # Training Setup
    log_path = "logs/" + args.alias +"_" + args.dataset +  "_Con_"  + str(args.condition_length) +  "_Pre_" + str(args.pred_length) + "_" + str(experimentID) + ".log"
    if not os.path.exists("logs/"):
        utils.makedirs("logs/")
    logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    logger.info(input_command)
    logger.info(str(args))
    logger.info(args.alias)

    # Optimizer
    if args.optimizer == "AdamW":
        optimizer =optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.l2)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.l2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 1000, eta_min=1e-9)


    wait_until_kl_inc = 10
    best_test_MAPE = np.inf
    best_test_RMSE = np.inf
    best_val_MAPE = np.inf
    best_val_RMSE = np.inf
    n_iters_to_viz = 1


    def train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef):

        optimizer.zero_grad()
        train_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,args.num_atoms,edge_lamda = args.edge_lamda, kl_coef=kl_coef,istest=False)

        loss = train_res["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

        optimizer.step()

        loss_value = loss.data.item()

        del loss
        torch.cuda.empty_cache()
        # train_res, loss
        return loss_value,train_res["MAPE"],train_res['MSE'],train_res["likelihood"],train_res["kl_first_p"],train_res["std_first_p"]

    def train_epoch(epo):
        model.train()
        loss_list = []
        MAPE_list = []
        MSE_list = []
        likelihood_list = []
        kl_first_p_list = []
        std_first_p_list = []

        torch.cuda.empty_cache()

        for itr in tqdm(range(train_batch)):

            #utils.update_learning_rate(optimizer, decay_rate=0.999, lowest=args.lr / 10)
            wait_until_kl_inc = 1000

            if itr < wait_until_kl_inc:
                kl_coef = 1
            else:
                kl_coef = 1*(1 - 0.99 ** (itr - wait_until_kl_inc))

            batch_dict_encoder = utils.get_next_batch_new(train_encoder, device)
            batch_dict_graph = utils.get_next_batch_new(train_graph, device)
            batch_dict_decoder = utils.get_next_batch(train_decoder, device)

            loss, MAPE,MSE,likelihood,kl_first_p,std_first_p = train_single_batch(model,batch_dict_encoder,batch_dict_decoder,batch_dict_graph,kl_coef)

            #saving results
            loss_list.append(loss), MAPE_list.append(MAPE), MSE_list.append(MSE),likelihood_list.append(
               likelihood)
            kl_first_p_list.append(kl_first_p), std_first_p_list.append(std_first_p)

            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
                #train_res, loss
            torch.cuda.empty_cache()

        scheduler.step()

        message_train = 'Epoch {:04d} [Train seq (cond on sampled tp)] | Loss {:.6f} | MAPE {:.6F} | RMSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            epo,
            np.mean(loss_list), np.mean(MAPE_list),np.sqrt(np.mean(MSE_list)), np.mean(likelihood_list),
            np.mean(kl_first_p_list), np.mean(std_first_p_list))

        return message_train,kl_coef

    def val_epoch(epo,kl_coef):
        model.eval()
        MAPE_list = []
        MSE_list = []


        torch.cuda.empty_cache()

        for itr in tqdm(range(val_batch)):
            batch_dict_encoder = utils.get_next_batch_new(val_encoder, device)
            batch_dict_graph = utils.get_next_batch_new(val_graph, device)
            batch_dict_decoder = utils.get_next_batch(val_decoder, device)

            val_res = model.compute_all_losses(batch_dict_encoder, batch_dict_decoder, batch_dict_graph,
                                                 args.num_atoms, edge_lamda=args.edge_lamda, kl_coef=kl_coef,
                                                 istest=False)

            MAPE_list.append(val_res['MAPE']), MSE_list.append(val_res['MSE'])
            del batch_dict_encoder, batch_dict_graph, batch_dict_decoder
            # train_res, loss
            torch.cuda.empty_cache()


        message_val = 'Epoch {:04d} [Val seq (cond on sampled tp)] |  MAPE {:.6F} | RMSE {:.6F} |'.format(
            epo,
            np.mean(MAPE_list), np.sqrt(np.mean(MSE_list)))

        return message_val, np.mean(MAPE_list),np.sqrt(np.mean(MSE_list))

    # Test once: for loaded model
    if args.load is not None:
        test_res, MAPE_each, RMSE_each = test_data_covid(model, args.pred_length, args.condition_length, dataloader,
                                                   device=device, args=args, kl_coef=0)

        message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Loss {:.6f} | MAPE {:.6F} | RMSE {:.6F} | Likelihood {:.6f} | KL fp {:.4f} | FP STD {:.4f}|'.format(
            0,
            test_res["loss"], test_res["MAPE"], test_res["RMSE"], test_res["likelihood"],
            test_res["kl_first_p"], test_res["std_first_p"])

        logger.info("Experiment " + str(experimentID))
        logger.info(message_test)
        logger.info(MAPE_each)
        logger.info(RMSE_each)

    # Training and Testing
    for epo in range(1, args.niters + 1):

        message_train, kl_coef = train_epoch(epo)
        message_val, MAPE_val, RMSE_val = val_epoch(epo,kl_coef)

        if epo % n_iters_to_viz == 0:
            # Logging Train and Val
            logger.info("Experiment " + str(experimentID))
            logger.info(message_train)
            logger.info(message_val)


            # Testing
            model.eval()
            test_res,MAPE_each,RMSE_each = test_data_covid(model, args.pred_length, args.condition_length, dataloader,
                                 device=device, args = args, kl_coef=kl_coef)
            message_test = 'Epoch {:04d} [Test seq (cond on sampled tp)] | MAPE {:.6F} | RMSE {:.6F}|'.format(
                epo,
                test_res["MAPE"], test_res["RMSE"])


            if MAPE_val < best_val_MAPE:
                best_val_MAPE = MAPE_val
                best_val_RMSE = RMSE_val
                logger.info("Best Val!")
                ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) + "_" + args.dataset + "_" + args.alias + "_" + str(
                    args.condition_length) + "_" + str(
                    args.pred_length) + "_epoch_" + str(epo) + "_mape_" + str(
                    test_res["MAPE"]) + '.ckpt')
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)



            logger.info(message_test)
            logger.info(MAPE_each)
            logger.info(RMSE_each)


            if test_res["MAPE"] < best_test_MAPE:
                best_test_MAPE = test_res["MAPE"]
                best_test_RMSE = test_res["RMSE"]
                message_best = 'Epoch {:04d} [Test seq (cond on sampled tp)] | Best Test MAPE {:.6f}|Best Test RMSE {:.6f}|'.format(epo,
                                                                                                        best_test_MAPE,best_test_RMSE)
                logger.info(MAPE_each)
                logger.info(RMSE_each)
                logger.info(message_best)
                ckpt_path = os.path.join(args.save, "experiment_" + str(
                    experimentID) +  "_" + args.dataset + "_" + args.alias+ "_" + str(
                    args.condition_length) + "_" + str(
                    args.pred_length) + "_epoch_" + str(epo) + "_mape_" + str(
                    best_test_MAPE) + '.ckpt')
                torch.save({
                    'args': args,
                    'state_dict': model.state_dict(),
                }, ckpt_path)


            torch.cuda.empty_cache()
            















