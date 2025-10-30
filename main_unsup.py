
"""
main code to train synthehtic dataset (One-Triangle and Two-Triangles)
"""
import numpy as np
import numpy_indexed as npi
import os
import pickle
import wandb
import argparse
import transformers

import torch
from torch_geometric.data import Data
from torch_geometric.data import Data, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models import *
from dataset import *
from utils import *
from hypergraph_predictor import HypergraphPredictor, HypergraphPredictorSeq


DEVICE = 'cuda:0'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--tag', default='hyperparameter_tuning')
parser.add_argument('--num_layers', type=int, default=1)
parser.add_argument('--heads', type=int, default=1)
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--num_features', type=int, default=2)
parser.add_argument('--MLP_hidden', type=int, default=128)
parser.add_argument('--MLP_num_layers', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--MLP_norm',  default='None')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--connectivity', default='learned')
parser.add_argument('--num_nodes', type=int, default=1)

#slot-attention specific
parser.add_argument('--num_iter', default=2, type=int)
parser.add_argument('--temperature', default=1.0, type=float)
parser.add_argument('--slot_nonlin', default='sigmoid', type=str, choices=['sigmoid', 'softmax', 'sparsemax', 'None'])
parser.add_argument('--init_slot', default='rand', type=str, choices=['rand', 'hgraph', 'rand_embedding'])
parser.add_argument('--selection_type', default='simple', type=str, choices=['gumbel_softmax', 'imle', 'aimle', 'simple'])
parser.add_argument('--slot_k', default=3, type=int)
parser.add_argument('--num_slots', default=2, type=int)
parser.add_argument('--check_learned_params', type=str2bool, default=False)
parser.add_argument('--logits_activation', type=str, default='logsoftmax', choices=['logsoftmax', 'logsigmoid', 'None'])


parser.add_argument('--model_type', type=str, default='HGNN', choices=['HGNN', 'AllDeepSets', 'HCHA'])
parser.add_argument('--encoder_type', type=str, default='MLP', choices=['tempCNN', 'MLP'])

parser.add_argument('--noise_distribution', type=str, default='gumbel', choices=['gumbel', 'sog']) #this is just for aimle
parser.add_argument('--imle_beta', default=0.1, type=float)

parser.add_argument('--trick_fixed_point', type=str, default='none') #'none', 'neumann' or 'bi-level'
parser.add_argument('--add_self_loop', type=str2bool, default=True)

parser.add_argument('--enc_len', default=22, type=int)
parser.add_argument('--ld', default=10, type=float) #learningrate decay factor for predictor
parser.add_argument('--degree_detached', type=str2bool, default=True)

parser.add_argument('--dataset', type=str, default='one_triangle', choices=['one_triangle', 'two_triangles'])
parser.add_argument('--deterministic', type=str2bool, default=True)

parser.add_argument('--sequential', type=str2bool, default=True)

config_defaults = parser.parse_args()

"""# Train the models for trajectory prediction"""

def visualize(model, model_pred, val_loader, epoch, table):
    """ Code used to visualise the predicted higher-order structure
    this can be used both as an interpretation tool
    and to debug the model.
    """
    first_batch = next(iter(val_loader))
    num_display = min(3, len(first_batch))

    model.eval()
    model_pred.eval()
    for graph_idx in range(num_display):
        crt_data = first_batch[graph_idx]
        gt_edge_index = crt_data.edge_index.detach()

        timesteps =  config_defaults.enc_len
        y_hat_tf, y, pred_edge_index_tf, _, _ = forward_step_teacher_forcing(crt_data, model, model_pred, timesteps)
        y_hat_roll, y, pred_edge_index_roll, _, _ = forward_step_pred(crt_data, model, model_pred, timesteps)

        y = y.cpu().numpy().reshape((-1, timesteps, 2))
        y = np.transpose(y,(1,0,2))

        y_hat_tf = y_hat_tf.detach().cpu().numpy().reshape((-1, timesteps, 2))
        y_hat_tf = np.transpose(y_hat_tf,(1,0,2))

        y_hat_roll = y_hat_roll.detach().cpu().numpy().reshape((-1, timesteps, 2))
        y_hat_roll = np.transpose(y_hat_roll,(1,0,2))

        # printing the groud truth
        gt_edge_index = npi.group_by(gt_edge_index[1, :]).split(gt_edge_index[0, :])
        f_gt = draw(y[:,:,:], gt_edge_index, './gt.jpg')
        gt_image = wandb.Image(f_gt)

        # printing the teacher-forcing trajectory
        pred_edge_index_tf = pred_edge_index_tf.detach().cpu().numpy()
        pred_edge_index_tf = npi.group_by(pred_edge_index_tf[1, :]).split(pred_edge_index_tf[0, :])
        f_tf = draw(y_hat_tf[:,:,:], pred_edge_index_tf, './pred_tf.jpg')
        teacher_forcing_image = wandb.Image(f_tf)

        # printing the multi-step rolled trajectory
        pred_edge_index_roll = pred_edge_index_roll.detach().cpu().numpy()
        pred_edge_index_roll = npi.group_by(pred_edge_index_roll[1, :]).split(pred_edge_index_roll[0, :])
        f_roll = draw(y_hat_roll[:,:,:], pred_edge_index_roll, './pred_roll.jpg')
        roll_pred_image = wandb.Image(f_roll) 

        id = f'{epoch}_{graph_idx}'
        table.add_row(id, teacher_forcing_image, roll_pred_image, gt_image, epoch)
    return table

def forward_step_teacher_forcing(data, model, model_predictor,timesteps):
    """ One feed-forward step with teacher-forcing (used during TRAINING).

    During training, we use a teacher-forcing approach: at each time step t, we
    feed as input the real position from the previous timestep t-1.

    Output: (pred, target, edge_index, H, slots) tuple.
    """
    batch = data.clone()
    batch.x = batch.x[:,:timesteps, :] # select to learn from the first timesteps steps
    batch.y = batch.y[:,:timesteps, :] # select to learn from the first timesteps steps

    x_enc = torch.permute(batch.x, (1,0,2)).to(DEVICE)
    num_nodes = batch.x.shape[0]

    # predict structure as slots
    edge_index, _, slots = model_predictor(x_enc) #H: bs x num_slots x num_nodes

    # convert slots into incidence matrix 
    H = torch.block_diag(*torch.unbind(slots, dim=0))
    H = torch.permute(H, (1,0))

    # add self loops if needed
    if config_defaults.add_self_loop:
        H = add_extra_self_loops_dense(H,  num_nodes) 
    
    batch = batch.to(DEVICE)
    # batch.x: num_nodes_batch x num_timesteps x 2; 
    # H: num_nodes_batch x num_edges_batch
    
    # use H in the hypergraph processor 
    y_hat  = model(batch.x, batch.omega, H)

    # (n_timesteps x n_nodes) x output_dim -> (n_nodes x n_timesteps) x output_dim
    y = batch.y.reshape(-1,2)
    y_hat = y_hat.reshape(-1,2)

    return y_hat, y, edge_index, H, slots

def forward_step_pred(batch, model, model_predictor, timesteps):
    """ One feed-forward step using previous predictions as input (EVAL only),

    Run for the first `timesteps` positions in the input.
    Input:
      # batch.x: n_samples x n_timesteps x n_nodes x dim


    Output: (pred, target, edge_index, H, slots) tuple.
    """

    # ->  n_nodes x dim x n_timesteps
    batch = batch.to(DEVICE)
    timesteps_enc = config_defaults.enc_len
    timesteps_dec = timesteps
    assert timesteps_enc+timesteps_dec <= batch.x.shape[1] # make sure you don't leak information


    batch_enc_x = batch.x[:,:timesteps_enc,:] # first timesteps_enc steps are observable
    batch_dec_x = batch.x[:,timesteps_enc:timesteps_enc+timesteps_dec,:] # following steps are what we want to predict
    batch_dec_y = batch.y[:,timesteps_enc:timesteps_enc+timesteps_dec,:]

    x_enc = torch.permute(batch_enc_x, (1,0,2)).to(DEVICE)
    num_nodes = batch.x.shape[0]

    # predict structure as slots
    edge_index, _, slots = model_predictor(x_enc)
    # convert slots into incidence matrix 
    H = torch.block_diag(*torch.unbind(slots, dim=0))
    H = torch.permute(H, (1,0))

    # add self loops if needed
    if config_defaults.add_self_loop:
        H = add_extra_self_loops_dense(H,  num_nodes) 

    feats = batch_dec_x[:,0,:].to(DEVICE).unsqueeze(1) #n_nodes x 1 x 2 # this is because forward consider timesteps as well
    # Run forward for each timestep, using the previous prediction as input.
    all_y_hat = []
    all_y = []
    for i in range(timesteps_dec):
      feats = model(feats, batch.omega, H)
      all_y_hat.append(feats.clone().squeeze(1))
      all_y.append(batch_dec_y[:,i,:])

    #n_timesteps x n_nodes x output_dim
    all_y_hat = torch.stack(all_y_hat, 0)
    #n_nodes x n_timesteps x output_dim -> (n_nodes * n_timesteps) x output_dim
    all_y_hat = all_y_hat.permute((1,0,2)).reshape(-1,all_y_hat.shape[-1])
    all_y = torch.stack(all_y, 0)
    all_y = all_y.permute((1,0,2)).reshape(-1,all_y.shape[-1])

    # all_y_hat: n_nodes x n_timesteps x output_dim
    return all_y_hat, all_y, edge_index, H, slots




def train_epoch(data_loader, model, model_predictor, optimiser, optimiser_pred, epoch, loss_fct, scheduler, scheduler_pred, loss_tmp):
    """ Train the model for one epoch. """
    model.train()
    model_predictor.train()

    param_checker =  LearnedParamChecker(model)
    param_checker_pred =  LearnedParamChecker(model_predictor)

    for _, batch in enumerate(data_loader):

        optimiser.zero_grad()
        optimiser_pred.zero_grad()

        # For training, we always use teacher forcing.
        y_hat, y, _, _, slots = forward_step_teacher_forcing(batch, model, model_predictor, config_defaults.enc_len) #
        loss = loss_fct(y_hat, y)

        loss.backward()
        optimiser.step()
        optimiser_pred.step()

        #compute accuracy of the structure prediction (just when we have 1 triangle)
        IoU_score = get_iou_k_slots(batch, slots)
        
    scheduler.step()
    scheduler_pred.step()

    if epoch %10 == 0:
        param_checker.compare_current_initial_params()
        param_checker_pred.compare_current_initial_params()

    return loss.item(), IoU_score

def eval_epoch(data_loader, model, model_predictor, loss_fct, loss_tmp, forward_fct, num_rollout=-1):
    """ Evaluate the model. """
    model.eval()
    model_predictor.eval()
    num_iter = 0
    loss_eval = 0
    IoU_eval = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # For evaluation, we switch between teacher forcing for validation and
            # using previous predictions at test-time.
            if num_rollout == -1:
                num_rollout = config_defaults.enc_len
            y_hat, y, _, _, slots = forward_fct(batch, model, model_predictor, num_rollout)

            loss = loss_fct(y_hat, y)
            loss_eval += loss.item()
            num_iter = num_iter + 1
            
            IoU_score = get_iou_k_slots(batch, slots)
            IoU_eval += IoU_score.item()

    loss_eval /= num_iter
    IoU_eval /= num_iter
    return loss_eval, IoU_eval



def train_eval_loop(model, model_predictor, train_loader, val_loader, test_loader,
               loss_fct, loss_tmp, num_epochs=100, lr=0.0005):
    """ Train/evaluate the model for `num_epochs` epochs. """
    # Instantiate our optimiser.
    optimiser = optim.Adam(list(model.parameters()), lr=lr)
    optimiser_pred = optim.Adam(list(model_predictor.parameters()), lr=lr/config_defaults.ld)

    scheduler = transformers.get_constant_schedule_with_warmup(optimiser, 
                                                         num_warmup_steps=0)
    scheduler_pred = transformers.get_constant_schedule_with_warmup(optimiser_pred, 
                                                         num_warmup_steps=0)

    # Initial evaluation (before training).
    val_loss, val_IoU_score = eval_epoch(
        val_loader, model, model_predictor, loss_fct, loss_tmp, forward_step_teacher_forcing)
    train_loss, train_IoU_score = eval_epoch(
        train_loader, model, model_predictor, loss_fct, loss_tmp, forward_step_teacher_forcing)

    print(f"[Epoch 0]",
          f"train loss: {train_loss:.5f} val loss: {val_loss:.5f}")


    train_loss_1step = []
    valid_loss_1step = []

    table = wandb.Table(columns=["ID", "teacher_forcing", "roll_pred", "gt", "epoch"])
    best_valid_loss_1step = 1000.0
    test_loss = 1000.0

    for epoch in range(num_epochs):
      train_loss, train_IoU_score  = train_epoch(
          train_loader, model, model_predictor, optimiser, optimiser_pred,  epoch, loss_fct, scheduler, scheduler_pred, loss_tmp)
   
      val_loss, val_IoU_score = eval_epoch(
          val_loader, model, model_predictor, loss_fct, loss_tmp, forward_step_teacher_forcing)

      torch.set_printoptions(precision=10)
      # Store the loss and the computed metric for the final plot.

      train_loss_1step.append(train_loss)
      valid_loss_1step.append(val_loss)

      if val_loss <= best_valid_loss_1step:
        best_valid_loss_1step = val_loss
        test_loss, _ = eval_epoch(
                        test_loader, model, model_predictor, loss_fct, loss_tmp, forward_step_teacher_forcing)
        test_loss_25steps, _ = eval_epoch(
            test_loader, model, model_predictor, loss_fct, loss_tmp, forward_step_pred, num_rollout=25)
        test_loss_5steps, _ = eval_epoch(
            test_loader, model, model_predictor, loss_fct, loss_tmp, forward_step_pred, num_rollout=5)
        wandb.log({
                f'test_loss_1step': test_loss,
                f'test_loss_25steps': test_loss_25steps,
                f'test_loss_5steps': test_loss_5steps
        }, step=epoch)

      log_corpus = {
                f'train_loss_1step': train_loss_1step[epoch],
                f'valid_loss_1step': valid_loss_1step[epoch],
                f'best_valid_loss_1step': best_valid_loss_1step,
                f'train_intersection': train_IoU_score,
                f'valid_Intersection': val_IoU_score,
                f'lr': optimiser.param_groups[0]["lr"]
            }
      wandb.log(log_corpus, step=epoch)

      if epoch % 10 == 0:
        train_loss_25steps, _ = eval_epoch(
            train_loader, model,  model_predictor, loss_fct, loss_tmp, forward_step_pred, num_rollout=25)
        valid_loss_25steps, _ = eval_epoch(
            val_loader, model, model_predictor, loss_fct, loss_tmp, forward_step_pred, num_rollout=25)
        train_loss_5steps, _ = eval_epoch(
            train_loader, model,  model_predictor, loss_fct, loss_tmp, forward_step_pred, num_rollout=5)
        valid_loss_5steps, _ = eval_epoch(
            val_loader, model, model_predictor, loss_fct, loss_tmp, forward_step_pred, num_rollout=5)

        log_corpus = {
                f'train_loss_25steps': train_loss_25steps,
                f'valid_loss_25steps': valid_loss_25steps,
                f'train_loss_5steps': train_loss_5steps,
                f'valid_loss_5steps': valid_loss_5steps
            }
        wandb.log(log_corpus, step=epoch)

        crt_lr = optimiser.param_groups[0]["lr"]
        print(f"[Epoch {epoch+1}]",
              f"lr {crt_lr}\n",
              f"train loss 1step: {train_loss:.5f} val loss 1step: {val_loss:.5f}\n",
              f"train loss 5step: {train_loss_5steps:.5f} val loss 5step: {valid_loss_5steps:.5f}\n",
              f"train loss 25step: {train_loss_25steps:.5f} val loss 25step: {valid_loss_25steps:.5f}\n",
              f"train intersection: {train_IoU_score:.5f} val intersection: {val_IoU_score:.5f}\n")

      if epoch % 50 == 0:
          table = visualize(model, model_predictor, val_loader, epoch, table)
    
    wandb.log({"Table": table})
    return 0

 
def main(num_examples):
    
    os.environ["WANDB_AGENT_MAX_INITIAL_FAILURES"]= "200"
    wandb.init(sync_tensorboard=False, project='slot_synth', reinit = False, config = config_defaults, entity='hyper_graphs', tags=[config_defaults.tag])
    print('Monitoring using wandb')

    # Read the data
    if config_defaults.dataset == "two_triangles":
        dataset_path = './datasets/dataset_two_triangles.pkl'
    elif config_defaults.dataset == "one_triangle":
        dataset_path = './datasets/dataset_one_triangle.pkl'

    with open(dataset_path, 'rb') as f:
	    dataset_dict = pickle.load(f)
    config_defaults.num_nodes = 6
        

    print("Configuration:")
    print(config_defaults)

    loc_train = dataset_dict['loc_train']
    edges_train = dataset_dict['edge_train']
    omega_train = dataset_dict['omega_train']

    loc_valid = dataset_dict['loc_valid']
    edges_valid = dataset_dict['edge_valid']
    omega_valid = dataset_dict['omega_valid']
    
    loc_test = dataset_dict['loc_test']
    edges_test = dataset_dict['edge_test']
    omega_test = dataset_dict['omega_test']

    connectivity = config_defaults.connectivity
    print(f"Running with {connectivity} connectivity.")
    
    edges_index_train = process_edges(edges_train)
    edges_index_valid = process_edges(edges_valid)
    edges_index_test = process_edges(edges_test)


    # Pick a subset
    loc_train_inputs, loc_train_targets = (
        loc_train[:num_examples, :-1, :, :], loc_train[:num_examples, 1:, :, :])
    loc_valid_inputs, loc_valid_targets = (
        loc_valid[:num_examples, :-1, :, :], loc_valid[:num_examples, 1:, :, :])
    loc_test_inputs, loc_test_targets = (
        loc_test[:num_examples, :-1, :, :], loc_test[:num_examples, 1:, :, :])
    
    edges_index_train_rand = edges_index_train[:num_examples]
    edges_index_valid_rand = edges_index_valid[:num_examples]
    edges_index_test_rand = edges_index_test[:num_examples]


    small_train_dataset = GroupInteractionDataset(loc_train_inputs, loc_train_targets, edges_index_train_rand, omega_train[:num_examples])
    train_loader = DataLoader(small_train_dataset, batch_size=config_defaults.batch_size, shuffle=True)

    small_valid_dataset = GroupInteractionDataset(loc_valid_inputs, loc_valid_targets, edges_index_valid_rand, omega_valid[:num_examples])
    valid_loader = DataLoader(small_valid_dataset, batch_size=config_defaults.batch_size, shuffle=False)

    small_test_dataset = GroupInteractionDataset(loc_test_inputs, loc_test_targets, edges_index_test_rand, omega_test[:num_examples])
    test_loader = DataLoader(small_test_dataset, batch_size=config_defaults.batch_size, shuffle=False)

    config_defaults.timesteps = small_train_dataset.n_timesteps

    # Create the model
    if config_defaults.model_type == 'HGNN':
        model =  HCHA(config_defaults)
    elif config_defaults.model_type == 'HCHA':
        model =  HCHA(config_defaults, use_attention=True)
    elif config_defaults.model_type == 'AllDeepSets':    
        model =  DenseAllDeepSets(config_defaults)

    if config_defaults.sequential:
        model_predictor = HypergraphPredictorSeq(
                            config_defaults.num_slots, config_defaults.MLP_hidden, config_defaults.num_iter, 
                            eps = 1e-8, hidden_dim =config_defaults.MLP_hidden, 
                            temperature=config_defaults.temperature, nonlin=config_defaults.slot_nonlin, init_slot=config_defaults.init_slot, 
                            slot_k=config_defaults.slot_k, trick_fixed_point=config_defaults.trick_fixed_point, selector=config_defaults.selection_type,
                            logits_activation = config_defaults.logits_activation, encoder_type=config_defaults.encoder_type, 
                            noise_distribution=config_defaults.noise_distribution,beta=config_defaults.imle_beta,
                            enc_len=config_defaults.enc_len, deterministic=config_defaults.deterministic, num_nodes=small_train_dataset.n_nodes)
    else:
        model_predictor = HypergraphPredictor(
                        config_defaults.num_slots, config_defaults.MLP_hidden, config_defaults.num_iter, 
                        eps = 1e-8, hidden_dim =config_defaults.MLP_hidden, 
                        temperature=config_defaults.temperature,  nonlin=config_defaults.slot_nonlin, init_slot=config_defaults.init_slot, 
                        slot_k=config_defaults.slot_k, trick_fixed_point=config_defaults.trick_fixed_point, selector=config_defaults.selection_type,
                        logits_activation = config_defaults.logits_activation, 
                        encoder_type=config_defaults.encoder_type, noise_distribution=config_defaults.noise_distribution,
                        beta=config_defaults.imle_beta, enc_len=config_defaults.enc_len)


    model = model.to(DEVICE)
    model_predictor = model_predictor.to(DEVICE)

    wandb.watch(model)
    # Train the model.
    _ = train_eval_loop(model, model_predictor, train_loader, valid_loader,
                                        test_loader, loss_fct=F.mse_loss, loss_tmp= F.binary_cross_entropy, num_epochs=config_defaults.num_epochs,lr=config_defaults.lr)

    return model, model_predictor, small_train_dataset, small_valid_dataset



num_examples = 1000
main(num_examples)


