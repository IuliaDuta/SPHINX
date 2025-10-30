import torch
import numpy as np
import torch.nn.functional as F

from models import *
from dataset import *
from utils import *

DEVICE = 'cuda:0'


def forward_step_pred_ksample(batch, model, model_predictor, timesteps, nb_sample, 
                                    add_velocity, enc_len, add_self_loop):
    """ One feed-forward step using previous predictions as input (EVAL only),

    Run for the first `timesteps` positions in the input.

    Input:
      # batch.x: n_samples x n_timesteps x n_nodes x dim
      #

    Output: (pred, target) tuple.
    """

    # ->  n_nodes x dim x n_timesteps
    batch = batch.to(DEVICE)
    timesteps_enc = enc_len

    timesteps_dec = timesteps

    assert timesteps_enc+timesteps_dec <= batch.x.shape[1]+1


    batch_enc_x = batch.x[:,:timesteps_enc,:]
    batch_dec_x = batch.x[:,timesteps_enc-1:timesteps_enc-1+timesteps_dec,:] # first stepo is observable anyway
    batch_dec_y = batch.y[:,timesteps_enc-1:timesteps_enc-1+timesteps_dec,:]
    #TODO: check overlap here

    #add velocity as features to the encoder
    if add_velocity:
        x_vel = batch_enc_x[:,1:,:2] - batch_enc_x[:,:-1, :2]
        x_vel = torch.cat([x_vel[:,[0]], x_vel], dim=1)
        batch_enc_x = torch.cat((batch_enc_x, x_vel),dim=-1)

    x_enc = torch.permute(batch_enc_x, (1,0,2)).to(DEVICE) #ts x (bs*nodes) x C
    #repeat the features to have multiple samplings
    x_enc = x_enc.unsqueeze(1)
    x_enc = torch.tile(x_enc, (1, nb_sample, 1, 1)) #ts x nb_sample x (bs*nodes) x C
    x_enc = x_enc.reshape(x_enc.shape[0], -1, x_enc.shape[-1]) #ts x (nb_sample*bs*nodes) x C

    _, _, slots = model_predictor(x_enc, use_old=False) #slots: (nb_sample*bs) x num_slots x num_nodes
    H = torch.block_diag(*torch.unbind(slots, dim=0)) 
    H = torch.permute(H, (1,0))
    
    num_nodes = batch.x.shape[0]*nb_sample

    # add self loops if needed
    if add_self_loop:
        H = add_extra_self_loops_dense(H,  num_nodes) 

    all_feats = batch_dec_x[:,0,:].to(DEVICE).unsqueeze(1) #n_nodes x 1 x 2 # this is because forward consider timesteps as well
    feats = all_feats[:,:,:2] #just position
    category = all_feats[:,:,2:5] #just category
    prev_feats =  batch_enc_x[:,-2,:2].to(DEVICE).unsqueeze(1)
    # Run forward for each timestep, using the previous prediction as input.
    all_y_hat = []
    all_y = []
    
    feats = torch.tile(feats.unsqueeze(0), (nb_sample, 1,1,1))
    feats = feats.reshape(num_nodes, feats.shape[2], feats.shape[3])

    prev_feats = torch.tile(prev_feats.unsqueeze(0), (nb_sample, 1,1,1))
    prev_feats = prev_feats.reshape(num_nodes, prev_feats.shape[2], prev_feats.shape[3])

    category = torch.tile(category.unsqueeze(0), (nb_sample, 1,1,1))
    category = category.reshape(num_nodes, category.shape[2], category.shape[3])

    for i in range(timesteps_dec):
      #add velocity as features to the dencoder, based on the previous prediction
      if add_velocity:
        crt_x_vel = (feats - prev_feats).detach()
        prev_feats = feats.clone()

        feats = torch.cat([feats, category, crt_x_vel], dim=-1)
      else:
        feats = torch.cat([feats, category], dim=-1)

      feats = model(feats, H)
      all_y_hat.append(feats.clone().squeeze(1))
      crt_y = torch.tile(batch_dec_y[:,i,:].unsqueeze(0), (nb_sample,1,1))
      crt_y = crt_y.reshape(crt_y.shape[0]*crt_y.shape[1], crt_y.shape[2])
      all_y.append(crt_y)

    #n_timesteps x n_nodes x output_dim
    all_y_hat = torch.stack(all_y_hat, 0)
    #n_nodes x n_timesteps x output_dim -> (n_nodes * n_timesteps) x output_dim
    all_y_hat = all_y_hat.permute((1,0,2)).reshape(-1,all_y_hat.shape[-1])

    all_y = torch.stack(all_y, 0)
    all_y = all_y.permute((1,0,2)).reshape(-1,all_y.shape[-1])

    # all_y_hat: n_nodes x n_timesteps x output_dim
    # If the target sequence is longer, only select first `timesteps` elements.
    return all_y_hat, all_y, None, H, slots

def eval_k_sample(data_loader, model, model_predictor, num_rollout=10, extra_metrics=False, nb_sample=3, 
                        add_velocity=None, enc_len=5, add_self_loop=None):
    """ Evaluate the model. """

    loss_fct=F.mse_loss

    model.eval()
    model_predictor.eval()
    num_iter = 0
    loss_eval = 0

    if extra_metrics:
        assert num_rollout >= 10

    with torch.no_grad():
        num_examples = 0
        other_metrics = np.array([0.0] * 8)
        for i, batch in enumerate(data_loader):
            # For evaluation, we switch between teacher forcing for validation and
            # using previous predictions at test-time.
            if num_rollout == -1:
                num_rollout = enc_len
            y_hat, y, _, _, _ = forward_step_pred_ksample(batch, model, model_predictor, 
                                                    num_rollout, nb_sample, add_velocity, enc_len, add_self_loop)

            if extra_metrics:
                #n_nodes x n_timesteps x output_dim -> (n_nodes * n_timesteps) x output_dim
                y_hat = y_hat.reshape(nb_sample, -1, num_rollout, y_hat.shape[-1])
                y = y.reshape(nb_sample, -1, num_rollout, y.shape[-1])
                num_examples += y.shape[1] 
                other_metrics += predict_metrics(y_hat, y)

            loss = loss_fct(y_hat, y)
            loss_eval += loss.item()
            num_iter = num_iter + 1

    if extra_metrics:
        other_metrics = [x/ num_examples for x in other_metrics]
    loss_eval /= num_iter


    log_corpus_extra_ksample = {
            f'20sample_valid_ADE_1sec': other_metrics[0],
            f'20sample_valid_ADE_2sec': other_metrics[1],
            f'20sample_valid_ADE_3sec': other_metrics[2],
            f'20sample_valid_ADE_4sec': other_metrics[3],
            f'20sample_valid_FDE_1sec': other_metrics[4],
            f'20sample_valid_FDE_2sec': other_metrics[5],
            f'20sample_valid_FDE_3sec': other_metrics[6],
            f'20sample_valid_FDE_4sec': other_metrics[7],
        }

    return log_corpus_extra_ksample