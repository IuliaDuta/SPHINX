#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2021 
#
# Distributed under terms of the MIT license.

"""
This script contains all models in our paper.
"""

import torch

import torch.nn as nn
import torch.nn.functional as F

from slot_attention import SlotAttention
from slot_attention_seq import SlotAttention as SlotAttentionV2

from samplers.imle import get_sampler
from samplers.aimle import get_sampler_aimle
from samplers.simple import get_simple_sampler

from utils import LearnedParamChecker
from models import MLP_model, TempCNN_model
from utils import gumbel_softmax


class HypergraphPredictor(nn.Module):
    def __init__(self, num_slots, dim, num_iter, temperature, nonlin, init_slot, slot_k, 
                        eps = 1e-8, hidden_dim = 128, trick_fixed_point='none', 
                        selector=None, logits_activation=None, MLP_norm='None', 
                        encoder_type='tempCNN', noise_distribution=None, beta=None, 
                        enc_len=None, in_feat=2, num_nodes=6):
        super(HypergraphPredictor, self).__init__()
        
        self.logits_activation = logits_activation
        self.encoder_type = encoder_type
        self.in_feat = in_feat
        self.num_nodes = num_nodes

        self.input_MLP = MLP_model(dim, hidden_dim, dim, 2,
                            dropout=0.0, normalization=MLP_norm)
        if self.encoder_type == 'tempCNN':
            self.seq_encoder = TempCNN_model(1, hidden_dim, hidden_dim, 3, #2
                                dropout=0.0, normalization=MLP_norm, in_features=in_feat)
        elif self.encoder_type == 'MLP':
            self.seq_encoder = MLP_model(in_feat*enc_len, hidden_dim, hidden_dim, 3, #2
                                dropout=0.0, normalization=MLP_norm)
        
        self.input_MLP2 = MLP_model(hidden_dim, hidden_dim, dim, 3, #1
                            dropout=0.0, normalization=MLP_norm)
        self.slot_process = SlotAttention(num_slots=num_slots, dim=dim, iters = num_iter, 
                                                    eps = eps, hidden_dim = hidden_dim, 
                                                    init_type=init_slot,
                                                    temperature=temperature, nonlin=nonlin, trick_fixed_point=trick_fixed_point,
                                                    )

        
        self.param_checker =  LearnedParamChecker(self.slot_process)
        self.slot_k = slot_k
        self.init_type = init_slot

        self.num_slots = num_slots 
        self.selector = selector

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        if self.selector == 'imle':
            self.imle_sampler_train, self.imle_sampler_valid = get_sampler(self.slot_k, device=device, beta=beta, noise_scale=1.0, num_train_ensemble=1, num_val_ensemble=1)
        elif self.selector == 'aimle':
            self.aimle_sampler_train, self.aimle_sampler_valid = get_sampler_aimle(self.slot_k, device=device, beta=beta, noise_scale=1.0, num_train_ensemble=1, num_val_ensemble=1, noise_distribution=noise_distribution)
        elif self.selector == 'simple':
            self.simple_sampler_train, self.simple_sampler_valid = get_simple_sampler(self.slot_k, device=device, num_train_ensemble=1, num_val_ensemble=1, logits_activation=self.logits_activation)

    def reset_parameters(self):
        self.slot_process.reset_parameters()
        self.input_MLP.reset_parameters()
        self.seq_encoder.reset_parameters()
        self.input_MLP2.reset_parameters()

    def forward(self, x):
        """
        x: time_steps x bs*num_nodes x dim
        """
        x = torch.permute(x, (1,2,0)).unsqueeze(1)

        # encode the input trejectory
        if self.encoder_type == 'MLP':
            x = x.reshape(x.shape[0], -1) # bs*num_nodes x dim*timesteps
            x = self.seq_encoder(x)
        elif self.encoder_type == 'tempCNN':
            x = self.seq_encoder(x)
            x = x.mean(-1).squeeze(-1)

        x = self.input_MLP2(x)

        batch_size = x.shape[0] // self.num_nodes
        num_slots = self.num_slots * batch_size
       
        # apply the soft clustering (non-sequentially)
        _, attn, dots = self.slot_process(x, num_slots = num_slots)
    
        # k-subset sampling for discrete hypergraphs 
        if self.selector == 'imle':
            attn = dots
            bs, s, n = attn.shape
            attn = attn.reshape((-1, attn.shape[-1])).unsqueeze(-1)
            if self.training == True:
                attn = self.imle_sampler_train(attn)
            else:
                attn = self.imle_sampler_valid(attn)
            
            attn = attn[0].squeeze(0).squeeze(-1)
            attn = attn.reshape((bs, s, -1))
        elif self.selector == 'aimle':
            attn = dots
            bs, s, n = attn.shape
            attn = attn.reshape((-1, attn.shape[-1])).unsqueeze(-1)
            if self.training == True:
                attn = self.aimle_sampler_train(attn)
            else:
                attn = self.aimle_sampler_valid(attn)
            
            attn = attn.squeeze(0).squeeze(-1)
            attn = attn.reshape((bs, s, -1))
        elif self.selector == 'simple':
            attn = dots
            bs, s, n = attn.shape
            attn = attn.reshape((-1, attn.shape[-1])).unsqueeze(-1)

            if self.training == True:
                attn = self.simple_sampler_train(attn)
            else:
                attn = self.simple_sampler_valid(attn)

            attn = attn[0].squeeze(0).squeeze(-1)
            attn = attn.reshape((bs, s, -1))
        else:
            attn = None
        
        # bring the hypergraph in the desired form for pytorch
        attn_sparse = attn.to_sparse()

        B = batch_size
        S = self.num_slots
        N = self.num_nodes


        edge_index_1 =  attn_sparse.indices()[0] * S + attn_sparse.indices()[1]
        edge_index_0 =  attn_sparse.indices()[0] * N + attn_sparse.indices()[2]
        edge_weights = attn_sparse.values() 

        edge_index = torch.stack((edge_index_0,edge_index_1), dim = 0)
        return edge_index, edge_weights, attn#attn #, slots


class HypergraphPredictorSeq(nn.Module):
    """
    sequential hypergraph predictor
    the difference between this one and HypergraphPredictor
    is that for this one, in order to predict a slot you received as input what was already predicted
    such that not all the slots attched to the same one
    """
    def __init__(self, num_slots, dim, num_iter, temperature, nonlin, init_slot, slot_k, 
                        eps = 1e-8, hidden_dim = 128, trick_fixed_point='none', 
                        selector=None, logits_activation=None, MLP_norm='None', 
                        encoder_type='tempCNN', noise_distribution=None, beta=None, 
                        enc_len=None, deterministic=True, num_nodes=0, in_feat=2, long_history=False, nb_sample=1):
        super(HypergraphPredictorSeq, self).__init__()
        
        self.num_nodes = num_nodes
        self.long_history = long_history
        self.logits_activation = logits_activation
        self.encoder_type = encoder_type
        self.nb_sample = nb_sample
        self.in_feat = in_feat
        self.nonlin = nonlin

        self.input_MLP = MLP_model(dim, hidden_dim, dim, 2,
                            dropout=0.0, normalization=MLP_norm)
        if self.encoder_type == 'tempCNN':
            self.seq_encoder = TempCNN_model(1, hidden_dim, hidden_dim, 3, #2
                                dropout=0.0, normalization=MLP_norm, in_features=in_feat)
        elif self.encoder_type == 'MLP':
            self.seq_encoder = MLP_model(in_feat*enc_len, hidden_dim, hidden_dim, 3, #2
                                dropout=0.0, normalization=MLP_norm)
        
        if self.long_history == False:
            self.input_MLP2 = MLP_model(hidden_dim, hidden_dim, dim-1, 3, #1
                            dropout=0.0, normalization=MLP_norm)
        else:
            self.input_MLP2 = MLP_model(hidden_dim, hidden_dim, dim-num_slots, 3, #1
                            dropout=0.0, normalization=MLP_norm)
        
        self.slot_process = SlotAttentionV2(num_slots=1, dim=dim, iters = num_iter, 
                                                    eps = eps, hidden_dim = hidden_dim, 
                                                    init_type=init_slot,
                                                    temperature=temperature, nonlin=nonlin, trick_fixed_point=trick_fixed_point,
                                                    deterministic = deterministic,
                                                    num_total_slots=num_slots,
                                                    long_history = long_history
                                                    )

        self.param_checker =  LearnedParamChecker(self.slot_process)
        self.slot_k = slot_k
        self.init_type = init_slot

        self.num_slots = num_slots 
        self.selector = selector

        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        if self.selector == 'imle':
            self.imle_sampler_train, self.imle_sampler_valid = get_sampler(self.slot_k, device=device, beta=beta, noise_scale=1.0, num_train_ensemble=1, num_val_ensemble=1, nb_sample=self.nb_sample)
        elif self.selector == 'aimle':
            self.aimle_sampler_train, self.aimle_sampler_valid = get_sampler_aimle(self.slot_k, device=device, beta=beta, noise_scale=1.0, num_train_ensemble=1, num_val_ensemble=1, noise_distribution=noise_distribution, nb_sample=self.nb_sample)
        elif self.selector == 'simple':
            self.simple_sampler_train, self.simple_sampler_valid = get_simple_sampler(self.slot_k, device=device, num_train_ensemble=1, num_val_ensemble=1, logits_activation=self.logits_activation, nb_sample=self.nb_sample)

    def reset_parameters(self):
        self.slot_process.reset_parameters()
        self.input_MLP.reset_parameters()
        self.seq_encoder.reset_parameters()
        self.input_MLP2.reset_parameters()

    def sample(self, attn, dots):
        if self.selector == 'gumbel_softmax':
            assert self.nonlin == 'sigmoid'
            non_attn = torch.ones_like(attn, device=attn.device)-attn
            full_attn = torch.stack((non_attn,attn),dim=-1)
            attn = gumbel_softmax(full_attn, tau=10, hard=True)[:,:,:,0]
        elif self.selector == 'imle':
            attn = dots
            bs, s, n = attn.shape
            attn = attn.reshape((-1, attn.shape[-1])).unsqueeze(-1)
            if self.training == True:
                attn = self.imle_sampler_train(attn)
            else:
                attn = self.imle_sampler_valid(attn)
            
            attn = attn[0].squeeze(0).squeeze(-1)
            attn = attn.reshape((bs, s, -1))
        elif self.selector == 'aimle':
            attn = dots
            bs, s, n = attn.shape
            attn = attn.reshape((-1, attn.shape[-1])).unsqueeze(-1)
            if self.training == True:
                attn = self.aimle_sampler_train(attn)
            else:
                attn = self.aimle_sampler_valid(attn) #BS*num_sample x num_channels 
            attn = attn.squeeze(0).squeeze(-1)
            attn = attn.reshape((bs, s, -1))
        elif self.selector == 'simple':
            attn = dots
            bs, s, n = attn.shape
            attn = attn.reshape((-1, attn.shape[-1])).unsqueeze(-1)

            if self.training == True:
                attn = self.simple_sampler_train(attn)
            else:
                attn = self.simple_sampler_valid(attn)

            attn = attn[0].squeeze(0).squeeze(-1)
            attn = attn.reshape((bs, s, -1))
        else:
            attn = None
        return attn

    def forward(self, x):
        """
        x: time_steps x bs*num_nodes x dim
        """
        x = torch.permute(x, (1,2,0)).unsqueeze(1)

        # encode the input trajectory
        if self.encoder_type == 'MLP':
            x = x.reshape(x.shape[0], -1) # bs*num_nodes x dim*timesteps
            x = self.seq_encoder(x)
        elif self.encoder_type == 'tempCNN':
            x = self.seq_encoder(x)
            x = x.mean(-1).squeeze(-1)

        x = self.input_MLP2(x)
        batch_size = x.shape[0] // self.num_nodes
        
        # create the history placeholder (b in our paper)
        if self.long_history == False:
            history_attn = torch.zeros((batch_size, 1, self.num_nodes), device=x.device)
        else:
            history_attn = torch.zeros((batch_size, self.num_slots, self.num_nodes), device=x.device)

        # apply the sequential slot-attention + k-subset sampling at each step
        all_attn = []
        for idx in range(self.num_slots):
            _, attn, dots = self.slot_process(x, history_attn, num_slots = batch_size, seed=idx)
            # do k-subset sampling for the predicted hyperedge
            attn = self.sample(attn, dots)
            # update the history with the recent prediction
            if self.long_history:
                #keep the entire history
                history_attn[:,idx,:] = attn.detach().squeeze(1)
            else:
                #only keep track of the last prediction
                history_attn[:,0,:] = attn.detach().squeeze(1)
            all_attn.append(attn.clone())

        # arrange the prediction in the desired format for pytorch
        attn = torch.cat(all_attn, 1)  
        attn_sparse = attn.to_sparse()

        B = batch_size
        S = self.num_slots
        N = self.num_nodes

        edge_index_1 =  attn_sparse.indices()[0] * S + attn_sparse.indices()[1]
        edge_index_0 =  attn_sparse.indices()[0] * N + attn_sparse.indices()[2]
        edge_weights = attn_sparse.values() # / attn_sparse.values().sum()

        edge_index = torch.stack((edge_index_0,edge_index_1), dim = 0)

        return edge_index, edge_weights, attn#attn #, slots


