import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from layers import *


class DenseAllDeepSets(nn.Module):
    def __init__(self, config_defaults):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classesFs
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
        """
        super().__init__()

        nfeat = config_defaults.num_features
        nhid = config_defaults.MLP_hidden
        nclass = config_defaults.num_classes

        self.dropout_rate = config_defaults.dropout
        self.num_layers = config_defaults.num_layers
        self.MLP_num_layers = config_defaults.MLP_num_layers
        self.MLP_norm = config_defaults.MLP_norm
        self.add_self_loop = config_defaults.add_self_loop
        self.degree_detached = config_defaults.degree_detached

        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu

        self.convs = torch.nn.ModuleList()
        self.convs.append(torch.nn.Linear(nfeat, nhid))

        for _ in range(self.num_layers):
            self.convs.append(DenseAllDeepSetsConv(nhid, nhid, nhid, nfeat, self.dropout_rate, self.MLP_num_layers, self.MLP_norm, self.degree_detached))
            
        # self.convs.append(torch.nn.Linear(nhid, nclass))
        self.convs.append(MLP_model(nhid, nhid, nclass, 2,
                            dropout=0.0, normalization=self.MLP_norm))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(self.dropout_rate) # 0.2 is chosen for GCNII

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, feats, omega, H):
        #feats is num_nodes x ts x feat
        x = feats
        x0 = x
        x = torch.permute(x, (1,0,2)) # x is ts x num_nodes x feat
        x0 = torch.permute(x0, (1,0,2))

        x = self.convs[0](x)
        for i, con in enumerate(self.convs[1:-1]):
            x = con(x, omega.unsqueeze(-1).float(), H, x0)
            x = self.dropout(x)

        x = self.convs[-1](x)

        x = x0 + x
        x = torch.permute(x, (1,0,2)) # x is num_nodes x ts x feat
        return x



#for NBA Dataset we don't have omega
class DenseAllDeepSetsNBA(nn.Module):
    def __init__(self, config_defaults):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
        """
        super().__init__()

        nfeat = config_defaults.num_features #because of the velocity
        nhid = config_defaults.MLP_hidden
        nclass = config_defaults.num_classes

        self.dropout_rate = config_defaults.dropout
        self.num_layers = config_defaults.num_layers
        self.MLP_num_layers = config_defaults.MLP_num_layers
        self.MLP_norm = config_defaults.MLP_norm
        self.add_self_loop = config_defaults.add_self_loop
        self.degree_detached = config_defaults.degree_detached

        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu

        self.convs = torch.nn.ModuleList()
        #nfeat+2 because of the velocity
        if config_defaults.add_velocity:
            self.convs.append(torch.nn.Linear(nfeat+2, nhid))
        else:
            self.convs.append(torch.nn.Linear(nfeat, nhid))

        for _ in range(self.num_layers):
            self.convs.append(DenseAllDeepSetsConvNBA(nhid, nhid, nhid, nfeat, self.dropout_rate, self.MLP_num_layers, self.MLP_norm, self.degree_detached))
            
        self.convs.append(MLP_model(nhid, nhid, nclass, 2,
                            dropout=0.0, normalization=self.MLP_norm))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(self.dropout_rate) # 0.2 is chosen for GCNII

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, feats, H):
        x = feats
        x0 = x[:,:,:2] #remove the velocity from this

        x = torch.permute(x, (1,0,2)) # x is ts x num_nodes x feat
        x0 = torch.permute(x0, (1,0,2))

        x = self.convs[0](x)
        for i,con in enumerate(self.convs[1:-1]):
            x = con(x, H, x0)
            x = self.dropout(x)

        x = self.convs[-1](x)

        x = x0 + x
        x = torch.permute(x, (1,0,2)) # x is num_nodes x ts x feat
        return x



class DenseAllDeepSetsNBAOrig(nn.Module):
    def __init__(self, config_defaults):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
        """
        super().__init__()

        nfeat = config_defaults.num_features #because of the velocity
        nhid = config_defaults.MLP_hidden
        nclass = config_defaults.num_classes

        self.dropout_rate = config_defaults.dropout
        self.num_layers = config_defaults.num_layers
        self.MLP_num_layers = config_defaults.MLP_num_layers
        self.MLP_norm = config_defaults.MLP_norm
        self.add_self_loop = config_defaults.add_self_loop
        self.degree_detached = config_defaults.degree_detached

        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu

        self.convs = torch.nn.ModuleList()
        #nfeat+2 because of the velocity
        if config_defaults.add_velocity:
        # if False:
            self.convs.append(torch.nn.Linear(nfeat+2, nhid))
        else:
            self.convs.append(torch.nn.Linear(nfeat, nhid))

        for _ in range(self.num_layers):
            self.convs.append(DenseAllDeepSetsConvNBAOrig(nhid, nhid, nhid, nfeat, self.dropout_rate, self.MLP_num_layers, self.MLP_norm, self.degree_detached))
            
        self.convs.append(MLP_model(nhid, nhid, nclass, 2,
                            dropout=0.0, normalization=self.MLP_norm))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(self.dropout_rate) # 0.2 is chosen for GCNII

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, feats, H):
        x = feats

        x0 = x[:,:,:2] #remove the velocity from this

        x = torch.permute(x, (1,0,2)) # x is ts x num_nodes x feat
        x0 = torch.permute(x0, (1,0,2))

        x = self.convs[0](x)
        for i,con in enumerate(self.convs[1:-1]):
            x = con(x, H, x0)
            x = self.dropout(x)

        x = self.convs[-1](x)

        x = x0 + x
        x = torch.permute(x, (1,0,2)) # x is num_nodes x ts x feat
        return x


class HCHA(nn.Module):
    def __init__(self, config_defaults, use_attention=False):
        """UniGNNII

        Args:
            args   (NamedTuple): global args
            nfeat  (int): dimension of features
            nhid   (int): dimension of hidden features, note that actually it\'s #nhid x #nhead
            nclass (int): number of classes
            nlayer (int): number of hidden layers
            nhead  (int): number of conv heads
        """
        super().__init__()

        nfeat = config_defaults.num_features
        nhid = config_defaults.MLP_hidden
        nclass = config_defaults.num_classes

        self.dropout_rate = config_defaults.dropout
        self.num_layers = config_defaults.num_layers
        self.MLP_num_layers = config_defaults.MLP_num_layers
        self.MLP_norm = config_defaults.MLP_norm
        self.add_self_loop = config_defaults.add_self_loop
        self.degree_detached = config_defaults.degree_detached

        act = {'relu': nn.ReLU(), 'prelu':nn.PReLU() }
        self.act = act['relu'] # Default relu

        self.convs = torch.nn.ModuleList()
        self.convs.append(MLP_model(nfeat, nhid, nhid, self.MLP_num_layers,
                            dropout=self.dropout_rate, normalization=self.MLP_norm))
        for _ in range(self.num_layers):
            self.convs.append(HCHAConv(nhid, nhid, nhid, nfeat, self.dropout_rate, self.MLP_num_layers, self.MLP_norm, self.degree_detached, use_attention))
            
        self.convs.append(MLP_model(nhid, nhid, nclass, 2,
                            dropout=0.0, normalization=self.MLP_norm))
        self.reg_params = list(self.convs[1:-1].parameters())
        self.non_reg_params = list(self.convs[0:1].parameters())+list(self.convs[-1:].parameters())
        self.dropout = nn.Dropout(self.dropout_rate) # 0.2 is chosen for GCNII


    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, feats, omega, H):
        #feats is num_nodes x ts x feat
        x = feats
        x0 = x
        x = torch.permute(x, (1,0,2)) # x is ts x num_nodes x feat
        x0 = torch.permute(x0, (1,0,2))

    
        x = self.convs[0](x)
        for i,con in enumerate(self.convs[1:-1]):
            x = con(x, omega.unsqueeze(-1).float(), H, x0)
            x = self.dropout(x)

        x = self.convs[-1](x)

        x = x0 + x
        x = torch.permute(x, (1,0,2)) # x is num_nodes x ts x feat
        return x