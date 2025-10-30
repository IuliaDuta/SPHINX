import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *


class TempCNN_model(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, num_features, MLP_hidden, out_channels, num_layers, dropout, normalization, 
                    InputNorm=False, kernel_size=(2,3), in_features=2):
        super(TempCNN_model, self).__init__()
        in_channels = num_features
        hidden_channels = MLP_hidden
        out_channels = out_channels
        num_layers = num_layers
        dropout = dropout
        Normalization = normalization

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm
        self.kernel_size = kernel_size

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Conv2d(in_channels, out_channels, self.kernel_size))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Conv2d(in_channels, hidden_channels, (in_features,3)))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Conv2d(hidden_channels, hidden_channels, (1,3)))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Conv2d(hidden_channels, out_channels, (1,3)))
        elif Normalization == 'ln':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Conv2d(in_channels, out_channels, self.kernel_size))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Conv2d(in_channels, hidden_channels, self.kernel_size))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Conv2d(hidden_channels, hidden_channels, (1,3)))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Conv2d(hidden_channels, out_channels, (1,3)))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Conv2d(in_channels, out_channels, self.kernel_size))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Conv2d(in_channels, hidden_channels, self.kernel_size))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Conv2d(hidden_channels, hidden_channels, (1,3)))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Conv2d(hidden_channels, out_channels, (1,3)))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class MLP_model(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, num_features, MLP_hidden, out_channels, num_layers, dropout, normalization, InputNorm=False):
        super(MLP_model, self).__init__()
        in_channels = num_features
        hidden_channels = MLP_hidden
        out_channels = out_channels
        num_layers = num_layers
        dropout = dropout
        Normalization = normalization

        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm

        assert Normalization in ['bn', 'ln', 'None']
        if Normalization == 'bn':
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.BatchNorm1d(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.BatchNorm1d(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        elif Normalization == 'ln':
            if num_layers == 1:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                if InputNorm:
                    self.normalizations.append(nn.LayerNorm(in_channels))
                else:
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.LayerNorm(hidden_channels))
                self.lins.append(nn.Linear(hidden_channels, out_channels))
        else:
            if num_layers == 1:
                # just linear layer i.e. logistic regression
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, out_channels))
            else:
                self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(in_channels, hidden_channels))
                self.normalizations.append(nn.Identity())
                for _ in range(num_layers - 2):
                    self.lins.append(
                        nn.Linear(hidden_channels, hidden_channels))
                    self.normalizations.append(nn.Identity())
                self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    

class DenseAllDeepSetsConv(nn.Module):
    def __init__(self, in_features, hid_features, out_features, nfeat, dropout, MLP_num_layers, MLP_norm, degree_detached):
        super().__init__()
    
        self.MLP1 = MLP_model(in_features, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        self.MLP2 = MLP_model(hid_features, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)

        self.W1 = MLP_model(2*hid_features, hid_features, out_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        self.W2 = MLP_model(in_features+nfeat, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm) #in_features+out_features
        self.W3 = MLP_model(2, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        self.degree_detached = degree_detached

    def reset_parameters(self):
        self.W1.reset_parameters()
        self.W2.reset_parameters()
        self.W3.reset_parameters()
        self.MLP1.reset_parameters()
        self.MLP2.reset_parameters()


    def forward(self, X, Z, H, X0):
        # Z is num_nodes x 1
        # H is num_nodes x num_edges
        # X is ts x num_nodes x feat

        T = X.shape[0]
        N = X.shape[1]
       
        Xe = H.transpose(1,0) @ self.MLP1(X) #ts x num_edges x feat
        Zve = Z #num_nodes 

        Xv = H @ self.MLP2(Xe) #ts x num_nodes x feat

        feats = torch.cat((X0, Xv), axis=-1) #num_nodes x feat
        rotation_angle = torch.cat((torch.cos(Zve*0.1), torch.sin(Zve*0.1)), axis=-1)
        rotation_angle = rotation_angle.unsqueeze(0).tile((feats.shape[0],1,1))
        
        X = self.W1(torch.cat((self.W2(feats), self.W3(rotation_angle)), axis=-1))
     
        return X
       

class DenseAllDeepSetsConvNBA(nn.Module):
    def __init__(self, in_features, hid_features, out_features, nfeat, dropout, MLP_num_layers, MLP_norm, degree_detached):
        super().__init__()
    
        self.MLP1 = MLP_model(in_features, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        self.MLP2 = MLP_model(hid_features, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)

        self.W1 = MLP_model(in_features+2, hid_features, out_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        
        self.degree_detached = degree_detached

    def reset_parameters(self):
        self.W1.reset_parameters()
        self.MLP1.reset_parameters()
        self.MLP2.reset_parameters()


    def forward(self, X, H, X0):
        # Z is num_nodes x 1
        # H is num_nodes x num_edges
        # X is ts x num_nodes x feat

        T = X.shape[0]
        N = X.shape[1]
        
        Xe = H.transpose(1,0) @ self.MLP1(X) #ts x num_edges x feat
        Xv = H @ self.MLP2(Xe) #ts x num_nodes x feat

        feats = torch.cat((X0, Xv), axis=-1) #num_nodes x feat

        X = self.W1(feats)
     
        return X


class DenseAllDeepSetsConvNBAOrig(nn.Module):
    def __init__(self, in_features, hid_features, out_features, nfeat, dropout, MLP_num_layers, MLP_norm, degree_detached):
        super().__init__()
    
        self.MLP1_1 = MLP_model(in_features, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        self.MLP1_2 = MLP_model(hid_features, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        self.MLP2_1 = MLP_model(hid_features, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        self.MLP2_2 = MLP_model(hid_features, hid_features, out_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)

        self.W1 = MLP_model(hid_features+2, hid_features, out_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        
        self.degree_detached = degree_detached

    def reset_parameters(self):
        self.MLP1_1.reset_parameters()
        self.MLP1_2.reset_parameters()
        self.MLP2_1.reset_parameters()
        self.MLP2_2.reset_parameters()



    def forward(self, X, H, X0):
        # Z is num_nodes x 1
        # H is num_nodes x num_edges
        # X is ts x num_nodes x feat

        T = X.shape[0]
        N = X.shape[1]
        
        Xe = F.relu(self.MLP1_2(H.transpose(1,0) @ F.relu(self.MLP1_1(X)))) #ts x num_edges x feat
        Xv = H @ F.relu(self.MLP2_1(Xe)) #ts x num_nodes x feat

        feats = torch.cat((X0, Xv), axis=-1) #num_nodes x feat
        X = self.W1(feats)
     
        return X

class HCHAConv(nn.Module):
    def __init__(self, in_features, hid_features, out_features, nfeat, dropout, 
                        MLP_num_layers, MLP_norm, degree_detached, use_attention):
        super().__init__()
    
        self.W1 = MLP_model(2*hid_features, hid_features, out_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        self.W2 = MLP_model(in_features+nfeat, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm) #in_features+out_features
        self.W3 = MLP_model(2, hid_features, hid_features, MLP_num_layers,
                            dropout=dropout, normalization=MLP_norm)
        self.degree_detached = degree_detached
        self.use_attention = use_attention
        if use_attention:
            self.lin1 = MLP_model(hid_features, hid_features, 1, 1,
                            dropout=dropout, normalization=MLP_norm) #in_features+out_features
            self.lin2 = MLP_model(hid_features, hid_features, 1, 1,
                            dropout=dropout, normalization=MLP_norm) #in_features+out_features

    def reset_parameters(self):
        self.W1.reset_parameters()
        self.W2.reset_parameters()
        self.W3.reset_parameters()
        if self.use_attention:
            self.lin1.reset_parameters()
            self.lin2.reset_parameters()

    def forward(self, X, Z, H, X0):
        # Z is num_nodes x 1
        # H is num_nodes x num_edges
        # X is ts x num_nodes x feat

        T = X.shape[0]
        N = X.shape[1]

        if self.use_attention:
            B = torch.nan_to_num(torch.pow(torch.diag(torch.sum(H, dim=0)), -1.0), 
                                nan=0, posinf=0, neginf=0).to(torch.float32) # this is to replace the mean aggregator
   
            hyperedge_attr =  (B @ H.transpose(1,0)).detach() @ X #ts x num_edges x feat
            num_edges = hyperedge_attr.shape[1]
            num_nodes = X.shape[1]
            # x: ts x num_nodes x feat
            x_sim = torch.tile(X.unsqueeze(2),(1,1,num_edges,1)) #ts x num_nodes x num_edges x feat
            e_sim = torch.tile(hyperedge_attr.unsqueeze(1), (1,num_nodes,1,1)) #ts x num_nodes x num_edges x feat
            x_sim = self.lin1(x_sim)
            e_sim = self.lin2(e_sim)

            sim =  (x_sim + e_sim).squeeze(-1)
            sim = F.leaky_relu(sim, 0.2)
            H = F.sigmoid(sim) * H
            
            Xe = H.transpose(2,1) @ X #ts x num_edges x feat

            Zve = Z #num_nodes 

            Xv = H @ Xe #ts x num_nodes x feat

        else:
            Xe = H.transpose(1,0) @ X #ts x num_edges x feat
            Zve = Z #num_nodes 
            Xv = H @ Xe #ts x num_nodes x feat

        feats = torch.cat((X0, Xv), axis=-1)

        rotation_angle = torch.cat((torch.cos(Zve*0.1), torch.sin(Zve*0.1)), axis=-1)
        rotation_angle = rotation_angle.unsqueeze(0).tile((feats.shape[0],1,1))

        X = self.W1(torch.cat((self.W2(feats), self.W3(rotation_angle)), axis=-1))
        return X
  

