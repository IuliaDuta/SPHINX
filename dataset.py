import numpy as np

from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class HyperEdgeData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            # return torch.tensor([[self.num_nodes], [value[1].max().item() + 1]])
            num_edges = value[1].max().item() + 1
            #be careful. value[0].max().item() + 1 might return smt else because some nodes are isolated
            num_nodes =  self.num_nodes
            # num_nodes=6
            return torch.tensor([[num_nodes], [num_edges]])
        else:
            return super(HyperEdgeData, self).__inc__(key, value)

class DenseHyperEdgeData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            # return torch.tensor([[self.num_nodes], [value[1].max().item() + 1]])
            num_edges = value[1].max().item() + 1
            #be careful. value[0].max().item() + 1 might return smt else because some nodes are isolated
            num_nodes =  self.num_nodes
            # num_nodes=6
            return torch.tensor([[num_nodes], [num_edges]])

        else:
            return super(DenseHyperEdgeData, self).__inc__(key, value)
    def __cat_dim__(self, key, value, *args, **kwargs):
        if 'index' in key:
            return 1
        elif key == 'adj':
            return 0, 1
        else:
            return 0


class GroupInteractionDataset(Dataset):
    """ Dataset to store Spring trajectories. """

    def __init__(self, loc_inputs, loc_targets, edges, omega, v=0, rewired=False):
        """
        Args:
            loc_inputs: location for the input sequence n_sample x n_timesteps x n_nodes x 2
            loc_targets: location for the target sequence n_sample x n_timesteps x n_nodes x 2
            edges: hedge index for each trajector: list of n_sample elements of dim [ 2 x nnz]
        """
        super(GroupInteractionDataset, self).__init__()

        self.dim = loc_targets.shape[-1]
        self.n_timesteps = loc_targets.shape[1]
        self.n_sample = loc_targets.shape[0]
        self.n_nodes = loc_targets.shape[2]

        self.edges = edges
        self.loc_inputs = loc_inputs
        self.loc_targets = loc_targets
        self.omega = omega
        self.v = v

    def len(self):
        return len(self.edges)

    def get(self, idx):
        crt_edge_index = self.edges[idx] # 2 x nnz
        crt_loc_inputs = self.loc_inputs[idx]  / 10 # n_timesteps x n_nodes x 2 
        crt_loc_targets = self.loc_targets[idx] / 10 # n_timesteps x n_nodes x 2 
        crt_omega = self.omega[idx]

        x = torch.tensor(crt_loc_inputs).to(torch.float32)
        y = torch.tensor(crt_loc_targets).to(torch.float32)
        crt_omega = torch.tensor(crt_omega)
        # Create a graph with `x` as features, `y` as labels and its
        # connectivity given by `edge_index`.

        x = torch.permute(x, (1,0,2)) # n_nodes x n_timesteps x 2 
        y = torch.permute(y, (1,0,2)) # n_nodes x n_timesteps x 2 
        graph = HyperEdgeData(edge_index=crt_edge_index, x=x, y=y, omega=crt_omega)
        return graph


class NBADataset(Dataset):
    """
    Adapted from https://github.com/MediaBrain-SJTU/GroupNet/blob/main/data/dataloader_nba.py
    """
    """Dataloder for the Trajectory datasets"""
    
    def __init__(self, data_path=None, split=None, num_examples=0):
        super(NBADataset, self).__init__()
        data_root = data_path # 'datasets/nba/train.npy' or 'datasets/nba/test.npy'
        
        trajs = np.load(data_root) 
        trajs /= (94/28) # Turn to meters


        trajs = torch.from_numpy(trajs).type(torch.float)

        
        if split == 'train':
            trajs = trajs[:32500]
        elif split == 'valid':
            trajs = trajs[:12500]
        elif split == 'test':
            trajs = trajs[12501:25000]

        if num_examples != 0:
            trajs = trajs[:num_examples]

        # pdb.set_trace()
        self.trajs = trajs
        self.dim = self.trajs.shape[-1]
        self.n_timesteps = self.trajs.shape[1]
        self.n_sample = self.trajs.shape[0]
        self.n_nodes = self.trajs.shape[2]
        
        # print(self.traj_abs.shape)

    def len(self):
        return self.n_sample

    def add_category(self,x):
        N = x.shape[0]
        T = x.shape[1]
        category = torch.zeros(N,3).type_as(x)
        category[0:5,0] = 1
        category[5:10,1] = 1
        category[10,2] = 1
        category = category.unsqueeze(1).repeat(1,T,1)
        x = torch.cat((x,category),dim=-1)
        return x

    def get(self, index):
        crt_traj = self.trajs[index]  #time_steps x players x 2
        
        x = crt_traj[:-1,:,:]
        y = crt_traj[1:,:,:]

        x = torch.permute(x, (1,0,2)) # n_nodes x n_timesteps x 2 
        y = torch.permute(y, (1,0,2)) # n_nodes x n_timesteps x 2 
        
        x = self.add_category(x) # n_nodes x n_timesteps x 5

        # Create a graph with `x` as features, `y` as labels and its
        # connectivity given by `edge_index`.
        fake_edge_index = torch.ones((2,3)).to(torch.int32)
        graph = HyperEdgeData(edge_index=fake_edge_index, x=x, y=y)
        return graph