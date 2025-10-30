import torch
from numpy import linspace
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

import itertools
import time
from torch.autograd import Variable
from scipy.interpolate import interp1d



def my_softmax(input, axis=1):
    trans_input = input.transpose(axis, 0).contiguous()
    soft_max_1d = F.softmax(trans_input)
    return soft_max_1d.transpose(axis, 0)

def edge_index_to_list(edge_index):
    edge_list = []
    key_func = lambda x: x[1]
    for key, group in itertools.groupby(edge_index.transpose(), key_func):
        edge_list.append(np.array(list(group)))
    return np.array(edge_list)[:,:,0]

def get_iou_k_slots(batch, slots):
    "super inefficient but it basically brutforce maximal matching between prediction and gt"
    "gt and pred have the same number of hyperedges"
    "gt and pred represent 3-regular hypergraphs"
    
    num_slots = slots.shape[1]
    gt = np.array([edge_index_to_list(batch[i].edge_index.detach().cpu().numpy()) for i in range(len(batch))])
    gt = F.one_hot(torch.from_numpy(gt), num_classes=6).numpy().sum(-2)
    pred = slots.detach().cpu().numpy().astype(int)
    # pred: bs x num_slots x num_nodes
    # gt: bs x num_slots x num_nodes

    intersection = np.zeros(gt.shape[0])

    for perm in itertools.permutations(list(range(num_slots))):
        pred_rotate = pred[:, perm, :]
        #this works seens both gt and pred has the same number of 1s
        intersection_2 = (gt & pred_rotate).sum(-1).sum(-1) #.sum() 
        intersection = np.maximum(intersection,intersection_2)
    
    intersection = intersection.sum()/gt.sum()
    return intersection
    


def block_diagonal(*arrs):
    bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
    if bad_args:
        raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)
    shapes = torch.tensor([a.shape for a in arrs])
    i = []
    v = []
    r, c = 0, 0
    for k, (rr, cc) in enumerate(shapes):
        first_index = torch.arange(r, r + rr, device=arrs[0].device)
        second_index = torch.arange(c, c + cc, device=arrs[0].device)
        index = torch.stack((first_index.tile((cc,1)).transpose(0,1).flatten(), second_index.repeat(rr)), dim=0)
        i += [index]
        v += [arrs[k].flatten()]
        r += rr
        c += cc
    out_shape = torch.sum(shapes, dim=0).tolist()

    if arrs[0].device == "cpu":
        out = torch.sparse.DoubleTensor(torch.cat(i, dim=1), torch.cat(v), out_shape)
    else:
        out = torch.cuda.sparse.DoubleTensor(torch.cat(i, dim=1).to(arrs[0].device), torch.cat(v), out_shape)
    return out

class LearnedParamChecker():
    def __init__(self,model):
        self.model = model
        self.initial_params = self.save_initial_params()
    
    def save_initial_params(self):
        initial_params = {}
        for name, p in self.model.named_parameters():
            initial_params[name] = p.detach().cpu().numpy()
        return initial_params

    def compare_current_initial_params(self):
        changed = True
        for name, current_p in self.model.named_parameters():
            initial_p = self.initial_params[name]
            current_p = current_p.detach().cpu().numpy()
            diff = np.mean(np.abs(initial_p - current_p))
            # print(f'params: {name} mean change : {diff}')
            if diff < 1e-30:
                print(f'ATTENTION params: {name} mean change :', "{0:0.10f}".format(diff))
                changed = False
            # elif 'slot_attention' in name:
            else:
                print(f'ATTENTION params: {name} mean change :', "{0:0.10f}".format(diff))
            # print(f'ATTENTION params: {name} mean change : {diff}')
        return changed

def ccworder(A):
    A= A- A.mean(1)[:, None]
    return np.argsort(np.arctan2(A[1, :], A[0, :]))

def draw(locations, edges=None, name='foo.png', check=True, num_nodes=6, traingles=True):
  f = plt.figure()
  plt.clf()
  min_coord = min(locations[:,:,0].min(), locations[:,:,1].min()) - 0.5
  max_coord = max(locations[:,:,0].max(), locations[:,:,1].max()) + 0.5
  plt.xlim(min_coord, max_coord)
  # plt.ylim(min_coord, max_coord)

  start = 0.0
  stop = 1.0
  number_of_lines= num_nodes
  cm_subsection = linspace(start, stop, number_of_lines)

  colors = [ cm.Set1(x) for x in cm_subsection ]

  for i in range(locations.shape[1]):
      for t in range(locations.shape[0]):
          plt.plot(locations[t, i, 0], locations[t, i, 1], 'o', markersize=3, color=colors[i], alpha=1 -(float(t)/locations.shape[0]))
      plt.plot(locations[0, i, 0], locations[0, i, 1], 'o', color=colors[i])


  if edges is not None:
    cm_subsection2 = linspace(start, stop, len(edges))
    colors2 = [ cm.Set1(x) for x in cm_subsection2 ]
    for j, one_edge in enumerate(edges):
        # if len(list(one_edge)) == 3:
        if True:
            all_points = []
            for idx in one_edge:
                x,y = locations[0,idx,0], locations[0,idx,1]
                all_points.append([x,y])
            all_points = np.transpose(np.array(all_points))
            order_idx = ccworder(all_points) 
            sorted_all_points = [all_points[:,i] for i in order_idx]
            sorted_all_points = sorted_all_points + [all_points[:,0]]

            num_elements = len(sorted_all_points)
            triangle_x = [sorted_all_points[i][0] for i in range(num_elements)]
            triangle_y = [sorted_all_points[i][1] for i in range(num_elements)]

            plt.fill(triangle_x, triangle_y, alpha=0.2, color=colors2[j])

  #   plt.show()
  plt.axis('square')
  # plt.savefig(name)
  return f

def process_edges(list_edges):
  '''
  list_edges: num_trajectories' list of sets
  '''
  all_edge_index = []
#   all_v_edges = []
  #temporary we add self edges when there are no edges
  for j, edges in enumerate(list_edges):
    # vs = v_edges[j]
    edge_list = []
    vertex_list = []
    idx = 0
    nodes_app = []
    for hedge in edges:
        edge_list += hedge
        vertex_list += [idx for _ in range(len(hedge))]
        idx = idx + 1
        for node_i in hedge:
            if node_i not in nodes_app:
                nodes_app.append(node_i)

    edge_index = torch.tensor([edge_list, vertex_list])
    all_edge_index.append(edge_index)
  return all_edge_index


def set_diff_1d(t1, t2, assume_unique=False):
    """
    Set difference of two 1D tensors.
    Returns the unique values in t1 that are not in t2.

    """
    if not assume_unique:
        t1 = torch.unique(t1)
        t2 = torch.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]

def add_extra_self_loops(edge_index, edge_weight, n_nodes):
    # this adds self loops only for isolated nodes 
    nodes_app = set(edge_index[0].cpu().detach().numpy())
    
    idx = max(edge_index[1]) + 1
    edge_index_0 = edge_index[0]
    edge_index_1 = edge_index[1]

    device = edge_index_0.device
    isolated_nodes = list(set(range(n_nodes)).difference(nodes_app))

    isolated_nodes = torch.tensor(isolated_nodes).long().to(device)
    isolated_edge_index = torch.tensor(np.arange(isolated_nodes.shape[0])).to(device)+idx
    isolated_edge_weights = torch.ones(isolated_nodes.shape[0]).float().to(device)


    edge_index_0 = torch.cat((edge_index_0, isolated_nodes), axis=-1)
    edge_index_1 = torch.cat((edge_index_1, isolated_edge_index), axis=-1)

    new_edge_index = torch.stack([edge_index_0, edge_index_1], axis=0)
    new_edge_weight = torch.cat([edge_weight, isolated_edge_weights], axis=0)

    return new_edge_index, new_edge_weight

def add_extra_self_loops_dense(H, n_nodes):
    # H: batch_nodes x batch_edges 
    # BATCH LEVEL!!! 
    # this adds self loops only for isolated nodes 
    edge_index_0 = H.to_sparse().indices()[0]
    nodes_app = set(edge_index_0.cpu().detach().numpy())
    
    device = H.device
    isolated_nodes = list(set(range(n_nodes)).difference(nodes_app))

    isolated_nodes = torch.tensor(isolated_nodes).long().to(device)
    isolated_H = F.one_hot(isolated_nodes, num_classes=n_nodes)
    isolated_H = torch.permute(isolated_H, (1,0))
    
    new_H = torch.cat((H, isolated_H), dim=1)

    return new_H

class Sparsemax(nn.Module):
    """Sparsemax function."""

    def __init__(self, dim=None):
        """Initialize sparsemax activation
        
        Args:
            dim (int, optional): The dimension over which to apply the sparsemax function.
        """
        super(Sparsemax, self).__init__()

        self.dim = -1 if dim is None else dim

    def forward(self, input, device):
        """Forward function.

        Args:
            input (torch.Tensor): Input tensor. First dimension should be the batch size

        Returns:
            torch.Tensor: [batch_size x number_of_logits] Output tensor

        """
        # Sparsemax currently only handles 2-dim tensors,
        # so we reshape to a convenient shape and reshape back after sparsemax
        input = input.transpose(0, self.dim)
        original_size = input.size()
        input = input.reshape(input.size(0), -1)
        input = input.transpose(0, 1)
        dim = 1

        number_of_logits = input.size(dim)

        # Translate input by max for numerical stability
        input = input - torch.max(input, dim=dim, keepdim=True)[0].expand_as(input)

        # Sort input in descending order.
        # (NOTE: Can be replaced with linear time selection method described here:
        # http://stanford.edu/~jduchi/projects/DuchiShSiCh08.html)
        zs = torch.sort(input=input, dim=dim, descending=True)[0]
        range = torch.arange(start=1, end=number_of_logits + 1, step=1, device=device, dtype=input.dtype).view(1, -1)
        range = range.expand_as(zs)

        # Determine sparsity of projection
        bound = 1 + range * zs
        cumulative_sum_zs = torch.cumsum(zs, dim)
        is_gt = torch.gt(bound, cumulative_sum_zs).type(input.type())
        k = torch.max(is_gt * range, dim, keepdim=True)[0]

        # Compute threshold function
        zs_sparse = is_gt * zs

        # Compute taus
        taus = (torch.sum(zs_sparse, dim, keepdim=True) - 1) / k
        taus = taus.expand_as(input)

        # Sparsemax
        self.output = torch.max(torch.zeros_like(input), input - taus)

        # Reshape back to original shape
        output = self.output
        output = output.transpose(0, 1)
        output = output.reshape(original_size)
        output = output.transpose(0, self.dim)

        return output

    def backward(self, grad_output):
        """Backward function."""
        dim = 1

        nonzeros = torch.ne(self.output, 0)
        sum = torch.sum(grad_output * nonzeros, dim=dim) / torch.sum(nonzeros, dim=dim)
        self.grad_input = nonzeros * (grad_output - sum.expand_as(grad_output))

        return self.grad_input

# From: https://github.com/alexmonti19/dagnet/blob/master/models/utils/utils.py#L76
def average_displacement_error(pred_traj, pred_traj_gt, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (sample, batch, seq_len,  2). Predicted trajectories.
    - pred_traj_gt: Tensor of shape (sample, batch, seq_len, 2). Ground truth predictions.
    Output:
    - error: total sum of displacement errors across all sequences inside the batch
    """

    loss = pred_traj_gt - pred_traj
    loss = loss ** 2
    loss = torch.sqrt(torch.sum(loss, dim=-1)) # over channles
    loss = torch.mean(loss, dim=2) # over timesteps
    loss, _ = torch.min(loss, dim=0) # over samples
    loss = torch.sum(loss, dim=0) # over batch size
    # if mode == 'sum':
    #     return torch.sum(loss)
    # elif mode == 'raw':
    #     return loss
    return loss

def final_displacement_error(pred_pos, pred_pos_gt, mode='sum'):
    """
    Input:
    - pred_pos: Tensor of shape (sample, batch, 1, 2). Predicted final positions.
    - pred_pos_gt: Tensor of shape (sample, batch, 1, 2). Ground truth final positions.
    Output:
    - error: total sum of fde for all the sequences inside the batch
    """
    loss = (pred_pos - pred_pos_gt) ** 2
    loss = torch.sqrt(torch.sum(loss, dim=-1)) # over channels
    loss, _ = torch.min(loss, dim=0) # over samples
    loss = torch.sum(loss, dim=0) # over batch size
    return loss

def predict_metrics(prediction, y):
    # pred_traj: bs x num_timesteps x num_nodes 
    l2error_avg = {}
    l2error_dest = {}
    for ts in range(10):
        l2error_avg[4*(ts+1)] = average_displacement_error(y[:, :,:ts+1,:], prediction[:, :,:ts+1,:]).item()
        l2error_dest[4*(ts+1)] = final_displacement_error(y[:, :,ts:ts+1,:], prediction[:, :,ts:ts+1,:]).item()

    l2error_avg_1s = (l2error_avg[8] + l2error_avg[12])/2
    l2error_avg_2s = l2error_avg[20]
    l2error_avg_3s = (l2error_avg[28] + l2error_avg[32])/2
    l2error_avg_4s = l2error_avg[40]

    l2error_dest_1s = (l2error_dest[8] + l2error_dest[12])/2
    l2error_dest_2s = l2error_dest[20]
    l2error_dest_3s = (l2error_dest[28] + l2error_dest[32])/2
    l2error_dest_4s = l2error_dest[40]

    return np.array([l2error_avg_1s, l2error_avg_2s, l2error_avg_3s, l2error_avg_4s, l2error_dest_1s, l2error_dest_2s, l2error_dest_3s, l2error_dest_4s])



def sample_gumbel(shape, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from Gumbel(0, 1)

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    U = torch.rand(shape).float()
    return - torch.log(eps - torch.log(U + eps))


def gumbel_softmax_sample(logits, tau=1, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Draw a sample from the Gumbel-Softmax distribution

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb
    (MIT license)
    """
    gumbel_noise = sample_gumbel(logits.size(), eps=eps)
    if logits.is_cuda:
        gumbel_noise = gumbel_noise.cuda()
    y = logits + Variable(gumbel_noise)
    return my_softmax(y / tau, axis=-1)


def gumbel_softmax(logits, tau=1, hard=False, eps=1e-10):
    """
    NOTE: Stolen from https://github.com/pytorch/pytorch/pull/3341/commits/327fcfed4c44c62b208f750058d14d4dc1b9a9d3

    Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      tau: non-negative scalar temperature
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probability distribution that sums to 1 across classes

    Constraints:
    - this implementation only works on batch_size x num_features tensor for now

    based on
    https://github.com/ericjang/gumbel-softmax/blob/3c8584924603869e90ca74ac20a6a03d99a91ef9/Categorical%20VAE.ipynb ,
    (MIT license)
    """
    y_soft = gumbel_softmax_sample(logits, tau=tau, eps=eps)
    if hard:
        shape = logits.size()
        _, k = y_soft.data.max(-1)
        # this bit is based on
        # https://discuss.pytorch.org/t/stop-gradients-for-st-gumbel-softmax/530/5
        y_hard = torch.zeros(*shape)
        if y_soft.is_cuda:
            y_hard = y_hard.cuda()
        y_hard = y_hard.zero_().scatter_(-1, k.view(shape[:-1] + (1,)), 1.0)
        # this cool bit of code achieves two things:
        # - makes the output value exactly one-hot (since we add then
        #   subtract y_soft value)
        # - makes the gradient equal to y_soft gradient (since we strip
        #   all other gradients)
        y = Variable(y_hard - y_soft.data) + y_soft
    else:
        y = y_soft
    return y

