LITERAL,DECOMPOSITION,TRUE = 0,1,2
import pdb


class Node:

    node_id=1
    def __init__(self, elements=None, type=DECOMPOSITION):
        self.elements = elements
        self.id = Node.node_id
        self.type=type
        if self.type==LITERAL:
            self.literal = elements
        Node.node_id += 1

    def __repr__(self):
        return str(self.id)

    def is_decomposition(self):
        return self.type ==  DECOMPOSITION

    def is_literal(self):
        return self.type ==  LITERAL

    def is_true(self):
        return self.type ==  TRUE

    def clear_bits(self,clear_data=False):
        """Recursively clears bits.  For use when recursively navigating an
        SDD by marking bits (not for use with SddNode.as_list).
        Set clear_data to True to also clear data."""
        if self._bit is False: return
        self._bit = False
        if clear_data: self.data = None
        if self.is_decomposition():
            for p,s in self.elements:
                p.clear_bits(clear_data=clear_data)
                s.clear_bits(clear_data=clear_data)

    def positive_iter(self,first_call=True,clear_data=False):
        """post-order (children before parents) generator, skipping false SDD
        nodes"""

        if not hasattr(self, '_bit'):
            self._bit = False

        if self._bit: return
        self._bit = True

        if self.is_decomposition():
            for p,s in self.elements:
                for node in p.positive_iter(first_call=False): yield node
                for node in s.positive_iter(first_call=False): yield node
        yield self

        if first_call:
            self.clear_bits(clear_data=clear_data)
        
import numpy as np
import math
from itertools import chain
import pickle


def lookup_node(elements, nodes, literals):
    elements = tuple(elements)
    el = nodes.get(elements)
    if not el:

        # For creating the circuit
        n = Node()
        n.elements = []
        for e in elements:
            p, s = e
            if p.type == DECOMPOSITION:
                p = nodes.get(tuple(p.elements))
            else:
                p = literals.get(p.elements)
            if s.type == DECOMPOSITION:
                s = nodes.get(tuple(s.elements))
            else:
                s = literals.get(s.elements)
            n.elements.append((p, s))
        nodes[tuple(n.elements)] = n

        el = n

    return el


def create_exactly_k(n, k):
    literals = dict(list(chain.from_iterable(
        ((i, Node(i, type=LITERAL)), (-i, Node(-i, type=LITERAL))) for i in
        range(1, n + 1))))
    nodes = {}

    dp_prev = np.ndarray((n, k + 1), dtype=Node)
    dp_prev.fill(None)
    for i in range(n):
        for j in range(2):
            dp_prev[i][j] = literals[-(i + 1)] if not j else literals[i + 1]

    for num_arr in (n // (2 ** i) for i in range(1, int(math.log2(n)) + 1)):
        dp_curr = np.ndarray((num_arr, k + 1), dtype=Node)
        dp_curr.fill(None)
        for i in range(0, num_arr):
            for j in range(0, k + 1):
                if n // num_arr < j: break
                l = []
                for jj in range(j + 1):
                    if (dp_prev[(i * 2), jj] and dp_prev[(i * 2) + 1, j - jj]):
                        l.append((dp_prev[(i * 2), jj], dp_prev[(i * 2) + 1, j - jj]))
                dp_curr[i, j] = lookup_node(l, nodes, literals)

        dp_prev = dp_curr

    return dp_curr


def create_and_save(n, k, root):
    alpha = create_exactly_k(n, k)[0][-1]
    with open(f'{root}/{n}C{k}.pkl', 'wb') as out:
        pickle.dump(alpha, out, pickle.HIGHEST_PROTOCOL)
        print(f"{n}C{k} done")

import os
import pickle
from collections import defaultdict
from typing import List

import torch
import torch._dynamo
torch._dynamo.reset()

DISABLE = False
MODE = 'default'
# MODE = 'reduce-overhead'


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def levelwiseSL(levels: List[torch.Tensor], idx2primesub: torch.Tensor,
                data: torch.Tensor, theta: torch.Tensor):
    for level in levels:
        theta[level] = data[idx2primesub[level]].sum(-2)
        data[level] = theta[level].logsumexp(-2)
        theta[level] -= data[level].unsqueeze(1)
    return data[levels[-1]]


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def levelwiseMars(levels: List[torch.Tensor], idx2primesub: torch.Tensor,
                  data: torch.Tensor, theta: torch.Tensor, parents: torch.Tensor):
    for level in reversed(levels):
        data[level] = (theta[parents[level].unbind(-1)] + data[
            parents[level].unbind(-1)[0]]).logsumexp(-2)


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def log1mexp(x):
    # Source: https://github.com/wouterkool/estimating-gradients-without-replacement/blob/9d8bf8b/bernoulli/gumbel.py#L7-L11
    # Computes log(1-exp(-|x|))
    # See https://cran.r-project.org/web/packages/Rmpfr/vignettes/log1mexp-note.pdf
    x = -x.abs()
    x = torch.where(
        x > -0.6931471805599453094,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )

    return x


def levelOrder(beta):
    """
    :type root: Node
    :rtype: List[List[int]]
    """
    seen = dict()
    nodes = [beta]
    level = []
    answer = []
    result = [[beta]]
    while len(nodes) != 0:
        for a in nodes:
            if not a.is_decomposition():
                continue
            for element in a.elements:
                for e in element:
                    if not e.is_decomposition():
                        continue
                    if seen.get(e) != None: continue
                    seen[e] = True
                    level.append(e)
        nodes = level
        for i in level:
            answer.append(i)
        level = []
        answer = list(dict.fromkeys(answer))
        result.append(answer)
        answer = []
    return result[:-1]


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def gumbel_keys(w, time_sampled):
    # sample some gumbels
    uniform = torch.rand((time_sampled,) + w.shape, device=w.device)  # .to(device)
    z = -torch.log(-torch.log(uniform))
    w = w + z
    return w


@torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
def sample_subset(w, k, time_sampled):
    '''
    Args:
        w (Tensor): Float Tensor of weights for each element. In gumbel mode
            these are interpreted as log probabilities
        k (int): number of elements in the subset sample
    '''
    with torch.no_grad():
        w = gumbel_keys(w, time_sampled)
        return w.topk(k, dim=-1).indices


class Layer:
    def __init__(self, n, k, device, root='./simple_configs'):

        if not os.path.isdir(root):
            os.mkdir(root)

        if not os.path.isfile(f'{root}/{n}C{k}.pkl'):
            create_and_save(n, k, root)
        with open(f'{root}/{n}C{k}.pkl', 'rb') as inp:
            beta = pickle.load(inp)

        max_elements = 0
        for node in beta.positive_iter():
            if node.is_decomposition():
                max_elements = max(max_elements, len(node.elements))

        levels_nodes = levelOrder(beta)

        # Reset ids
        nodes = [node for node in beta.positive_iter()]
        nodes = list(dict.fromkeys(nodes))

        id = 0
        for e in nodes:
            e.id = id
            id += 1
        self.id = id

        parents_dict = defaultdict(list)
        for node in beta.positive_iter():
            if node.is_decomposition():
                for i, (p, s) in enumerate(node.elements):
                    parents_dict[p.id] += [[node.id, i]]
                    parents_dict[s.id] += [[node.id, i]]

        # Set up the parents for an efficient backward pass
        max_parents = 0
        for p in parents_dict.values():
            max_parents = max(len(p), max_parents)

        parents = torch.empty((id, max_parents, 2), dtype=torch.int, device=device).fill_(id)
        for k, v in parents_dict.items():
            parents[k] = torch.tensor(v + [[id, 0]] * (max_parents - len(v)),
                                      dtype=torch.int, device=device)  # .to(device)
            # parents[k] = torch.nn.functional.pad(tmp, (0,0,0,max_parents - len(tmp)), value=id)
        self.parents = parents

        # Levels
        levels = []
        for level in levels_nodes:
            levels.append(torch.tensor([l.id for l in level], dtype=torch.int, device=device))
        levels.reverse()
        self.levels = levels

        # true indices
        true_indices = torch.tensor([node.id for node in nodes if node.is_true()], dtype=torch.int, device=device)
        self.true_indices = true_indices

        # Literal indices
        literal_indices = torch.tensor(
            [[node.id, node.literal] for node in nodes if node.is_literal()],
            dtype=torch.int, device=device)
        literal_indices, literal_mask = literal_indices.unbind(-1)
        literal_mask = literal_mask.abs() - 1, (literal_mask > 0).long()
        self.literal_indices = literal_indices
        self.literal_mask = literal_mask

        order = self.literal_mask[0][self.literal_mask[1].bool()].sort().indices
        self.pos_literals = self.literal_indices[self.literal_mask[1].bool()][order]

        # Map nodes to their primes/subs
        idx2primesub = torch.zeros((id, max_elements, 2), dtype=torch.int, device=device)
        for node in nodes:
            if node.is_decomposition():
                tmp = torch.tensor([[p.id, s.id] for p, s in node.elements],
                                   dtype=torch.int)
                idx2primesub[node.id] = torch.nn.functional.pad(tmp, (
                0, 0, 0, max_elements - len(tmp)), value=id)
        self.idx2primesub = idx2primesub

    def __call__(self, log_probs, k):
        samples = self.sample(log_probs, k)
        marginals = self.log_pr(log_probs).exp().permute(1, 0)
        return (samples - marginals).detach() + marginals, marginals

    # @torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
    def log_pr(self, log_probs):
        lit_weights = torch.stack((log1mexp(-log_probs.detach()), log_probs), dim=-1).permute(1, 2, 0)

        # data = torch.empty(self.id + 1, log_probs.size(0), device=log_probs.device)
        data = torch.zeros(self.id + 1, log_probs.size(0), device=log_probs.device)
        theta = torch.zeros(self.id + 1, self.idx2primesub.size(1), log_probs.size(0), device=log_probs.device)

        data[self.true_indices] = 0
        data[self.id] = -float(1000)
        data[self.literal_indices] = lit_weights[self.literal_mask[0], self.literal_mask[1]]

        # import pdb; pdb.set_trace()
        res = levelwiseSL(self.levels, self.idx2primesub, data, theta)
        data[self.levels[-1]] -= data[self.levels[-1]]
        levelwiseMars([self.literal_indices] + self.levels[:-1], self.idx2primesub, data, theta, self.parents)

        return data[self.pos_literals]

    @torch.compile(fullgraph=True, mode=MODE, disable=DISABLE)
    def sample(self, lit_weights, k, time_sampled = 1):
        with torch.no_grad():
            samples = sample_subset(lit_weights, k, time_sampled)
            samples_hot = lit_weights.new_zeros((time_sampled,) + lit_weights.shape)
            samples_hot.scatter_(2, samples, 1)
            return samples_hot.float()

def select_candidates(scores: torch.Tensor, k: int):
    Batch, Nmax, ensemble = scores.shape
    if k >= Nmax:
        return scores.new_ones(scores.shape)

    thresh = torch.topk(scores, k, dim=1, largest=True, sorted=True).values[:, -1, :][:, None, :]
    mask = (scores >= thresh).to(torch.float)
    return mask
def self_defined_softmax(scores, mask):
    """
    A specific function

    Args:
        scores: B, N, N, E
        mask: same shape as scores

    Returns:

    """
    scores = scores - scores.detach().max()  # for numerical stability
    exp_scores = torch.exp(scores)
    exp_scores = exp_scores * mask
    softmax_scores = exp_scores / exp_scores.sum()
    return softmax_scores
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LARGE_NUMBER = 1.e10

def logsigmoid(x):
    return -F.softplus(-x) + 1.e-7


class EdgeSIMPLEBatched(nn.Module):
    def __init__(self,
                 k,
                 device,
                 val_ensemble=1,
                 train_ensemble=1,
                 logits_activation=None,
                 nb_sample=1):
        super(EdgeSIMPLEBatched, self).__init__()
        self.k = k
        self.device = device
        self.layer_configs = dict()
        self.adj = None  # for potential usage
        assert val_ensemble > 0 and train_ensemble > 0
        self.val_ensemble = val_ensemble
        self.train_ensemble = train_ensemble
        self.logits_activation = logits_activation
        self.nb_sample = nb_sample

    def forward(self, scores, train = True):
        times_sampled = self.train_ensemble if train else self.val_ensemble

        bsz, Nmax, ensemble = scores.shape
        flat_scores = scores.permute((0, 2, 1)).reshape(bsz * ensemble, Nmax)
        target_size = Nmax
        local_k = min(self.k, Nmax)


        N = 2 ** math.ceil(math.log2(target_size))
        if (N, local_k) in self.layer_configs:
            layer = self.layer_configs[(N, local_k)]
        else:
            layer = Layer(N, local_k, self.device)
            self.layer_configs[(N, local_k)] = layer

        # padding
        flat_scores = torch.cat(
            [flat_scores,
             torch.full((flat_scores.shape[0], N - flat_scores.shape[1]),
                        fill_value=-LARGE_NUMBER,
                        dtype=flat_scores.dtype,
                        device=flat_scores.device)],
            dim=1)

        #default logits activation is none
        if self.logits_activation == 'None' or self.logits_activation is None:
            pass
        elif self.logits_activation == 'logsoftmax':
            # todo: it is bad heuristic to detect the padding
            masks = (flat_scores.detach() > - LARGE_NUMBER / 2).float()
            flat_scores = torch.vmap(self_defined_softmax, in_dims=0, out_dims=0)(flat_scores, masks)
            flat_scores = torch.log(flat_scores + 1 / LARGE_NUMBER)
        elif self.logits_activation == 'logsigmoid':
            # todo: sigmoid is not good, it makes large scores too similar, i.e. close to 1.
            flat_scores = logsigmoid(flat_scores)
        else:
            raise NotImplementedError

        # we potentially need to sample multiple times
        marginals = layer.log_pr(flat_scores).exp().permute(1, 0)
        # (times_sampled) x (B x E) x (N x N)
        samples = layer.sample(flat_scores, local_k, times_sampled)
        samples = (samples - marginals[None]).detach() + marginals[None]

        # unpadding
        samples = samples[..., :target_size]
        marginals = marginals[:, :target_size]

        # VE x (B x E) x Nmax -> VE x B x Nmax x E
        new_mask = samples.reshape(times_sampled, bsz, ensemble, Nmax).permute((0, 1, 3, 2))
        # (B x E) x Nmax -> B x Nmax x E
        new_marginals = marginals.reshape(bsz, ensemble, Nmax).permute((0, 2, 1))
        # print(new_marginals.max(), new_mask.max())
        # if new_mask.max()==0 or torch.isnan(new_mask).any(): 
        #     pdb.set_trace()
        return new_mask, new_marginals

    @torch.no_grad()
    def validation(self, scores):
        """
        during the inference we need to margin-out the stochasticity
        thus we do top-k once or sample multiple times

        Args:
            scores: shape B x N x N x E

        Returns:
            mask: shape B x N x N x (E x VE)

        """
        if self.nb_sample == 1:
            _, marginals = self.forward(scores, False)

            # do deterministic top-k
            mask = select_candidates(scores, self.k)

            return mask[None], marginals
        else:
            return self.forward(scores, False)

def get_simple_sampler(sample_k, device, num_train_ensemble=1.0, num_val_ensemble=1.0, logits_activation=None, nb_sample=1):
    # logits_activation = 'None' 
    # logits_activation = 'logsoftmax'
    # logsigmoid
    simple_sampler = EdgeSIMPLEBatched(sample_k,
                                            device,
                                            val_ensemble=num_val_ensemble,
                                            train_ensemble=num_train_ensemble,
                                            logits_activation=logits_activation,
                                            nb_sample=nb_sample)
    train_forward = simple_sampler
    val_forward = simple_sampler.validation
    return train_forward, val_forward