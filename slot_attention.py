import torch
from torch import nn
from torch.nn import init
import pdb
from utils import Sparsemax


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, temperature, nonlin, iters = 3, eps = 1e-8, hidden_dim = 128, 
                    init_type="rand", trick_fixed_point='none'):
        '''
        when init_type = rand: original randomly initialize slot attention
        when init_type = hgraph: init with the avg nodes from each hedge
        '''
        super().__init__()
        self.num_slots_per_graph = num_slots
        self.iters = iters
        self.eps = eps

        self.scale = dim ** -0.5
        self.init_type = init_type
        self.temperature = temperature
        self.nonlin = nonlin
        self.trick_fixed_point = trick_fixed_point

        self.slots_mu = nn.Parameter(torch.randn(1, dim), requires_grad=True)
        self.slots_logsigma = nn.Parameter(torch.zeros(1, dim), requires_grad=True)
        
        init.xavier_uniform_(self.slots_logsigma)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(inplace = True),
            nn.Linear(hidden_dim, dim)
        )

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)
        self.sparsemax = Sparsemax(dim=-1)

    def reset_parameters(self):
        self.norm_input.reset_parameters()
        self.norm_slots.reset_parameters()
        self.norm_pre_ff.reset_parameters()
        for layer in self.mlp.children():
            if not isinstance(layer, nn.ReLU):
                layer.reset_parameters()

        self.gru.reset_parameters()
        self.to_q.reset_parameters()
        self.to_k.reset_parameters()
        self.to_v.reset_parameters()
        if self.init_type == 'rand':
            init.xavier_uniform_(self.slots_logsigma)
            init.normal_(self.slots_mu, mean=1, std=1)
        elif self.init_type == 'rand_embedding':
            init.xavier_uniform_(self.slots_init.weight)

    def step(self, slots, k, v):
        # this is one iteration of self attention as applied in SPHINX
        slots = self.norm_slots(slots)
        q = self.to_q(slots)

        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        dots = dots / self.temperature

        if self.nonlin == 'softmax':
            attn = torch.nn.functional.softmax(dots, dim=-1) + self.eps
        elif self.nonlin == 'sigmoid':
            attn = torch.sigmoid(dots) + self.eps
        elif self.nonlin == 'sparsemax':
            attn = self.sparsemax(dots, device=slots.device)
      
        updates = torch.einsum('bjd,bij->bid', v, attn)
        slots = updates
        return slots, attn, dots

    def forward(self, inputs, num_slots = None, sigma=0):
        n_s = num_slots 
        batch_size = n_s//self.num_slots_per_graph

        mu = self.slots_mu.expand(n_s, -1)
        sigma = self.slots_logsigma.exp().expand(n_s, -1)
        
        g = torch.Generator()
        g.manual_seed(0)

        slots_init = mu + sigma * torch.normal(0, 1, size=sigma.shape, generator=g).to(sigma.device)
        slots_init = slots_init.reshape(-1, self.num_slots_per_graph, slots_init.shape[-1])
        slots = slots_init
       
        inputs = self.norm_input(inputs)   
        inputs = inputs.reshape(batch_size, -1, inputs.shape[-1]) # bs x num_nodes x inp_dim
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters-1):
            # apply self attention   
            slots, attn, dots = self.step(slots, k, v)

        if self.trick_fixed_point == 'none':
            slots, attn, dots = self.step(slots, k, v)
        elif self.trick_fixed_point == 'neumann':
            slots, attn, dots = self.step(slots.detach(), k, v)
        elif self.trick_fixed_point == 'bi-level':
            slots, attn, dots = self.step(slots.detach(), k, v)
            slots = slots + slots_init - slots_init.detach()    
    
        final_slots = slots
        final_attn = attn
        final_dots = dots

        if torch.isnan(final_slots).any():
            pdb.set_trace()

        return final_slots, final_attn, final_dots
    

