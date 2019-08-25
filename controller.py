import torch as tr
import torch.nn as nn

"""
Helpers for sampling {c}hoices from {d}istributions and getting their {l}ikelihoods
pd[t,b,p] is a bernoulli distribution for plastic[p] at time t in batch b
pc and pl are corresponding choices/likelihoods of the same shape
ad[q][t,b,p] is a softmax distribution for activity in incoming[q][p] at time t in batch b
ac[q][t,b,0] and al[q][t,b,0] are the corresponding choices/likelihoods from the softmax
"""
def sample_choices(ad, pd):
    # sample {c}hoices from {d}istributions
    T, B = pd.shape[:2]
    ac = {q: tr.multinomial(ad[q].view(T*B, -1), 1).view(T, B, 1) for q in ad.keys()}
    pc = tr.bernoulli(pd)
    return ac, pc

def get_likelihoods(ac, pc, ad, pd):
    # {l}ikelihood of {c}hoices given {d}istributions
    al = {q: tr.gather(ad[q], 2, ac[q]) for q in ad.keys()}
    pl = tr.where(pc == 1, pd, 1 - pd)
    return al, pl

class Controller(nn.Module):
    """
    activity, plasticity: dicts = controller(v: dict)
    MLP with one recurrent hidden layer
    """
    def __init__(self, layer_sizes, pathways, hidden_size, plastic, input_keys=None):
        if input_keys is None: input_keys = layer_sizes.keys()
        super(Controller, self).__init__()
        self.layer_sizes = layer_sizes
        self.input_keys = input_keys
        self.plastic = plastic
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(sum([layer_sizes[q] for q in input_keys]), hidden_size)
        self.incoming = { # pathways organized by destination layer
            q: [p for p, (q_,r) in pathways.items() if q_ == q]
            for q in self.layer_sizes}
        self.activity_readouts = nn.ModuleDict({
            q: nn.Sequential(
                nn.Linear(hidden_size, len(self.incoming[q])),
                nn.Softmax(dim=-1))
            for q in self.layer_sizes})
        self.plasticity_readout = nn.Sequential(
            nn.Linear(hidden_size, len(plastic)),
            nn.Sigmoid())

    def forward(self, v, h_0):
        # recur, starting from time T
        # v[t,b,:] = concatenated input layers at time T+t in batch b
        # h_0[0,b,:] = hidden layer at time T-1 in batch b
        # h[t,b,:] = hidden layer at time T+t in batch b
        # ad[q][t,b,r] = Pr(choosing incoming r at time t in batch b)
        # pd[t,b,p] = Pr(learning in pathway p at time t in batch b)
        h, _ = self.rnn(v, h_0) 
        ad = {q: self.activity_readouts[q](h) for q in self.layer_sizes}
        pd = self.plasticity_readout(h)
        return ad, pd, h

    def act(self, v, h, choices=None):
        # v[q] = input layer q activity at current time
        # h = hidden layer activity at previous time
        # provide choices = ac, pc to override sampling
        ad, pd, h = self.forward(
            tr.cat([v[k] for k in self.input_keys]).view(1,1,-1),
            h.view(1,1,-1))
        distributions = ad, pd
        h = h.view(-1)
        
        if choices is None: choices = sample_choices(*distributions)
        likelihoods = get_likelihoods(*choices, *distributions)
        
        return distributions, choices, likelihoods, h

