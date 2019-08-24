import torch as tr
import torch.nn as nn

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

    def forward(self, v, h, override=None):
        # override[q] = choice for q's activity gate
        # override["plasticity"] = plasticity gates
        _, h = self.rnn(
            tr.cat([v[k] for k in self.input_keys]).view(1,1,-1),
            h)
        activity, plasticity = {}, {}
        # activity
        for q in self.layer_sizes:
            gates = self.activity_readouts[q](h).flatten()
            choice = tr.multinomial(gates, 1) if override is None else tr.tensor([override[q]])
            action = self.incoming[q][choice]
            prob = gates[choice]
            activity[q] = (gates, action, prob)
        # plasticity
        gates = self.plasticity_readout(h).flatten()
        action = tr.bernoulli(gates) if override is None else override["plasticity"]
        probs = tr.where(action == 0, 1 - gates, gates)
        for i, p in enumerate(self.plastic):
            plasticity[p] = (gates[i], action[i], probs[[i]])
                
        return activity, plasticity, h

