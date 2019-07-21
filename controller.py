import torch as tr
import torch.nn as nn
import numpy as np

class Controller(nn.Module):
    """
    activity, plasticity: dicts = controller(v: dict)
    MLP with one recurrent hidden layer
    """
    def __init__(self, layer_sizes, pathways, hidden_size, input_keys=None):
        if input_keys is None: input_keys = layer_sizes.keys()
        super(Controller, self).__init__()
        self.layer_sizes = layer_sizes
        self.input_keys = input_keys
        self.pathway_keys = pathways.keys()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(sum([layer_sizes[q] for q in input_keys]), hidden_size)
        self.incoming = {
            q: np.array([p for p, (q_,r) in pathways.items() if q_ == q])
            for q in self.layer_sizes}
        self.activity_readouts = nn.ModuleDict({
            q: nn.Sequential(
                nn.Linear(hidden_size, len(self.incoming[q])),
                nn.Softmax(dim=-1))
            for q in self.layer_sizes})
        self.plasticity_readout = nn.Sequential(
            nn.Linear(hidden_size, len(pathways)),
            nn.Sigmoid())

    def forward(self, v, h):
        _, h = self.rnn(
            tr.cat([v[k] for k in self.input_keys], dim=1).unsqueeze(0),
            h)
        activity, plasticity = {}, {}
        # activity
        for q in self.layer_sizes:
            gates = self.activity_readouts[q](h).squeeze(0)
            choice = tr.multinomial(gates, 1)
            action = self.incoming[q][choice].reshape(gates.shape[0]) # without reshape, action is scalar when batch_size=1
            prob = [gates[b,c] for (b,c) in enumerate(choice)]
            activity[q] = (gates, action, prob)
        # plasticity
        gates = self.plasticity_readout(h).squeeze(0)
        action = tr.bernoulli(gates)
        probs = tr.where(action == 0, 1 - gates, gates)
        for i, p in enumerate(self.pathway_keys):
            plasticity[p] = (gates[:,i], action[:,i], probs[:,i])
                
        return activity, plasticity, h

