import torch as tr
import torch.nn as nn

class Controller(nn.Module):
    """
    activity, plasticity: dicts = controller(v: dict)
    MLP with one recurrent hidden layer
    """
    def __init__(self, layer_sizes, pathways, hidden_size):
        super(Controller, self).__init__()
        self.input_keys = layer_sizes.keys()
        self.pathway_keys = pathways.keys()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(sum(layer_sizes.values()), hidden_size)
        # self.readout = nn.Linear(hidden_size, 2*len(pathways))
        self.incoming = {
            q: [p for p, (q_,r) in pathways.items() if q_ == q]
            for q in self.input_keys}
        self.activity_readouts = nn.ModuleDict({
            q: nn.Sequential(
                nn.Linear(hidden_size, len(self.incoming[q])),
                nn.Softmax(dim=-1))
            for q in self.input_keys})
        self.plasticity_readout = nn.Sequential(
            nn.Linear(hidden_size, len(pathways)),
            nn.Sigmoid())

    def forward(self, v, h):
        _, h = self.rnn(
            tr.cat([v[k] for k in self.input_keys]).view(1,1,-1),
            h)
        activity, plasticity = {}, {}
        # activity
        for q in self.input_keys:
            gates = self.activity_readouts[q](h).squeeze()
            choice = tr.multinomial(gates, 1)
            action = self.incoming[q][choice]
            prob = gates[choice]
            activity[q] = (gates, action, prob)
        # plasticity
        gates = self.plasticity_readout(h).squeeze()
        action = tr.bernoulli(gates)
        probs = tr.where(action == 0, 1 - gates, gates)
        for i, p in enumerate(self.pathway_keys):
            plasticity[p] = (gates[i], action[i], probs[i])
                
        return activity, plasticity, h

