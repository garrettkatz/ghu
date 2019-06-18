import torch as tr
import torch.nn as nn

class Controller(nn.Module):
    """
    s, l: dicts = controller(v: dict)
    MLP with one recurrent hidden layer
    """
    def __init__(self, layer_sizes, pathways, hidden_size):
        super(Controller, self).__init__()
        self.input_keys = layer_sizes.keys()
        self.pathway_keys = pathways.keys()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(sum(layer_sizes.values()), hidden_size)
        self.readout = nn.Linear(hidden_size, 2*len(pathways))

    def forward(self, v, h):
        _, h = self.rnn(
            tr.cat([v[k] for k in self.input_keys]).view(1,1,-1),
            h)
        g = tr.sigmoid(self.readout(h).squeeze())
        a = (tr.rand_like(g) < g).float()
        s, l = {}, {}
        for i, p in enumerate(self.pathway_keys):
            j = i + len(self.pathway_keys)
            s[p] = (g[i], a[i])
            l[p] = (g[j], a[j])
        return s, l, g, a, h

