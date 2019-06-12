"""
Activation rule for layer q:
    v[q][t+1] = tanh(sum_(p,q,r) s[p]*W[p].dot(v[r][t])
Learning rule for pathway p to q from r:
    W[p] += l[p] * (arctanh(v[q][t+1]) - W[p].dot(v[r][t])) * v[r][t].T / N
"""

import numpy as np
import torch as tr
import torch.nn as nn

class GatedHebbianUnit(nn.Module):
    def __init__(self, layer_sizes, pathways, controller):
        """
        layer_sizes[k] (dict): size of layer k
        pathways[p]: (destination, source) for pathway p
        controller: dict of layer activity -> s,l gate dicts
        """
        self.layer_sizes = layer_sizes
        self.pathways = pathways
        self.controller = controller
        self.activity = {
            name: tr.zeros(size)
            for name, size in layer_sizes.items()}

    def forward(self):
        # Extract gate values
        s, l = self.controller.forward(self.activity)
        
        # Do activation rule
        h = {
            q: tr.zeros(size)
            for q, size in self.layer_sizes.items()}
        for c, (q, r) in self.pathways.items():
            h[q] = h[q] + s[c] * tr.mm(self.W[c], v[r])
        v_new = {q: tr.tanh(h[q]) for q in h}
        
        # Do learning rule
        for c, (q,r) in self.pathways:
            N = self.layer_sizes[r]
            dW = (tr.arctanh(v_new[q]) - tr.mm(self.W[c], v[r])) * v[r].T / N
            self.W[c] = self.W[c] + l[c] * dW

        # Return new activity
        return v_new

class DefaultController:
    """
    l, s, d: dicts = controller(v: dict)
    MLP with one hidden layer
    """
    def __init__(self, layer_sizes, pathways, hidden_size):
        num_gates = len(layer_sizes)

        self.layer_sizes = layer_sizes
        self.hidden_size = hidden_size
        self.input_names = layer_sizes.keys()
        self.inputs = {
            name: nn.Linear(layer_size, hidden_size)
            for name, layer_size in layer_sizes.items()}
        self.output = nn.Linear(hidden_size, num_gates)

    def parameters(self):
        return self.output.parameters() + sum(
            [linear.parameters() for linear in self.inputs]

    def forward(self, v):
        h = tr.zeros(hidden_size)
        for name, linear in self.inputs.items():
            h = h + linear(v[name])
        gates = self.output(tr.tanh(h))

        L = len(self.input_names)
        l = {name: gates[i]
            for i, name in enumerate(self.input_names)}
        s = {name: gates[L**2 + i]
            for i, name in enumerate(self.input_names)}
        d = {name: gates[2*L**2 + i]
            for i, name in enumerate(self.input_names)}

        return l, s, d

if __name__ == "__main__":
    
    

