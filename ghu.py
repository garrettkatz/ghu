"""
Activation rule for layer q:
    v[q][t+1] = tanh(d[q]*v[q][t] + sum_(c,q,r) s[c]*W[c].dot(v[r][t])
Learning rule for connection c to q from r:
    W[c] += l[c] * (arctanh(v[q][t+1]) - W[c].dot(v[r][t])) * v[r][t].T / N
"""

import numpy as np
import torch as tr
import torch.nn as nn

class GHU:
    def __init__(self, layer_sizes, connections, controller):
        """
        l, s, d: dicts = controller(v: dict)
        """
        self.layer_sizes = layer_sizes
        self.connections = connections
        self.controller = controller
        self.old_activity = {
            name: tr.zeros(size)
            for name, size in layer_sizes.items()}

    def tick(self, v):
        # Extract gate values
        l, s, d = self.controller.forward(v)
        
        # Do activation rule
        h = {
            q: d[q] * v[q]
            for q, size in self.layer_sizes.items()}
        for c, (q, r) in self.connections.items():
            h[q] = h[q] + s[c] * tr.mm(self.W[c], v[r])
        v_new = {q: tr.tanh(h[q]) for q in h}
        
        # Do learning rule
        for c, (q,r) in self.connections:
            N = self.layer_sizes[r]
            dW = (tr.arctanh(v_new[q]) - tr.mm(self.W[c], v[r])) * v[r].T / N
            self.W[c] = self.W[c] + l[c] * dW

        # Return new activity
        return v_new

class DefaultController:
    """
    l, s, d: dicts = controller(v: dict)
    """
    def __init__(self, layer_sizes, hidden_size):
        L = len(layer_sizes)
        num_gates = 2*L**2 + L

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

