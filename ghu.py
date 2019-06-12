"""
Activation rule for layer q:
    v[q][t+1] = tanh(sum_(p,q,r) s[p]*W[p].dot(v[r][t])
Learning rule for pathway p to q from r:
    W[p] += l[p] * (arctanh(v[q][t+1]) - W[p].dot(v[r][t])) * v[r][t].T / N
"""

import numpy as np
import torch as tr
import torch.nn as nn

class Codec(object):
    def __init__(self, layer_sizes, symbols, rho = .9999):
        self.rho = rho
        self.lookup = {
            k: {
                s: rho * tr.tensor(np.sign(np.random.randn(size,1)))
                for s in symbols}
            for k, size in layer_sizes.items()}
    def encode(self, layer, symbol):
        return self.lookup[layer][symbol]
    def decode(self, layer, pattern):
        for s, p in self.lookup[layer].items():
            if (p * pattern > 0).all(): return s
    def parameters(self):
        for lk in self.lookup.values():
            for p in lk.values():
                yield p

class GatedHebbianUnit(nn.Module):
    def __init__(self, layer_sizes, pathways, controller, codec):
        """
        layer_sizes[k] (dict): size of layer k
        pathways[p]: (destination, source) for pathway p
        controller: dict of layer activity -> s,l gate dicts
        """
        super(GatedHebbianUnit, self).__init__()
        self.layer_sizes = layer_sizes
        self.pathways = pathways
        self.controller = controller
        self.codec = codec
        self.W = {p:
            tr.zeros(layer_sizes[q], layer_sizes[r])
            for (p, (q,r)) in pathways}
        self.v = {
            q: tr.zeros(size)
            for q, size in layer_sizes.items()}

    def rehebbian(W, x, y):
        r = self.codec.rho
        N = x.shape[0]
        return (tr.arctanh(y) - tr.mm(W, x)) * x.T / (N * r**2)

    def tick(self):
        # Extract gate values
        s, l = self.controller(self.v)
        
        # Do activation rule
        h = {
            q: tr.zeros(size)
            for q, size in self.layer_sizes.items()}
        for p, (q, r) in self.pathways.items():
            h[q] = h[q] + s[p] * tr.mm(self.W[p], v[r])
        v = {q: tr.tanh(h[q]) for q in h}
        
        # Do learning rule
        for p, (q,r) in self.pathways:
            dW = self.rehebbian(self.W[p], self.v[r], v[q])
            self.W[p] = self.W[p] + l[p] * dW

        # Update activity
        self.v = v

    def associate(self, associations):
        for p, s, t in associations:
            q, r = self.pathways[p]
            x = self.codec.encode(r, s)
            y = self.codec.encode(q, t)
            dW = self.rehebbian(self.W[p], x, y)
            self.W[p] = self.W[p] + dW
            

class DefaultController(nn.Module):
    """
    l, s, d: dicts = controller(v: dict)
    MLP with one hidden layer
    """
    def __init__(self, layer_sizes, pathways, hidden_size):
        super(DefaultController, self).__init__()
        num_gates = 2*len(pathways)
        self.pathways = pathways
        self.hidden_size = hidden_size
        self.inputs = nn.ModuleDict({
            k: nn.Linear(layer_size, hidden_size)
            for k, layer_size in layer_sizes.items()})
        self.output = nn.Linear(hidden_size, num_gates)

    def forward(self, v):
        h = tr.zeros(self.hidden_size)
        for q, linear in self.inputs.items():
            h = h + linear(v[q])
        gates = self.output(tr.tanh(h))

        s = {p: gates[i]
            for i, p in enumerate(self.pathways.keys())}
        l = {p: gates[len(self.pathways)+i]
            for i, p in enumerate(self.pathways.keys())}

        return s, l

if __name__ == "__main__":
    
    
    layer_sizes = {"r0": 10, "r1":20}
    pathways = [(0,("r0","r0")), (1, ("r1","r0"))]
    hidden_size = 5

    c = Codec(layer_sizes, "01")
    dc = DefaultController(layer_sizes, pathways, hidden_size)
    ghu = GatedHebbianUnit(layer_sizes, pathways, dc, c)

    
    a = c.encode("r0","0")
    b = c.encode("r0","1")
    print(a)
    x = c.decode("r0",a)
    print(x == "0")
    print(x == "1")
    y = c.decode("r0",b)
    print(y == "0")
    print(y == "1")

    print(c.parameters())

    
    
