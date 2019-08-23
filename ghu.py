"""
Activation rule for layer q:
    v[q][t+1] = tanh(sum_(p,q,r) s[p]*W[p].dot(v[r][t])
Learning rule for pathway p to q from r:
    W[p] += l[p] * (arctanh(v[q][t+1]) - W[p].dot(v[r][t])) * v[r][t].T / N
"""
import numpy as np
import torch as tr
import torch.nn as nn
from codec import Codec
from controller import Controller

class GatedHebbianUnit(object):
    def __init__(self, layer_sizes, pathways, controller, codec, plastic=[]):
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
        self.plastic = plastic
        self.W = {0:
            {p: tr.zeros(layer_sizes[q], layer_sizes[r])
                for p, (q,r) in pathways.items()}}
        self.v = {t:
            {q: tr.zeros(size)
                for q, size in layer_sizes.items()}
            for t in [-1, 0]}
        self.h = {-1: tr.zeros(1,1,controller.hidden_size)}
        self.ag = {}
        self.pg = {}

    def clone(self):
        # copies initial activity and weight matrices at time 0
        ghu = GatedHebbianUnit(
            layer_sizes = self.layer_sizes,
            pathways = self.pathways,
            controller = self.controller,
            codec = self.codec,
            plastic = self.plastic)
        ghu.v = {t:
            {q: self.v[t][q].clone().detach() for q in self.layer_sizes.keys()}
            for t in [-1, 0]}
        ghu.W = {0: {p: self.W[0][p].clone().detach() for p in self.pathways.keys()}}
        return ghu

    def rehebbian(self, W, x, y):
        r = self.codec.rho
        n = x.nelement() * r**2
        g = 0.5*(np.log(1. + r) - np.log(1. - r)) / r # formula for arctanh(r)
        dW = tr.ger(g*y - tr.mv(W, x), x) / n # ger is outer product
        return dW

    def tick(self, num_steps=1, detach=True, overrides=[None]):
        # overrides passed to controller
        # detach gates so that they are treated as actions on environment?

        T = len(self.ag)
        for t in range(T, T+num_steps):

            # Controller (activity gate, plasticity gate, hidden vector)
            self.ag[t], self.pg[t], self.h[t] = self.controller(
                self.v[t], self.h[t-1], override=overrides[t-T])
    
            # Associative learning
            self.W[t+1] = dict(self.W[t])
            # for p, (q,r) in self.pathways.items():
            #     if p not in self.plastic: continue
            for p in self.plastic:
                q, r = self.pathways[p]
                dW = self.rehebbian(self.W[t][p], self.v[t-1][r], self.v[t][q])
                _, a, _ = self.pg[t][p]
                if detach: a = a.detach()
                self.W[t+1][p] = self.W[t][p] + a * dW

            # Associative recall
            self.v[t+1] = {}
            for q in self.layer_sizes.keys():
                _, p, _ = self.ag[t][q]
                _, r = self.pathways[p]
                self.v[t+1][q] = tr.tanh(tr.mv(self.W[t][p], self.v[t][r]))

    def associate(self, associations):
        T = len(self.W)-1
        for p, s, t in associations:
            q, r = self.pathways[p]
            x = self.codec.encode(r, s)
            y = self.codec.encode(q, t)
            dW = self.rehebbian(self.W[T][p], x, y)
            self.W[T][p] = self.W[T][p] + dW
    
    def saturation(self):
        # return [min(float(prob), float(1. - prob))
        return [float(prob)
            for g in [self.ag, self.pg]
                for gates in g.values()
                    for (_, _, prob) in gates.values()]
        # s = []
        # for t, g in self.ag.items():
        #     for _, (_, _, prob) in g.items():
        #         s.append(min(float(prob), float(1. - prob)))
        # for t, g in self.pg.items():
        #     for p,(_, _, prob) in g.items():
        #         s.append(min(float(prob), float(1. - prob)))
        # return s

def default_initializer(register_names, symbols):
    pathways = {
        q+"<"+r: (q,r)
        for q in register_names
        for r in register_names}
    associations = [(p,a,a)
        for p in list(pathways.keys())
        for a in symbols]
    return pathways, associations

def turing_initializer(register_names, num_addresses):

    # defaults
    symbols = list(map(str, range(num_addresses)))
    pathways, associations = default_initializer(["m"] + register_names, symbols)

    # tape shifts
    pathways.update({k: ("m","m") for k in ["inc-m","dec-m"]})
    associations += [
        (k, str(a), str((a+x) % num_addresses))
        for k,x in [("inc-m",1), ("dec-m",-1)]
        for a in range(num_addresses)]

    return pathways, associations

if __name__ == "__main__":
    
    
    layer_sizes = {"r0": 3, "r1":3}
    pathways = {0:("r0","r0"), 1:("r1","r0"), 2:("r1","r1")}
    hidden_size = 5

    codec = Codec(layer_sizes, "01")
    controller = Controller(layer_sizes, pathways, hidden_size)
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec)

    ghu.associate([
        (0, "0", "0"),
        (1, "0", "0"),
        (2, "0", "0"),
        (0, "1", "1"),
        (1, "1", "1"),
        (2, "1", "1"),
        ])
    
    a = codec.encode("r0","0")
    b = codec.encode("r0","1")
    print(a)
    x = codec.decode("r0",a)
    print(x == "0")
    print(x == "1")
    y = codec.decode("r0",b)
    print(y == "0")
    print(y == "1")

    print(codec.parameters())
    
    ghu.v[-1]["r0"] = codec.encode("r0", str(1))
    ghu.v[-1]["r1"] = codec.encode("r1", str(1))
    ghu.v[0]["r0"] = codec.encode("r0", str(0))
    ghu.v[0]["r1"] = codec.encode("r1", str(0))
    ghu.tick(num_steps=2)

    e = ghu.g[len(ghu.g)-1].sum()
    e.backward()
    
    print("codec")
    print(codec)
    for p in codec.parameters():
        print(p.grad)
    print("ctrl")
    print(controller)
    for p in controller.parameters():
        print(p.grad)

    pathways, associations = turing_initializer(["r0","r1"], 3)
    for p in pathways:
        print(p, pathways[p])
        for p1,a,b in associations:
            if p1 == p:
                print(" %s -> %s"% (a,b))
