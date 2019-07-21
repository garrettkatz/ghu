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
    def __init__(self, layer_sizes, pathways, controller, codec, batch_size=1):
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
        self.W = {0:
            {p: tr.zeros(batch_size, layer_sizes[q], layer_sizes[r])
                for p, (q,r) in pathways.items()}}
        self.v = {t:
            {q: tr.zeros(batch_size, layer_size)
                for q, layer_size in layer_sizes.items()}
            for t in [-1, 0]}
        self.h = {-1: tr.zeros(1,batch_size,controller.hidden_size)}
        self.ag = {}
        self.pg = {}

    def rehebbian(self, W, x, y):
        r = self.codec.rho
        n = x.shape[1] * r**2
        g = 0.5*(np.log(1. + r) - np.log(1. - r)) / r # formula for arctanh(r)
        dW = tr.matmul(
            g*y.unsqueeze(2) - tr.matmul(W, x.unsqueeze(2)),
            x.unsqueeze(1)) / n
        return dW

    def tick(self, num_steps=1, detach=True, plastic=[]):
        # detach gates so that they are treated as actions on environment?

        T = len(self.ag)
        for t in range(T, T+num_steps):

            # Controller (activity gate, plasticity gate, hidden vector)
            self.ag[t], self.pg[t], self.h[t] = self.controller(self.v[t], self.h[t-1])
    
            # Associative learning
            self.W[t+1] = dict(self.W[t])
            for p, (q,r) in self.pathways.items():
                if p not in plastic: continue
                _, l, _ = self.pg[t][p]
                b = tr.nonzero(l)
                dW = self.rehebbian(self.W[t][p][b], self.v[t-1][r][b], self.v[t][q][b])
                self.W[t+1][p][b] = self.W[t][p][b] + dW

            # Associative recall
            self.v[t+1] = {}
            for q in self.layer_sizes.keys():
                _, a, _ = self.ag[t][q]
                Wv = []
                for b,p in enumerate(a):
                    _, r = self.pathways[p]
                    Wv.append(
                        tr.mv(self.W[t][p][b], self.v[t][r][b]))
                self.v[t+1][q] = tr.tanh(tr.stack(Wv))

    def associate(self, associations):
        T = len(self.W)-1
        for p, s, t in associations:
            q, r = self.pathways[p]
            x = self.codec.encode(r, s)
            y = self.codec.encode(q, t)
            dW = self.rehebbian(self.W[T][p], x.unsqueeze(0), y.unsqueeze(0))
            self.W[T][p] = self.W[T][p] + dW

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
    pathways = {0:("r0","r0"), 1:("r1","r0"), 2:("r1","r1"), 3:("r0","r1")}
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
    
    ghu.v[-1]["r0"] = codec.encode("r0", str(1)).unsqueeze(0)
    ghu.v[-1]["r1"] = codec.encode("r1", str(1)).unsqueeze(0)
    ghu.v[0]["r0"] = codec.encode("r0", str(0)).unsqueeze(0)
    ghu.v[0]["r1"] = codec.encode("r1", str(0)).unsqueeze(0)
    ghu.tick(num_steps=2)

    e = ghu.ag[len(ghu.ag)-1]["r1"][0].sum()
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

