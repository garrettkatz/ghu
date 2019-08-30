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
    def __init__(self, layer_sizes, pathways, controller, codec, batch=1, plastic=[]):
        """
        layer_sizes[k] (dict): size of layer k
        pathways[p]: (destination, source) for pathway p
        controller: dict of layer activity -> s,l gate dicts
        batch: number of examples to process at a time
        """
        super(GatedHebbianUnit, self).__init__()
        self.layer_sizes = layer_sizes
        self.pathways = pathways
        self.controller = controller
        self.codec = codec
        self.batch = batch
        self.plastic = plastic
        self.W = {p: tr.zeros(batch, layer_sizes[q], layer_sizes[r])
            for p, (q,r) in pathways.items()}
        self.v = {t:
            {q: tr.zeros(batch, size)
                for q, size in layer_sizes.items()}
            for t in [-1, 0]}
        self.h = {-1: tr.zeros(1, batch, controller.hidden_size)}
        self.al = {}
        self.pl = {}
        self.ac = {}
        self.pc = {}

    def clone(self):
        # copies associative weight matrices and initial activity
        # assumes no ticks have been called yet
        ghu = GatedHebbianUnit(
            layer_sizes = self.layer_sizes,
            pathways = self.pathways,
            controller = self.controller,
            codec = self.codec,
            plastic = self.plastic)
        ghu.v = {t:
            {q: self.v[t][q].clone().detach() for q in self.layer_sizes.keys()}
            for t in [-1, 0]}
        ghu.W = {p: self.W[p].clone().detach() for p in self.pathways.keys()}
        return ghu

    def rehebbian(self, W, x, y):
        r = self.codec.rho
        n = x.shape[-1] * r**2
        g = 0.5*(np.log(1. + r) - np.log(1. - r)) / r # formula for arctanh(r)
        dW = tr.matmul(
            g*y.unsqueeze(2) - tr.matmul(W, x.unsqueeze(2)),
            x.unsqueeze(1)) / n
        return dW

    def tick(self, num_steps=1, detach=True):
        # choices passed to controller

        T = len(self.al)
        for t in range(T, T+num_steps):

            # Controller
            _, choices, likelihoods, self.h[t] = self.controller.act(
                self.v[t] if not detach else
                    {q: v.clone().detach() for q, v in self.v[t].items()},
                self.h[t-1])    
            self.ac[t], self.pc[t] = choices
            self.al[t], self.pl[t] = likelihoods
    
            # Associative recall
            self.v[t+1] = {}
            for q in self.layer_sizes.keys():
            
                # Select out p,r for each batch element to prepare the batch matmul
                W_q, v_q = [], []
                for b,i in enumerate(self.ac[t][q][0]):
                    p = self.controller.incoming[q][i]
                    _, r = self.pathways[p]
                    W_q.append(self.W[p][b])
                    v_q.append(self.v[t][r][b,:])
                W_q, v_q = tr.stack(W_q), tr.stack(v_q)
                self.v[t+1][q] = tr.tanh(
                    tr.matmul(W_q, v_q.unsqueeze(2))).squeeze(2)

            # Associative learning
            for i,p in enumerate(self.plastic):
                a = self.pc[t][0,0,i]
                if a == 0: continue
                q, r = self.pathways[p]
                dW = self.rehebbian(self.W[p], self.v[t-1][r], self.v[t][q])
                self.W[p] = self.W[p] + dW

    def associate(self, associations):
        T = len(self.W)-1
        for p, s, t in associations:
            q, r = self.pathways[p]
            x = self.codec.encode(r, s).view(1,-1)
            y = self.codec.encode(q, t).view(1,-1)
            dW = self.rehebbian(self.W[p], x, y)
            self.W[p] = self.W[p] + dW
    
    def saturation(self):
        return tr.cat([l
            for t in self.al.keys() for l in
                [xl.view(-1) for xl in list(self.al[t].values()) + [self.pl[t]]]]).detach().numpy()


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
    batch = 2

    codec = Codec(layer_sizes, "01")
    controller = Controller(layer_sizes, pathways, hidden_size)
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, batch=batch)

    ghu.associate([
        (0, "0", "0"),
        (1, "0", "0"),
        (2, "0", "0"),
        (0, "1", "1"),
        (1, "1", "1"),
        (2, "1", "1"),
        ])
    
    # a = codec.encode("r0","0")
    # b = codec.encode("r0","1")
    # print(a)
    # x = codec.decode("r0",a)
    # print(x == "0")
    # print(x == "1")
    # y = codec.decode("r0",b)
    # print(y == "0")
    # print(y == "1")
    # print(codec.parameters())
    
    for t in [-1,0]:
        for k in ["r0","r1"]:
            ghu.v[t][k] = tr.repeat_interleave(
                codec.encode(k, str(0-t)).view(1,-1),
                batch, dim=0)
            # ghu.v[-1]["r0"] = codec.encode("r0", str(1))
            # ghu.v[-1]["r1"] = codec.encode("r1", str(1))
            # ghu.v[0]["r0"] = codec.encode("r0", str(0))
            # ghu.v[0]["r1"] = codec.encode("r1", str(0))
    ghu.tick(num_steps=2)

    # e = ghu.g[len(ghu.g)-1].sum()
    # e.backward()
    
    # print("codec")
    # print(codec)
    # for p in codec.parameters():
    #     print(p.grad)
    # print("ctrl")
    # print(controller)
    # for p in controller.parameters():
    #     print(p.grad)

    # pathways, associations = turing_initializer(["r0","r1"], 3)
    # for p in pathways:
    #     print(p, pathways[p])
    #     for p1,a,b in associations:
    #         if p1 == p:
    #             print(" %s -> %s"% (a,b))
