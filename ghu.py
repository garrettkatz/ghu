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
    def __init__(self, layer_sizes, pathways, controller, codec, batch_size=1, plastic=[]):
        """
        layer_sizes[k] (dict): size of layer k
        pathways[p]: (destination, source) for pathway p
        controller: dict of layer activity -> s,l gate dicts
        batch_size: number of examples to process at a time
        """
        super(GatedHebbianUnit, self).__init__()
        self.layer_sizes = layer_sizes
        self.pathways = pathways
        self.controller = controller
        self.codec = codec
        self.batch_size = batch_size
        self.plastic = plastic
        self.WL = {p: tr.zeros(batch_size, layer_sizes[q], 1)
            for p, (q,r) in pathways.items()}
        self.WR = {p: tr.zeros(batch_size, 1, layer_sizes[r])
            for p, (q,r) in pathways.items()}
        self.v = {t:
            {q: tr.zeros(batch_size, size)
                for q, size in layer_sizes.items()}
            for t in [-1, 0]}
        self.h = {-1: tr.zeros(1, batch_size, controller.hidden_size)}
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
            batch_size = self.batch_size,
            plastic = self.plastic)
        ghu.v = {t:
            {q: self.v[t][q].clone().detach() for q in self.layer_sizes.keys()}
            for t in [-1, 0]}
        ghu.WL = {p: self.WL[p].clone().detach() for p in self.pathways.keys()}
        ghu.WR = {p: self.WR[p].clone().detach() for p in self.pathways.keys()}
        return ghu

    def rehebbian(self, WL, WR, x, y):
        r = self.codec.rho
        n = x.shape[-1] * r**2
        g = 0.5*(np.log(1. + r) - np.log(1. - r)) / r # formula for arctanh(r)
        dWL = g*y.unsqueeze(2) - tr.matmul(WL, tr.matmul(WR, x.unsqueeze(2)))
        dWR = x.unsqueeze(1) / n
        # WL, WR = tr.cat((WL, dWL), dim=2), tr.cat((WR, dWR), dim=1)
        # return WL, WR
        return dWL, dWR

    def tick(self, detach=True, choices=None, verbose=0):
        # if provided, choices get passed to controller

        # Controller
        if verbose > 0: print("  Controller forward pass...")
        t = len(self.al)
        _, choices, likelihoods, self.h[t] = self.controller.act(
            self.v[t] if not detach else
                {q: v.clone().detach() for q, v in self.v[t].items()},
            self.h[t-1], choices)
        self.ac[t], self.pc[t] = choices
        self.al[t], self.pl[t] = likelihoods

        # Associative recall
        if verbose > 0: print("  Associative recall...")
        self.v[t+1] = {}
        WL, WR, rv = {}, {}, {} # accumulate data from different gate choices in different episodes
        if verbose > 1: print("   Selecting pathways for batch matmul...")
        for q, qsz in self.layer_sizes.items():
            # Get largest number of updates to an incoming pathway to layer q
            qt = max([self.WL[p].shape[-1] for p in self.controller.incoming[q]])
            # Get largest source layer size in the pathways to layer q
            rsz = max([self.layer_sizes[self.pathways[p][1]] for p in self.controller.incoming[q]])
            # Init sufficiently large pathway and source layer tensors
            WL[q] = tr.zeros(self.batch_size, qsz, qt)
            WR[q] = tr.zeros(self.batch_size, qt, rsz)
            rv[q] = tr.zeros(self.batch_size, rsz)
            # Process all possible incoming pathways
            for i, p in enumerate(self.controller.incoming[q]):
                # Get source layer and number of weight updates to current pathway
                _, r = self.pathways[p] # source layer
                pt = self.WL[p].shape[-1] # number of updates
                # Get mask of episodes where current pathway was ungated
                b = (self.ac[t][q][0] == i).squeeze()
                # Store weights and source activity from respective episodes
                WL[q][b, :, :pt], WR[q][b, :pt, :] = self.WL[p][b], self.WR[p][b]
                rv[q][b, :self.layer_sizes[r]] = self.v[t][r][b]
        if verbose > 1: print("   Performing matmul...")
        for q in self.layer_sizes.keys():
            self.v[t+1][q] = tr.tanh(
                tr.matmul(WL[q], tr.matmul(WR[q], rv[q].unsqueeze(2)))).squeeze(2)

        # Associative learning
        if verbose > 0: print("  Associative learning...")
        for p in self.pathways.keys():
            q, r = self.pathways[p]
            dWL = tr.zeros(self.batch_size, self.layer_sizes[q], 1)
            dWR = tr.zeros(self.batch_size, 1, self.layer_sizes[r])
            # Select out batch elements where pathway p learns
            if p in self.plastic:
                i = self.plastic.index(p)
                b = (self.pc[t][0,:,i] == 1)
                if b.sum() > 0:
                    dWL[b], dWR[b] = self.rehebbian(
                        self.WL[p][b], self.WR[p][b], self.v[t-1][r][b], self.v[t][q][b])
            self.WL[p] = tr.cat((self.WL[p], dWL), dim=2)
            self.WR[p] = tr.cat((self.WR[p], dWR), dim=1)

    def associate(self, associations, check=True):
        T = len(self.WL)-1
        for p, s, t in associations:
            q, r = self.pathways[p]
            x = tr.repeat_interleave(
                self.codec.encode(r, s).view(1,-1),
                self.batch_size, dim=0)
            y = tr.repeat_interleave(
                self.codec.encode(q, t).view(1,-1),
                self.batch_size, dim=0)
            dWL, dWR = self.rehebbian(self.WL[p], self.WR[p], x, y)
            self.WL[p] = tr.cat((self.WL[p], dWL), dim=2)
            self.WR[p] = tr.cat((self.WR[p], dWR), dim=1)

        if not check: return
        for p,s,t in associations:
            q,r = self.pathways[p]
            Wv = tr.matmul(self.WL[p][0], tr.matmul(self.WR[p][0], self.codec.encode(r, s).view(-1,1)))
            assert(self.codec.decode(q, Wv.squeeze()) == t)

    def fill_layers(self, symbol):
        for t in self.v:
            for k in self.layer_sizes.keys():
                self.v[t][k] = tr.repeat_interleave(
                    self.codec.encode(k, symbol).view(1,-1),
                    self.batch_size, dim=0)
    
    def dbg_run(self, inputs, episode_duration, choices):
        choices = [(
            {q: tr.stack([
                tr.tensor([self.controller.incoming[q].index(p)])
                for b in range(self.batch_size)]).reshape(1,self.batch_size,1) for q,p in ac.items()},
            tr.stack([tr.tensor(pc) for b in range(self.batch_size)]).reshape(1,self.batch_size,1))
            for (ac, pc) in choices]
        encoded = {}
        for t in range(episode_duration):
            if t < len(inputs[0]):
                encoded[t] = tr.stack([
                    self.codec.encode("rinp", inputs[0][t])
                    for b in range(self.batch_size)])
        for t in range(episode_duration):
            print(" t=%d..." % t)
            if t < len(inputs[0]): self.v[t]["rinp"] = encoded[t]
            self.tick(choices=choices[t], verbose=0) # Take a step
        for t in range(episode_duration+1):
            print(" Step %d" % t)
            print(" layers: ",
                {k: self.codec.decode(k, self.v[t][k][0]) for k in self.layer_sizes.keys()})
            if t == episode_duration: break
            print(" choices: ",
                {q: self.controller.incoming[q][ac[0,0].item()] for q,ac in self.ac[t].items()},
                [pc[0].item() for pc in self.pc[t] if len(self.plastic) > 0])
            print(" likelihoods: ",
                {q: "%.3f" % al[0,0].item() for q,al in self.al[t].items()},
                ["%.3f" % pl[0].item() for pl in self.pl[t] if len(self.plastic) > 0])

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
    batch_size = 2

    codec = Codec(layer_sizes, "01")
    controller = Controller(layer_sizes, pathways, hidden_size)
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, batch_size=batch_size)

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
                batch_size, dim=0)
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
