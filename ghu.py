"""
Activation rule for layer q:
    v[q][t+1] = tanh(sum_(p,q,r) s[p]*W[p].dot(v[r][t])
Learning rule for pathway p to q from r:
    W[p] += l[p] * (arctanh(v[q][t+1]) - W[p].dot(v[r][t])) * v[r][t].T / N
"""

import torch as tr
import torch.nn as nn

class Codec(object):
    def __init__(self, layer_sizes, symbols, rho = .999):
        self.rho = rho
        self.lookup = {
            k: {
                s: (rho * tr.sign(tr.randn(size))).requires_grad_()
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
            for p, (q,r) in pathways.items()}
        self.v = {
            q: tr.zeros(size)
            for q, size in layer_sizes.items()}
        self.v_old = {
            q: tr.zeros(size)
            for q, size in layer_sizes.items()}

    def rehebbian(self, W, x, y):
        r = self.codec.rho
        N = x.nelement()
        # ay = 0.5*(tr.log(1 + y) - tr.log(1 - y))
        # dW = tr.ger(ay - tr.mv(W, x), x) / (N * r**2)
        dW = tr.ger(y - tr.mv(W, x), x) / (N * r**2)
        return dW

    def tick(self, stochastic=True):
        # Extract gate values
        self.controller.tick(self.v)
        s, l = self.controller.gates(stochastic)

        # print('tick')
        # print(self.v_old)
        # print(self.v)
        # print(s, l)
        # for p, (q,r) in self.pathways.items():
        #     print((q,r), self.W[p].detach().numpy().max(), self.W[p].detach().numpy().min())
        
        # Compute learning rule
        W = dict(self.W)
        for p, (q,r) in self.pathways.items():
            continue
            dW = self.rehebbian(self.W[p], self.v_old[r], self.v[q])
            W[p] = W[p] + l[p] * dW
        
        # Compute activation rule
        v = {
            q: tr.zeros(size)
            for q, size in self.layer_sizes.items()}
        for p, (q, r) in self.pathways.items():
            v[q] = v[q] + s[p] * tr.mv(self.W[p], self.v[r])
        v = {q: tr.tanh(v[q]) for q in v}
        
        # Update network
        self.v_old = self.v
        self.v = v
        self.W = W

    def associate(self, associations):
        for p, s, t in associations:
            q, r = self.pathways[p]
            x = self.codec.encode(r, s)
            y = self.codec.encode(q, t)
            dW = self.rehebbian(self.W[p], x, y)
            self.W[p] = self.W[p] + dW            

class DefaultController(nn.Module):
    """
    s, l: dicts = controller(v: dict)
    MLP with one recurrent hidden layer
    """
    def __init__(self, layer_sizes, pathways, hidden_size):
        super(DefaultController, self).__init__()
        self.input_keys = layer_sizes.keys()
        self.pathway_keys = pathways.keys()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(sum(layer_sizes.values()), hidden_size)
        self.readout = nn.Linear(hidden_size, 2*len(pathways))
        self.reset()
    
    def reset(self):
        self.h = tr.zeros(1,1,self.hidden_size)
        self.g = tr.zeros(2*len(self.pathway_keys))

    def tick(self, v):
        _, self.h = self.rnn(
            tr.cat([v[k] for k in self.input_keys]).view(1,1,-1),
            self.h)
        self.g = self.readout(self.h).squeeze()
        self.p = tr.sigmoid(self.g)
        self.a = (tr.rand_like(self.p) < self.p).detach().float()

    def gates(self, stochastic=True):
        pi = self.a if stochastic else self.p
        s, l = {}, {}
        for i, p in enumerate(self.pathway_keys):
            s[p], l[p] = pi[i], pi[len(self.pathway_keys)+i]
        return s, l

def default_initializer(register_names, symbols):
    pathways = {
        q+"<"+r: (q,r)
        for q in register_names
        for r in register_names}
    associations = [(p,a,a)
        for p in pathways.keys()
        for a in symbols]
    return pathways, associations

def turing_initializer(register_names, num_addresses):

    # defaults
    symbols = map(str, range(num_addresses))
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

    c = Codec(layer_sizes, "01")
    dc = DefaultController(layer_sizes, pathways, hidden_size)
    ghu = GatedHebbianUnit(layer_sizes, pathways, dc, c)

    ghu.associate([
        (0, "0", "0"),
        (1, "0", "0"),
        (2, "0", "0"),
        (0, "1", "1"),
        (1, "1", "1"),
        (2, "1", "1"),
        ])
    
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
    
    ghu.v_old["r0"] = c.encode("r0", str(1))
    ghu.v_old["r1"] = c.encode("r1", str(1))
    ghu.v["r0"] = c.encode("r0", str(0))
    ghu.v["r1"] = c.encode("r1", str(0))
    g_history = []
    for t in range(2):
        # ghu.v["r0"] = c.encode("r0", str(t % 2))
        # ghu.v["r1"] = c.encode("r1", str(t % 2))
        ghu.tick()
        g_history.append(ghu.controller.g)

    # ghu.tick()
    # g_history.append(ghu.controller.g)
    # ghu.tick()
    # g_history.append(ghu.controller.g)

    # e = ghu.v["r0"].sum() + ghu.v["r1"].sum()
    e = tr.cat(g_history).sum()
    print(e)
    e.backward()
    print("codec")
    for p in c.parameters():
        print(p.grad)
    print("cntrl")
    for p in ghu.parameters():
        print(p.grad)
    print(ghu)

    pathways, associations = turing_initializer(["r0","r1"], 3)
    for p in pathways:
        print(p, pathways[p])
        for p1,a,b in associations:
            if p1 == p:
                print(" %s -> %s"% (a,b))
