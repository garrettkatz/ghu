import torch as tr

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


