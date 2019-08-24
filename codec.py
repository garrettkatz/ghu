import numpy as np
import torch as tr

class Codec(object):
    def __init__(self, layer_sizes, symbols, rho = .999, requires_grad=False):
        self.rho = rho
        self.lookup = {
            k: {
                s: tr.tensor(
                    rho * np.sign(np.random.randn(size)).astype(np.float32),
                    requires_grad=requires_grad)
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


