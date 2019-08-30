import numpy as np
import torch as tr

class Codec(object):
    def __init__(self, layer_sizes, symbols, rho = .999, requires_grad=False):
        self.rho = rho
        self.encoder = { k: {
            s: tr.tensor(
                rho * np.sign(np.random.randn(size)).astype(np.float32),
                requires_grad=requires_grad)
            for s in symbols}
            for k, size in layer_sizes.items()}
        self.decoder = {k: {
            (p.data.numpy() > 0).tobytes(): s
            for s, p in self.encoder[k].items()}
            for k in layer_sizes.keys()}
    def encode(self, layer, symbol):
        return self.encoder[layer][symbol]
    def decode(self, layer, pattern):
        # for s, p in self.lookup[layer].items():
        #     if (p * pattern > 0).all(): return s
        return self.decoder[layer].get(
            (pattern.data.numpy() > 0).tobytes(), None)
    def parameters(self):
        for lk in self.encoder.values():
            for p in lk.values():
                yield p


