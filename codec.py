import numpy as np
import torch as tr
import torch.nn as nn
from orthogonal_patterns import *

def getsize(size):
    n = nearest_valid_hadamard_size(size)
    return n #if n%2==0 else (n+1)

class Codec(nn.Module):
    def __init__(self, layer_sizes, symbols, rho = .999, requires_grad=False,ortho=False):
        self.rho = rho
        if ortho:
            
            self.encoder = {}
            n = max(getsize(len(symbols)),32)
            n = getsize(max(len(symbols),32))
            for k,size in layer_sizes.items():
                mat = random_orthogonal_patterns(n,len(symbols))
                temp = { k: {symbols[s]: tr.tensor(rho * mat[:,s].astype(np.float32),requires_grad=requires_grad)for s in range(len(symbols))}}
                self.encoder.update(temp)
        else:
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

    def show(self):
        print(self.encoder)
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


if __name__=="__main__":
    # a = tr.tensor(0.999 * np.sign(np.random.randn(8)).astype(np.float32),requires_grad=False)
    # print(a)
    # b = tr.tensor(0.999 * random_orthogonal_patterns(5,4).astype(np.float32),requires_grad=False)
    # print(b)
    # num_symbols = 8
    # layer_sizes = {"rinp": 9, "rout":9, "rtemp":9}
    
    # symbols = [str(a) for a in range(num_symbols+1)]
    # codec = Codec(layer_sizes, symbols, rho=.9999)
    # codec.show()
    a = nearest_valid_hadamard_size(9)
    print("SSSS",a)


    # mat = random_orthogonal_patterns(len(symbols),len(symbols))
    #         self.encoder = { k: {
    #             symbols[s]: tr.tensor(
    #                 rho * mat[:,s].astype(np.float32),
    #                 requires_grad=requires_grad)
    #             for s in range(len(symbols))}
    #             for k, size in layer_sizes.items()}
    
    # self.encoder = {}
    # for k,size in layer_sizes.items():
    #     mat = random_orthogonal_patterns(len(symbols),len(symbols))
    #     temp = { k: {
    #             symbols[s]: tr.tensor(
    #                 rho * mat[:,s].astype(np.float32),
    #                 requires_grad=requires_grad)
    #             for s in range(len(symbols))}}
    #     self.encoder.update(temp)



