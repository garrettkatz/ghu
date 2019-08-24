import torch as tr
from ghu import *
from codec import Codec
from controller import Controller

def gradtree(x, prefix=''):
    if hasattr(x, 'grad_fn'):
        s = prefix + str(x.grad_fn) + ":\n"
        for f in x.grad_fn.next_functions:
            s += gradtree(f[0], prefix + ' ')
    else:
        s = prefix + str(x) + "\n"

    return s

if __name__ == "__main__":
    # GHU settings
    num_symbols = 3
    layer_sizes = {"rinp": 64, "rout":64}
    hidden_size = 16
    plastic = []

    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    codec = Codec(layer_sizes, symbols, rho=.9)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec, plastic=plastic)
    ghu.associate(associations)
    for p,s,t in associations:
        q,r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[p], codec.encode(r, s))) == t)

    # Initialize layers
    ghu.v[0]["rinp"] = codec.encode("rinp", "0")
    ghu.v[0]["rout"] = codec.encode("rout", "1")

    # Run GHU
    ghu.tick() # Take a step
    ghu.tick() # Take a step
    print(ghu.ag)
    
    print(gradtree(ghu.ag[1]["rout"][2]))

    
    
