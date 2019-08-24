"""
Swap input (rinp) on output (rout) with two registers (r0, r1)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import reinforce

if __name__ == "__main__":
    print("*******************************************************")
    
    num_symbols = 4
    # layer_sizes = {"rinp": 64, "rout":64, "r0": 64, "r1": 64}
    layer_sizes = {"rinp": 64, "rout":64, "r0": 64}
    hidden_size = 16
    plastic = []

    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer(
        layer_sizes.keys(), symbols)

    codec = Codec(layer_sizes, symbols, rho=.9)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic)
    ghu.associate(associations)
    for p,s,t in associations:
        q,r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[p], codec.encode(r, s))) == t)
    ghu_init = ghu

    # Initialize layers
    separator = "0"
    for k in layer_sizes.keys():
        ghu_init.v[0][k] = codec.encode(k, separator)

    # training example generation
    def training_example():
        # Randomly choose swap symbols (excluding 0 separator)
        inputs = np.random.choice(symbols[1:], size=2, replace=False)
        targets = inputs[::-1]
        return inputs, targets
    
    # # reward calculation from LVD
    # def reward(ghu, targets, outputs):
    #     # Assess reward: negative LVD after separator filtering
    #     outputs_ = [out for out in outputs if out != separator]
    #     l, _ = lvd(outputs_, targets)
    #     return -l
    
    # reward calculation based on individual steps
    def reward(ghu, targets, outputs):
        idx = [i for i, out in enumerate(outputs) if out != separator]
        outputs_ = [out for out in outputs if out != separator]
        _, d = lvd(outputs_, targets)
        r = np.zeros(len(outputs))
        for i in range(1,d.shape[0]):
            r[idx[i-1]] = +1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        return r
            
    # Optimization settings
    avg_rewards, grad_norms = reinforce(
        ghu_init,
        num_epochs = 200,
        num_episodes = 500,
        episode_duration = 3,
        training_example = training_example,
        reward = reward,
        task = "swap",
        learning_rate = .1)
    
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.show()



