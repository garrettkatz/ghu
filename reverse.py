"""
Reverse input (rinp) on output (rout) with turing layer m
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
    
    verbose = 1
    
    num_addresses = 4
    # register_names = ["rinp","rout","r0"]
    register_names = ["rinp","rout"]
    # layer_sizes = {"rinp": 64, "rout":64, "r0": 64, "r1": 64}
    # layer_sizes = {"rinp": 256, "rout": 256, "m": 256}
    layer_sizes = {q: 64 for q in register_names+["m"]}
    hidden_size = 3
    # plastic = ["%s<m"%q for q in register_names]
    plastic = ["rinp<m"]

    symbols = [str(a) for a in range(num_addresses)]
    pathways, associations = turing_initializer(
        register_names, num_addresses)

    # constrain pathways inductive bias
    remove_pathways = ["rout<m", "m<rout"]
    for p in remove_pathways: pathways.pop(p)
    associations = list(filter(lambda x: x[0] not in remove_pathways, associations))
    # print(pathways)
    # print(associations)

    codec = Codec(layer_sizes, symbols, rho=.999)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic)
    ghu.associate(associations)
    for p, s, t in associations:
        q, r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[0][p], codec.encode(r, s))) == t)
    ghu_init = ghu

    # Initialize layers
    separator = symbols[0]
    for k in layer_sizes.keys():
        ghu_init.v[0][k] = codec.encode(k, separator)
  
    # training example generation
    list_symbols = 4
    min_length = 3
    max_length = 3
    def training_example():
        list_length = np.random.randint(min_length, max_length+1)
        inputs = np.array([separator]*(list_length+1))
        inputs[:-1] = np.random.choice(symbols[1:list_symbols], size=list_length, replace=False)
        targets = inputs[:-1][::-1]
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
    
    # Optimization
    avg_rewards, grad_norms = reinforce(
        ghu_init,
        num_epochs = 100,
        num_episodes = 10000,
        episode_duration = 2*max_length+1,
        training_example = training_example,
        reward = reward,
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


