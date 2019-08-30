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
    
    num_addresses = 5
    # register_names = ["rinp","rout","r0"]
    register_names = ["rinp","rout","r0", "r1"]
    # layer_sizes = {"rinp": 64, "rout":64, "r0": 64, "r1": 64}
    # layer_sizes = {"rinp": 256, "rout": 256, "m": 256}
    layer_sizes = {q: 256 for q in register_names+["m"]}
    hidden_size = 32
    # plastic = ["%s<m"%q for q in register_names]
    plastic = ["rinp<m"]
    rho = .999
    num_episodes = 2000

    # Setup GHU
    print("Setting up GHU...")
    symbols = [str(a) for a in range(num_addresses)]
    pathways, associations = turing_initializer(
        register_names, num_addresses)
    # constrain pathways inductive bias
    remove_pathways = ["rout<m", "m<rout"]
    for p in remove_pathways: pathways.pop(p)
    associations = list(filter(lambda x: x[0] not in remove_pathways, associations))
    codec = Codec(layer_sizes, symbols, rho=rho)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec,
        batch_size = num_episodes, plastic = plastic)
    ghu.associate(associations)

    # Initialize layers
    separator = symbols[0]
    for k in layer_sizes.keys():
        ghu.v[0][k] = tr.repeat_interleave(
            codec.encode(k, separator).view(1,-1),
            num_episodes, dim=0)
  
    # training example generation
    list_symbols = 5
    min_length = 4
    max_length = 4
    def training_example():
        list_length = np.random.randint(min_length, max_length+1)
        inputs = np.array([separator]*(list_length))
        inputs[:] = np.random.choice(symbols[1:list_symbols], size=list_length, replace=False)
        targets = inputs[::-1]
        return inputs, targets
    
    # # reward calculation from LVD
    # def reward(ghu, targets, outputs):
    #     # Assess reward: negative LVD after separator filtering
    #     outputs_ = [out for out in outputs if out != separator]
    #     l, _ = lvd(outputs_, targets)
    #     return -l
    
    # reward calculation based on individual steps
    # def reward(ghu, targets, outputs):
    #     idx = [i for i, out in enumerate(outputs) if out != separator]
    #     outputs_ = [out for out in outputs if out != separator]
    #     _, d = lvd(outputs_, targets)
    #     r = np.zeros(len(outputs))
    #     for i in range(1,d.shape[0]):
    #         r[idx[i-1]] = +1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.

    #     # # Additional term to penalize time-steps where actions did not match
    #     # for t in range(1, len(outputs)):
    #     #     matches = []
    #     #     for q in ghu.layer_sizes.keys():
    #     #         matches.append(ghu.ac[t-1][q] == ghu.ac[t][q])
    #     #     for p in range(len(ghu.plastic)):
    #     #         matches.append(ghu.pc[t-1][p] == ghu.pc[t][p])
    #     #     r[t] -= (1. - float(sum(matches)) / len(matches)) / len(outputs)

    #     # Additional term to penalize empty outputs
    #     if len(outputs_) == 0: 
    #         r[-1] -= (len(outputs)+1)

    #     return r

    def reward(ghu, targets, outputs):
        outputs_ = outputs[len(targets)-1:]
        #zeros = [o for o in outputs[len(targets)-1:] if o==separator]
        #totzeros = len(zeros)
        r = np.zeros(len(outputs))
        # if len(outputs_)==0:
        #     r[-1] -= 2*(len(outputs[len(targets)-1:])+1)
        # else:
        _,d = lvd(outputs_,targets) 
        for i in range(1,d.shape[0]):
            r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -2.
            #r[-1] -= 0.1*totzeros
        return r
    
    # Run optimization
    avg_rewards, grad_norms = reinforce(ghu,
        num_epochs = 500,
        episode_duration = 2*max_length-1,
        training_example = training_example,
        reward = reward,
        task = "reverse",
        learning_rate = .2,
        verbose = 2)
    
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve for reverse")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.savefig("reverse.png")
    pt.show()


