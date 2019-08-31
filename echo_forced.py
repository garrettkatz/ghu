"""
Echo input (rinp) at output (rout)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce_brute import reinforce_brute

if __name__ == "__main__":
    print("*******************************************************")
    
    # Configuration
    num_symbols = 3
    layer_sizes = {"rinp": 64, "rout":64}
    hidden_size = 16
    rho = .99
    plastic = []
    episode_duration = 5

    # Set up all possible training examples
    symbols = [str(a) for a in range(num_symbols)]
    all_training_examples = [([s], [s]) for s in symbols[1:]]

    # Setup GHU
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=rho)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # number of possible action sequences:
    # ( 2**len(plastic) * product([len(incoming[q]) for q in layer_sizes] )**episode_duration
    num_acts = (
        2**len(plastic) * np.prod(list(map(len, controller.incoming.values())))
        ) ** episode_duration
    num_episodes = int(num_acts) * len(all_training_examples)

    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec,
        batch_size = num_episodes, plastic = plastic)
    ghu.associate(associations)

    # Initialize layers
    separator = "0"
    for k in layer_sizes.keys():
        ghu.v[0][k] = tr.repeat_interleave(
            codec.encode(k, separator).view(1,-1),
            num_episodes, dim=0)

    # reward calculation based on leading LVD at individual steps
    def reward(ghu, targets, outputs):
        idx = [i for i, out in enumerate(outputs) if out != separator]
        outputs_ = [out for out in outputs if out != separator]
        _, d = lvd(outputs_, targets)
        r = np.zeros(len(outputs))
        for i in range(1,d.shape[0]):
            r[idx[i-1]] = +1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        return r

    # Run optimization
    avg_rewards, grad_norms = reinforce_brute(ghu,
        num_epochs = 50,
        episode_duration = episode_duration,
        all_training_examples = all_training_examples,
        reward = reward,
        task = "echo",
        learning_rate = .1,
        verbose = 1)
    
    pt.figure(figsize=(4,3))
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.tight_layout()
    pt.show()
