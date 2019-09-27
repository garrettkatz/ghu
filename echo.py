"""
Echo input (rinp) at output (rout)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
import pickle as pk
from ghu import *
from codec import *
from controller import Controller
from lvd import lvd
from reinforce import reinforce


if __name__ == "__main__":
    print("*******************************************************")
    
    # Configuration
    num_symbols = 3
    #layer_sizes = {"rinp": 3, "rout":3}
    hidden_size = 16
    rho = .99
    plastic = ["rinp<rout"]
    num_episodes = 200

    # Setup GHU
    symbols = [str(a) for a in range(num_symbols)]
    length = max(getsize(len(symbols)),32)
    layer_sizes = {"rinp": length, "rout":length}
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=rho, requires_grad=False,ortho=True)
    #codec.show()
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)
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

    # training example generation
    def training_example():
        # Randomly choose echo symbol (excluding 0 separator)
        inputs = np.random.choice(symbols[1:], size=1)
        targets = inputs
        return inputs, targets
    
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
    avg_rewards, grad_norms = reinforce(ghu,
        num_epochs = 1,
        episode_duration = 5,
        training_example = training_example,
        reward = reward,
        task = "echo",
        learning_rate = .1,
        verbose = 1,
        distribution_variance_coefficient = 0.05,
        save_file = "tmp.pkl")
    
    with open("tmp.pkl","rb") as f:
        config, avg_rewards, grad_norms, dist_vars = pk.load(f)

    print(config)
    print(avg_rewards[-10:])
    print(grad_norms[-10:])
    print(dist_vars[-10:])
    
    pt.figure(figsize=(4,3))
    pt.subplot(3,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve")
    pt.ylabel("Avg Reward")
    pt.subplot(3,1,2)
    pt.plot(grad_norms)
    pt.ylabel("||Grad||")
    pt.subplot(3,1,3)
    pt.plot(dist_vars)
    pt.ylabel("Var(D)")
    pt.xlabel("Epoch")
    pt.tight_layout()
    pt.show()
