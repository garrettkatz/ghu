"""
Swap input (rinp) on output (rout) with one extra registers (rtmp)
"""
import pickle as pk
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import reinforce

def swap_trial(distribution_variance_coefficient, save_file):

    # Configuration
    num_symbols = 4
    layer_sizes = {"rinp": 64, "rout":64, "rtmp": 64}
    hidden_size = 32
    rho = .99
    plastic = []
    num_episodes = 1000

    # Setup GHU
    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=rho)
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
        # Randomly choose swap symbols (excluding 0 separator)
        inputs = np.random.choice(symbols[1:], size=2, replace=False)
        targets = inputs[::-1]
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
        num_epochs = 100,
        episode_duration = 3,
        training_example = training_example,
        reward = reward,
        task = "swap",
        learning_rate = .2,
        # line_search_iterations = 5,
        # distribution_cap = .1,
        # likelihood_cap = .7,
        distribution_variance_coefficient = distribution_variance_coefficient,
        verbose = 1,
        save_file = save_file)

if __name__ == "__main__":
    print("*******************************************************")
    
    dvcs = [0., 0.001, 0.01, 0.1, 1.]
    num_reps = 30
    
    # Run the experiment
    for dvc in dvcs:
        for rep in range(num_reps):
            save_file = "results/swap/run_%f_%d.pkl" % (dvc, rep)
            swap_trial(dvc, save_file)

    # Load results
    results = {}
    for dvc in dvcs:
        results[dvc] = {}
        for rep in range(num_reps):
            save_file = "results/swap/run_%f_%d.pkl" % (dvc, rep)
            with open(save_file,"rb") as f:
                results[dvc][rep] = pk.load(f)
    
    # Plot results
    # config, avg_rewards, grad_norms = results[dvc][rep]
    avg_rewards = np.array([results[dvc][rep][1]
        for dvc in results.keys() for rep in results[dvc].keys()]).T
    grad_norms = np.array([results[dvc][rep][2]
        for dvc in results.keys() for rep in results[dvc].keys()]).T

    pt.figure(figsize=(4,3))
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curves")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.tight_layout()
    pt.show()
