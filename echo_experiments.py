"""
Echo input (rinp) at output (rout)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
import pickle as pk
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import reinforce

def echo_trial(episode_duration, save_file):

    # Configuration
    num_symbols = 3
    layer_sizes = {"rinp": 64, "rout":64}
    hidden_size = 16
    rho = .99
    plastic = []
    num_episodes = 200

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
        num_epochs = 100,
        episode_duration = episode_duration,
        training_example = training_example,
        reward = reward,
        task = "echo",
        learning_rate = .1,
        verbose = 1,
        save_file = save_file)
    
    return avg_rewards, grad_norms

if __name__ == "__main__":
    print("*******************************************************")

    durations = range(1,6)
    num_reps = 30
    
    # # Run the experiment
    # for dur in durations:
    #     for rep in range(num_reps):
    #         save_file = "results/echo/run_%d_%d.pkl" % (dur, rep)
    #         echo_trial(dur, save_file)

    # Load results
    results = {}
    for dur in durations:
        results[dur] = {}
        for rep in range(num_reps):
            save_file = "results/echo/run_%d_%d.pkl" % (dur, rep)
            with open(save_file,"rb") as f:
                results[dur][rep] = pk.load(f)
    
    # Plot results
    pt.figure(figsize=(4.25,2.25))
    bg = (.9,.9,.9) # background color
    for d,dur in enumerate(durations):
        avg_rewards = np.array([results[dur][rep][1]
            for rep in results[dur].keys()]).T

        pt.plot(avg_rewards, c=bg, zorder=0)
        fg = tuple([d/6]*3) # foreground color
        pt.plot(avg_rewards.mean(axis=1), c=fg, zorder=1, label=("T=%d" % dur))

    # pt.title("Learning curves")
    pt.ylabel("Average Reward")
    pt.xlabel("Epoch")
    pt.ylim([-1,1])
    pt.legend(loc="lower center")
    pt.tight_layout()
    pt.savefig('echo_learning_curves.eps')
    pt.show()
    
    # Histograms of final rewards
    pt.figure(figsize=(4.25,2))
    finals = []
    for d,dur in enumerate(durations):
        avg_rewards = np.array([results[dur][rep][1]
            for rep in results[dur].keys()]).T
        finals.append(avg_rewards[-1,:])
    pt.boxplot(finals, showfliers=False)

    # pt.title("Final Average Rewards")
    pt.ylabel("Reward")
    pt.xlabel("Episode duration")
    pt.tight_layout()
    pt.savefig('echo_finals.eps')
    pt.show()
    
