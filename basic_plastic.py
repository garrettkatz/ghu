"""
Reverse input (rinp) on output (rout) with turing layer m
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

def basic_plastic_trial(num_episodes, save_file):

    # Configuration
    register_names = ["rinp","rout","m"]
    layer_sizes = {q: 32 for q in register_names}
    hidden_size = 32
    rho = .99
    plastic = ["rinp<m"]
    remove_pathways = ["rinp<rout", "m<rinp", "m<rout", "rout<m"]

    # Setup GHU
    num_symbols = 3
    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer(register_names, symbols)
    for p in remove_pathways: pathways.pop(p)
    associations = list(filter(lambda x: x[0] not in remove_pathways, associations))
    codec = Codec(layer_sizes, symbols, rho=rho, ortho=True)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic, nonlinearity='relu')
    # controller = Controller(layer_sizes, pathways, hidden_size, plastic, nonlinearity='tanh')
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec,
        batch_size = num_episodes, plastic = plastic)
    ghu.associate(associations)

    # Initialize layers
    separator = "0"
    ghu.fill_layers(separator)

    # training example generation
    episode_duration = 3
    def training_example():
        inputs = np.random.choice(symbols[1:], size=2, replace=False)
        targets = np.array(["0","0",inputs[0]])
        return inputs, targets
    
    def reward(ghu, targets, outputs):        
        # All or nothing
        r = np.zeros(len(outputs))
        if lvd(outputs, targets)[0] == 0: r[-1] = +1.        
        return r

    # ################### Sanity check
    inputs = [["2", "1"]]
    correct_choices = [
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "m":"m<m"}, [1.0]),
        ({"rinp": "rinp<m",    "rout": "rout<rout", "m":"m<m"}, [0.0]),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "m":"m<m"}, [0.0]),
    ]
    # ghu.clone().dbg_run(inputs, episode_duration, correct_choices)
    # input("???????")
    # ################### Sanity check
            
    # Run optimization
    avg_rewards, grad_norms = reinforce(ghu,
        num_epochs = 500,
        episode_duration = episode_duration,
        training_example = training_example,
        reward = reward,
        task = "basic_plastic",
        learning_rate = .05,
        # line_search_iterations = 5,
        # distribution_cap = .1,
        # likelihood_cap = .7,
        # distribution_variance_coefficient = 0.05,
        # choices=correct_choices, # perfect rewards with this
        verbose = 1,
        save_file = save_file)

if __name__ == "__main__":
    print("*******************************************************")
    
    num_reps = 1
    num_episodes = 500
    save_base = "results/basic_plastic/run_%d_%d.pkl"
    
    # Run the experiment
    for rep in range(num_reps):
        save_file = save_base % (num_episodes, rep)
        basic_plastic_trial(num_episodes, save_file)
    
    # Load results
    results = {}
    for rep in range(num_reps):
        save_file = save_base % (num_episodes, rep)
        with open(save_file,"rb") as f:
            results[rep] = pk.load(f)
    
    # Plot results
    pt.figure(figsize=(4.25,2.25))
    bg = (.9,.9,.9) # background color
    avg_rewards = np.array([results[rep][1] for rep in results.keys()]).T

    pt.plot(avg_rewards, c=bg, zorder=0)
    fg = tuple([.1]*3) # foreground color
    pt.plot(avg_rewards.mean(axis=1), c=fg, zorder=1, label=("avg of %d reps" % num_reps))

    # pt.title("Learning curves")
    pt.ylabel("Average Reward")
    pt.xlabel("Epoch")
    pt.ylim([-1,1])
    pt.legend(loc="lower center")
    pt.tight_layout()
    # pt.savefig('big_reverse_learning_curves.eps')
    pt.show()
    


