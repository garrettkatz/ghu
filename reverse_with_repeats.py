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

def reverse_trial(num_episodes, save_file):

    # Configuration
    register_names = ["rinp","rout"]
    layer_sizes = {q: 64 for q in register_names + ["m"]}
    hidden_size = 64
    rho = .99
    plastic = ["rinp<m"]
    remove_pathways = ["rinp<rout", "m<rinp", "m<rout", "rout<m"]
    # input_keys = ["m"]
    input_keys = None

    # Setup GHU
    num_addresses = 4
    symbols = [str(a) for a in range(num_addresses)]
    pathways, associations = turing_initializer(
        register_names, num_addresses)
    for p in remove_pathways: pathways.pop(p)
    associations = list(filter(lambda x: x[0] not in remove_pathways, associations))
    codec = Codec(layer_sizes, symbols, rho=rho, ortho=True)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic,
        input_keys=input_keys, nonlinearity='relu')
    # controller = Controller(layer_sizes, pathways, hidden_size, plastic, nonlinearity='tanh')
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec,
        batch_size = num_episodes, plastic = plastic)
    ghu.associate(associations)

    # Initialize layers
    separator = "0"
    ghu.fill_layers(separator)

    # training example generation
    list_symbols = 5
    min_length = 4
    max_length = 4
    episode_duration = 2*max_length - 1
    def training_example():
        list_length = np.random.randint(min_length, max_length+1)
        # inputs = np.array(["0"]*(list_length+1))
        # # inputs[1:] = np.random.choice(symbols[1:list_symbols], size=list_length, replace=False)
        # inputs[1:] = np.random.choice(symbols[1:list_symbols], size=list_length, replace=True)
        inputs = np.random.choice(symbols[1:list_symbols], size=list_length, replace=True)
        targets = inputs[::-1]
        return inputs, targets
    
    # reward calculation based on leading LVD at individual steps
    def reward(ghu, targets, outputs):
        r = np.zeros(len(outputs))
        
        # All or nothing
        outputs_ = outputs[-len(targets):]
        if lvd(outputs_, targets)[0] == 0: r[-1] = +1.
        
        return r

    # ################### Sanity check
    correct_choices = [
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "m":"inc-m"}, [1.0]),
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "m":"inc-m"}, [1.0]),
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "m":"m<m"}, [1.0]),
        ({"rinp": "rinp<m", "rout": "rout<rinp", "m":"dec-m"}, [0.0]),
        ({"rinp": "rinp<m", "rout": "rout<rinp", "m":"dec-m"}, [0.0]),
        ({"rinp": "rinp<m", "rout": "rout<rinp", "m":"m<m"}, [0.0]),
        ({"rinp": "rinp<m", "rout": "rout<rinp", "m":"m<m"}, [0.0]),
        # ({"rinp": "rinp<m", "rout": "rout<rinp", "m":"m<m"}, [0.0]),
    ]
    # ################### Sanity check

    # Run optimization
    avg_rewards, grad_norms = reinforce(ghu,
        num_epochs = 250,
        episode_duration = episode_duration,
        training_example = training_example,
        reward = reward,
        task = "reverse",
        learning_rate = .1,
        # line_search_iterations = 5,
        # distribution_cap = .1,
        # likelihood_cap = .7,
        distribution_variance_coefficient = 0.05,
        # choices = correct_choices, # perfect reward with this
        verbose = 1,
        save_file = save_file)

if __name__ == "__main__":
    print("*******************************************************")
    
    num_reps = 20
    num_episodes = 8000
    
    # Run the experiment
    for rep in range(num_reps):
        save_file = "results/big_reverse/run_%d_%d.pkl" % (num_episodes, rep)
        reverse_trial(num_episodes, save_file)
    
    # Load results
    results = {}
    for rep in range(num_reps):
        save_file = "results/big_reverse/run_%d_%d.pkl" % (num_episodes, rep)
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
    

