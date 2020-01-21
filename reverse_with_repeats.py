"""
Reverse input (rinp) on output (rout) with turing layer m
"""
import pickle as pk
import itertools as it
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
    num_addresses = 10
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

    # Dataset of all possible input lists
    list_symbols = 6
    min_length = 4
    max_length = 4
    episode_duration = 2*max_length - 1
    all_inputs = [np.array(inputs)
        for list_length in range(min_length, max_length+1)
            for inputs in it.product(symbols[1:list_symbols], repeat=list_length)]
    # input("%d..." % len(all_inputs))
    split = int(.80*len(all_inputs))

    # example generation
    def training_example():
        # list_length = np.random.randint(min_length, max_length+1)
        # # inputs = np.array(["0"]*(list_length+1))
        # # # inputs[1:] = np.random.choice(symbols[1:list_symbols], size=list_length, replace=False)
        # # inputs[1:] = np.random.choice(symbols[1:list_symbols], size=list_length, replace=True)
        # inputs = np.random.choice(symbols[1:list_symbols], size=list_length, replace=True)
        inputs = all_inputs[np.random.randint(split)]
        targets = inputs[::-1]
        return inputs, targets
    def testing_example():
        inputs = all_inputs[np.random.randint(split, len(all_inputs))]
        targets = inputs[::-1]
        return inputs, targets
    
    # reward calculation 
    def reward(ghu, targets, outputs):
        r = np.zeros(len(outputs))        
        # All or nothing
        outputs_ = outputs[-len(targets):]
        # if lvd(outputs_, targets)[0] == 0: r[-1] = +1.
        if len(outputs_) == len(targets): r[-1] = (outputs_ == targets).all()
        
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
    avg_rewards, avg_general, grad_norms = reinforce(ghu,
        num_epochs = 250,
        episode_duration = episode_duration,
        training_example = training_example,
        testing_example = testing_example,
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
    
    # num_reps = 20
    # num_episodes = 5000
    num_reps = 20
    num_episodes = 8000
    save_base = "results/big_reverse/len4/run_%d_%d.pkl"
    
    # # Run the experiment
    # for rep in range(num_reps):
    #     save_file = save_base % (num_episodes, rep)
    #     reverse_trial(num_episodes, save_file)
    
    # Load results
    results = {}
    for rep in range(num_reps):
        save_file = save_base % (num_episodes, rep)
        with open(save_file,"rb") as f:
            results[rep] = pk.load(f)
    
    # Plot results
    pt.figure(figsize=(4.25,2.))
    bg = (.9,.9,.9) # background color
    avg_rewards = np.array([results[rep][1] for rep in results.keys()]).T

    pt.plot(avg_rewards, c=bg, zorder=0)
    fg = tuple([.1]*3) # foreground color
    pt.plot(avg_rewards.mean(axis=1), c=fg, zorder=1, label=("avg of %d reps" % num_reps))

    # pt.title("Learning curves")
    pt.ylabel("Average Reward")
    pt.xlabel("Epoch")
    pt.ylim([-.3,1])
    pt.legend(loc="lower right")
    pt.tight_layout()
    # pt.savefig('big_reverse_learning_curves.eps')
    pt.show()
    

