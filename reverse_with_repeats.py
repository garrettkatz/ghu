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

def reverse_trial(num_episodes, save_file):

    # Configuration
    register_names = ["rinp","rout"]
    layer_sizes = {q: 8 for q in register_names + ["m"]}
    hidden_size = 32
    rho = .99
    plastic = ["rinp<m"]
    remove_pathways = ["rinp<rout", "m<rinp", "m<rout", "rout<m"]

    # Setup GHU
    num_addresses = 4
    symbols = [str(a) for a in range(num_addresses)]
    pathways, associations = turing_initializer(
        register_names, num_addresses)
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
    for k in layer_sizes.keys():
        ghu.v[0][k] = tr.repeat_interleave(
            codec.encode(k, separator).view(1,-1),
            num_episodes, dim=0)

    # training example generation
    list_symbols = 4
    min_length = 3
    max_length = 3
    episode_duration = 2*max_length
    def training_example():
        list_length = np.random.randint(min_length, max_length+1)
        # inputs = np.random.choice(symbols[1:list_symbols], size=list_length, replace=True)
        inputs = np.random.choice(symbols[1:list_symbols], size=list_length, replace=False)
        targets = inputs[::-1]
        return inputs, targets
    
    # reward calculation based on leading LVD at individual steps
    def reward(ghu, targets, outputs):
        r = np.zeros(len(outputs))
        
        # All or nothing
        outputs_ = outputs[len(targets):]
        if lvd(outputs_, targets)[0] == 0: r[-1] = +1.
        
        return r
            
    # Run optimization
    avg_rewards, grad_norms = reinforce(ghu,
        num_epochs = 100,
        episode_duration = episode_duration,
        training_example = training_example,
        reward = reward,
        task = "reverse",
        learning_rate = 1.,
        # line_search_iterations = 5,
        # distribution_cap = .1,
        # likelihood_cap = .7,
        # distribution_variance_coefficient = 0.05,
        verbose = 2,
        save_file = save_file)

if __name__ == "__main__":
    print("*******************************************************")
    
    num_reps = 5
    num_episodes = 16000
    
    # Run the experiment
    for rep in range(num_reps):
        save_file = "results/big_reverse/run_%d_%d.pkl" % (num_episodes, rep)
        reverse_trial(num_episodes, save_file)
    
