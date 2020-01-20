"""
Associative recall
key-value pairs, followed by key, in rinp
associated value in rout
one extra register (rtmp) for key>value learning in rtmp>rinp
"""
import numpy as np
import torch as t5
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import reinforce

if __name__ == "__main__":
    print("*******************************************************")
    
    # Configuration
    layer_sizes = {"rinp": 64, "rout": 64, "rtmp": 64}
    hidden_size = 64
    rho = .99
    plastic = ["rout<rtmp"]
    remove_pathways = ["rinp<rout", "rinp<rtmp", "rtmp<rout"]
    num_episodes = 10000

    # Setup GHU
    num_symbols = 5
    chars = "abcdefghi"
    numbs = "123456789"
    symbols = chars[:num_symbols] + "0" + numbs[:num_symbols]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
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
    def training_example():
        # Randomly choose key-value pairs (excluding 0 separator)
        keys = np.random.choice(list(chars[:num_symbols]), size=2, replace=False)
        vals = np.random.choice(list(numbs[:num_symbols]), size=2, replace=False)
        i = np.random.randint(2) # index of k-v pair for prompt
        inputs = [keys[0], vals[0], keys[1], vals[1], keys[i], "0"]
        targets = [vals[i]]
        return inputs, targets
    
    # reward calculation based on leading LVD at individual steps
    def reward(ghu, targets, outputs):
        # # Care everywhere with separators
        # idx = list(range(8))
        # outputs_ = outputs
        # targets = [0]*7 + targets
        # # Filter separators anywhere
        # idx = [i for i, out in enumerate(outputs) if out != separator]
        # outputs_ = [out for out in outputs if out != separator]

        # # Only care about time after input
        # idx = [7]
        # outputs_ = outputs[7:]
        # _, d = lvd(outputs_, targets)
        # r = np.zeros(len(outputs))
        # for i in range(1,d.shape[0]):
        #     r[idx[i-1]] = +1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.

        # all or nothing at final time-step
        r = np.zeros(len(outputs))
        r[-1] = (targets[-1] == outputs[-1])
        return r

    # ################### Sanity check
    correct_choices = [
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "rtmp":"rtmp<rinp"}, [0.0]),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "rtmp":"rtmp<rtmp"}, [0.0]),
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "rtmp":"rtmp<rinp"}, [1.0]),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "rtmp":"rtmp<rtmp"}, [0.0]),
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "rtmp":"rtmp<rinp"}, [1.0]),
        ({"rinp": "rinp<rinp", "rout": "rout<rtmp", "rtmp":"rtmp<rtmp"}, [0.0]),
    ]
    # ################### Sanity check
            
    # Run optimization
    avg_rewards, grad_norms = reinforce(ghu,
        num_epochs = 500,
        episode_duration = 6,
        training_example = training_example,
        reward = reward,
        task = "recall",
        learning_rate = .1,
        # line_search_iterations = 5,
        # distribution_cap = .1,
        # likelihood_cap = .7,
        distribution_variance_coefficient = .05,
        # choices=correct_choices, # perfect rewards with this
        verbose = 1,
        save_file = "recall_plastic.pkl")
    
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
    pt.savefig("recall_plastic.png")
    pt.show()
