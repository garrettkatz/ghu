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
    plastic = ["rout<m"]
    # remove_pathways = ["rinp<rout", "m<rinp", "m<rout", "rout<m"]
    remove_pathways = ["rinp<rout", "rinp<m", "m<rinp", "m<rout"]
    # remove_pathways = []
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

    # Dataset of all possible input lists
    # list_symbols = 6
    min_length = 4
    max_length = 4
    episode_duration = 2*max_length
    all_inputs = [np.array(inputs + (separator,))
        for list_length in range(min_length, max_length+1)
            # for inputs in it.product(symbols[1:list_symbols], repeat=list_length)]
            for inputs in it.product(symbols[1:], repeat=list_length)]
    # input("%d..." % len(all_inputs))
    split = int(.80*len(all_inputs))

    # example generation
    def example(dataset):
        inputs = dataset[np.random.randint(len(dataset))]
        targets = inputs[:-1][::-1]
        return inputs, targets
    def training_example(): return example(all_inputs[:split])
    def testing_example(): return example(all_inputs[split:])
    
    # all or nothing reward calculation 
    def reward(ghu, targets, outputs):
        r = np.zeros(len(outputs))
        outputs = outputs[-len(targets):]
        if len(outputs) == len(targets): r[-1] = (outputs == targets).all()
        return r

    # ################### Sanity check
    correct_choices = [
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "m":"inc-m"}, []),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "m":"inc-m"}, ["rout<m"]),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "m":"inc-m"}, ["rout<m"]),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "m":"inc-m"}, ["rout<m"]),
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "m":"dec-m"}, []),
        ({"rinp": "rinp<rinp", "rout": "rout<m", "m":"dec-m"}, []),
        ({"rinp": "rinp<rinp", "rout": "rout<m", "m":"dec-m"}, []),
        ({"rinp": "rinp<rinp", "rout": "rout<m", "m":"dec-m"}, []),
    ]
    # ################### Sanity check

    # Run optimization
    avg_rewards, _, grad_norms = reinforce(ghu,
        num_epochs = 400,
        episode_duration = episode_duration,
        training_example = training_example,
        # testing_example = testing_example,
        testing_example = None,
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

    # Assess generalization similarly after run
    print("Cloning GHU for generalization...")
    ghu_gen = ghu.clone()
    print("Sampling problem instances...")
    inputs, targets = zip(*[testing_example() for b in range(ghu_gen.batch_size)])
    print("Running on test data...")
    outputs, rewards = ghu_gen.run(
        episode_duration, inputs, targets, reward, verbose=1)
    R_gen = rewards.sum(axis=1)

    # Overwrite file dump with R_gen in place of avg_general
    with open(save_file, "rb") as f:
        result = pk.load(f)
    result = list(result)
    result[2] = R_gen
    result = tuple(result)
    with open(save_file, "wb") as f:
        pk.dump(result, f)


if __name__ == "__main__":
    print("*******************************************************")
    
    # num_reps = 20
    # num_episodes = 50
    num_reps = 30
    num_episodes = 5000
    save_base = "results/big_reverse/len3new/run_%d_%d.pkl"
    
    # Run the experiment
    for rep in range(num_reps):
        save_file = save_base % (num_episodes, rep)
        reverse_trial(num_episodes, save_file)
    
    # Load results
    results = {}
    for rep in range(num_reps):
        save_file = save_base % (num_episodes, rep)
        with open(save_file,"rb") as f:
            results[rep] = pk.load(f)
    avg_rewards = np.array([results[rep][1] for rep in results.keys()]).T
    R_gen = np.array([results[rep][2] for rep in results.keys()])

    # Plot results
    bg = (.9,.9,.9) # background color
    fg = (.1,.1,.1) # foreground color
    pt.figure(figsize=(4.25,4.))

    pt.subplot(2,1,1)
    pt.plot(avg_rewards, c=bg, zorder=0)
    pt.plot(avg_rewards.mean(axis=1), c=fg, zorder=1, label=("avg of %d reps" % num_reps))
    pt.title("Learning curves (training set)")
    pt.ylabel("Average Reward")
    pt.xlabel("Epoch")
    pt.ylim([-.3,1])
    pt.legend(loc="lower right")

    pt.subplot(2,1,2)
    pt.hist(R_gen.mean(axis=1), bins=np.linspace(0,1,100), color='w', edgecolor='k')
    pt.ylabel("Frequency")
    pt.xlabel("Average reward (test data)")
    pt.title("Generalization performance after training")

    pt.tight_layout()
    # pt.savefig('big_reverse_learning_curves.eps')
    pt.show()
    

