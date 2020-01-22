"""
Associative recall
key-value pairs, followed by key, in rinp
associated value in rout
one extra register (rtmp) for key>value learning in rtmp>rinp
"""
import itertools as it
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import reinforce

def recall_trial(num_episodes, save_file):
    
    # Configuration
    layer_sizes = {"rinp": 64, "rout": 64, "rtmp": 64}
    hidden_size = 64
    rho = .99
    plastic = ["rout<rtmp"]
    # remove_pathways = ["rinp<rout", "rinp<rtmp", "rtmp<rout"]
    remove_pathways = []

    # Setup GHU
    num_symbols = 5
    chars = "abcdefghi"
    numbs = "123456789"
    symbols = chars[:num_symbols] + "0" + numbs[:num_symbols-1]
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

    # Dataset of all possible input lists
    all_inputs = [np.array([k1, v1, k2, v2, k, "0"])
        for (k1, k2) in it.permutations(chars[:num_symbols], 2)
            for (v1, v2) in it.permutations(numbs[:num_symbols-1], 2)
                for k in [k1, k2]]
    input_length = 6
    output_window = 2
    episode_duration = input_length+output_window
    split = int(.80*len(all_inputs))

    # example generation
    def example(dataset):
        inputs = dataset[np.random.randint(len(dataset))]
        targets = inputs[[1 if inputs[0] == inputs[4] else 3]]
        return inputs, targets
    def training_example(): return example(all_inputs[:split])
    def testing_example(): return example(all_inputs[split:])

    # reward calculation based on leading LVD at individual steps
    def reward(ghu, targets, outputs):
        # all or nothing at final time-step
        r = np.zeros(len(outputs))
        outputs = np.array([out for out in outputs[input_length-1:] if out != separator])
        if len(outputs) == len(targets): r[-1] = (targets == outputs).all()
        return r

    # ################### Sanity check
    correct_choices = [
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "rtmp":"rtmp<rinp"}, []),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "rtmp":"rtmp<rtmp"}, []),
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "rtmp":"rtmp<rinp"}, ["rout<rtmp"]),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "rtmp":"rtmp<rtmp"}, []),
        ({"rinp": "rinp<rinp", "rout": "rout<rout", "rtmp":"rtmp<rinp"}, ["rout<rtmp"]),
        ({"rinp": "rinp<rinp", "rout": "rout<rtmp", "rtmp":"rtmp<rtmp"}, []),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "rtmp":"rtmp<rtmp"}, []),
        ({"rinp": "rinp<rinp", "rout": "rout<rinp", "rtmp":"rtmp<rtmp"}, []),
    ]
    # ################### Sanity check
            
    # Run optimization
    avg_rewards, avg_general, grad_norms = reinforce(ghu,
        num_epochs = 500,
        episode_duration = episode_duration,
        training_example = training_example,
        testing_example = None,
        reward = reward,
        task = "recall",
        learning_rate = .1,
        # line_search_iterations = 5,
        # distribution_cap = .1,
        # likelihood_cap = .7,
        distribution_variance_coefficient = .05,
        # choices=correct_choices, # perfect rewards with this
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

    num_reps = 30
    num_episodes = 5000
    save_base = "results/recall/new/run_%d_%d.pkl"
    
    # Run the experiment
    for rep in range(num_reps):
        save_file = save_base % (num_episodes, rep)
        recall_trial(num_episodes, save_file)
    
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
    pt.savefig('recall_curves.eps')
    # pt.show()

