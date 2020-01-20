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
    num_symbols = 10
    layer_sizes = {"rinp": 32, "rout":32, "rtmp": 32}
    hidden_size = 32
    rho = .99
    plastic = []
    num_episodes = 100

    # Setup GHU
    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=rho, ortho=True)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec,
        batch_size = num_episodes, plastic = plastic)
    ghu.associate(associations)

    # Initialize layers
    separator = "0"
    ghu.fill_layers(separator)

    # Generate dataset
    input_length = int(episode_duration/2)
    all_inputs = []
    for symbol in symbols[1:]:
        for t in range(input_length):
            inputs = [separator]*input_length
            inputs[t] = symbol
            all_inputs.append(inputs)
    split = int(.80*len(all_inputs))

    # example generation
    def example(dataset):
        # Randomly choose echo symbol (excluding 0 separator)
        inputs = dataset[np.random.randint(len(dataset))]
        targets = [inp for inp in inputs if inp != separator]
        return inputs, targets
    def training_example(): return example(all_inputs[:split])
    def testing_example(): return example(all_inputs[split:])
    
    # all or nothing reward
    def reward(ghu, targets, outputs):
        r = np.zeros(len(outputs))
        outputs = np.array([out for out in outputs[input_length:] if out != separator])
        if len(outputs) == len(targets): r[-1] = (targets == outputs).all()
        return r

    # # correct choices for debugging
    # correct_choices = \
    #     [({"rinp": "rinp<rout", "rout": "rout<rinp"}, [])]*2 + \
    #     [({"rinp": "rinp<rinp", "rout": "rout<rout"}, [])]*(episode_duration-2)
    # # run it to debug:
    # inputs, targets = zip(*[training_example() for b in range(ghu.batch_size)])
    # ghu.run(episode_duration, inputs, targets, reward, choices=correct_choices, verbose=3)
    # input("????")

    # Run optimization
    avg_rewards, avg_general, grad_norms = reinforce(ghu,
        num_epochs = 100,
        episode_duration = episode_duration,
        training_example = training_example,
        testing_example = testing_example,
        reward = reward,
        task = "echo",
        learning_rate = .1,
        verbose = 1,
        save_file = save_file)
    
    return avg_rewards, avg_general, grad_norms

if __name__ == "__main__":
    print("*******************************************************")

    durations = range(3,7)
    num_reps = 10
    save_base = "results/echo/run_%d_%d.pkl"
    
    # Run the experiment
    for dur in durations:
        for rep in range(num_reps):
            save_file = save_base % (dur, rep)
            echo_trial(dur, save_file)

    # Load results
    results = {}
    for dur in durations:
        results[dur] = {}
        for rep in range(num_reps):
            save_file = save_base % (dur, rep)
            with open(save_file,"rb") as f:
                results[dur][rep] = pk.load(f)
    
    # Plot results
    pt.figure(figsize=(4.25,4.00))
    bg = (.9,.9,.9) # background color
    for i,mode in enumerate(["Training set", "Testing set"]):
        pt.subplot(2,1,i+1)
        for d,dur in enumerate(durations):
            avg_rewards = np.array([results[dur][rep][i+1]
                for rep in results[dur].keys()]).T

            pt.plot(avg_rewards, c=bg, zorder=0)
            fg = tuple([d/6]*3) # foreground color
            pt.plot(avg_rewards.mean(axis=1), c=fg, zorder=1, label=("T=%d" % dur))

        # pt.title(mode)
        # pt.ylabel("Avg. Reward")
        if i == 0: pt.title("Average Reward")
        pt.ylabel(mode)
        if i == 1: pt.xlabel("Epoch")
        pt.ylim([-.5,1])
        pt.legend(loc="lower right")
    pt.tight_layout()
    pt.savefig('echo_learning_curves.eps')
    pt.show()
    
    # # Histograms of final rewards
    # pt.figure(figsize=(4.25,2))
    # finals = []
    # for d,dur in enumerate(durations):
    #     avg_rewards = np.array([results[dur][rep][1]
    #         for rep in results[dur].keys()]).T
    #     finals.append(avg_rewards[-1,:])
    # pt.boxplot(finals, showfliers=False)
    # pt.title("Final Average Rewards")
    # pt.ylabel("Reward")
    # pt.xlabel("Episode duration")
    # pt.tight_layout()
    # pt.savefig('echo_finals.eps')
    # pt.show()
    
