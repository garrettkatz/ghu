import itertools as it
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import *
import json

def trials(i, avgrew, avggen, gradnorm):
    print("***************************** Trial ",str(i+1)," *******************************")
    
    # Configuration
    num_symbols = 10
    layer_sizes = {"rinp": 32, "rout":32, "rtemp":32}
    hidden_size = 32
    rho = .99
    plastic = []
    num_episodes = 500

    # Setup GHU
    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=rho, ortho=True)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec, plastic=plastic, batch_size = num_episodes)
    ghu.associate(associations)

    # Initialize layers
    separator = symbols[0]
    ghu.fill_layers(separator)

    # Generate dataset
    input_length = 5
    all_inputs = [np.array(inputs)
        for inputs in it.product(symbols[1:], repeat=input_length)]
    split = int(.80*len(all_inputs))

    # example generation
    def example(dataset):
        inputs = dataset[np.random.randint(len(dataset))]
        targets = np.array([max(inputs)])
        return inputs, targets
    def training_example(): return example(all_inputs[:split])
    def testing_example(): return example(all_inputs[split:])
    
    # all or nothing reward
    def reward(ghu, targets, outputs):
        r = np.zeros(len(outputs))
        r[-1] = (outputs[-1] == targets[0])
        return r
    
    # Optimization settings
    avg_rewards, avg_general, grad_norms = reinforce(
        ghu,
        num_epochs = 100,
        episode_duration = input_length,
        training_example = training_example,
        testing_example = testing_example,
        reward = reward,
        task = "max",
        learning_rate = .1,
        verbose = 1)

    gradnorm[i+1]=grad_norms.tolist()
    avgrew[i+1]=avg_rewards.tolist()
    avggen[i+1]=avg_general.tolist()
  
allgradnorms = {}
allavgrewards = {}  
allavggeneral = {}

for i in range(30):
    trials(i, allavgrewards, allavggeneral, allgradnorms)

with open("maxavgrwd.json","w") as fp:
    json.dump(allavgrewards, fp)

with open("maxavggen.json","w") as fp:
    json.dump(allavggeneral, fp)

with open("maxgradnorm.json","w") as fp:
    json.dump(allgradnorms, fp)
        
        
