"""
Echo input (rinp) at output (rout)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import *
from controller import Controller
from lvd import lvd
from reinforce import reinforce
import json 

def trials(i, avgrew, gradnorm):
    print("***************************** Trial ",str(i+1),"*******************************")
    
    # GHU settings
    num_symbols =6
    
    hidden_size = 24
    plastic = []
    num_episodes=2000

    symbols = [str(a) for a in range(num_symbols)]
    length = getsize(len(symbols))
    layer_sizes = {"rinp": length, "rout":length}
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    codec = Codec(layer_sizes, symbols, rho=.9, requires_grad = False, ortho = True)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec, plastic=plastic, batch_size=num_episodes)
    ghu.associate(associations)

    # Initialize layers
    separator = "0"
    ghu.fill_layers(separator)

    # training example generation
    def training_example():
        # Randomly choose echo symbol (excluding 0 separator)
        inputs = np.random.choice(symbols[1:], size=1)
        targets = [inputs[0] for i in range(int(inputs[0]))]
        return inputs, targets
    
    # # reward calculation from LVD
    # def reward(ghu, targets, outputs):
    #     # Assess reward: negative LVD after separator filtering
    #     outputs_ = [out for out in outputs if out != separator]
    #     l, _ = lvd(outputs_, targets)
    #     return -l

    # reward calculation based on individual steps
    def reward(ghu, targets, outputs):
        outputs_ = [out for out in outputs if out != separator]
        _, d = lvd(outputs_, targets)
        r = np.zeros(len(outputs))
        for i in range(1,d.shape[0]):
            r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        return r
    
    filename = "echov2"+str(i+1)+".png"
    # Optimization settings
    avg_rewards, grad_norms = reinforce(
        ghu,
        num_epochs = 1200,
        episode_duration = 6,
        training_example = training_example,
        reward = reward,
        task = "echov2",
        learning_rate = 0.008,
        verbose = 1)

    gradnorm[i+1]=grad_norms.tolist()
    avgrew[i+1]=avg_rewards.tolist()

    


allgradnorms = {}
allavgrewards = {}  


for i in range(1):
    trials(i,allavgrewards, allgradnorms)

# with open("echov2avgrwd.json","w") as fp:
#     json.dump(allavgrewards, fp)

# with open("echov2gradnorm.json","w") as fp:
#     json.dump(allgradnorms, fp)


