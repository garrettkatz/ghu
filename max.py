import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import *

def trials(i, finalrewards):
    print("***************************** Trial ",str(i+1)," *******************************")
    
    # GHU settings
    num_symbols = 7
    layer_sizes = {"rinp": 512, "rout":512, "rtemp":512}
    hidden_size = 128
    plastic = []
    num_episodes = 2000

    symbols = [str(a) for a in range(num_symbols+1)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    print(symbols)

    codec = Codec(layer_sizes, symbols, rho=.9999)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec, plastic=plastic, batch_size = num_episodes)
    ghu.associate(associations)
    
    separator = "0"
    for k in layer_sizes.keys():
        # ghu_init.v[0][k] = codec.encode(k, separator) # !! no good anymore
        # !! Now we have to repeat the separator for each episode in the batch
        # !! v[t][k][e,:] is time t, layer k activity for episode e
        ghu.v[0][k] = tr.repeat_interleave(
            codec.encode(k, separator).view(1,-1),
            num_episodes, dim=0)

    def training_example():
        # Randomly choose echo symbol (excluding 0 separator)
        #max_time = 6
        list_symbols = 7
        min_length = 5
        max_length = 5
        list_length = np.random.randint(min_length, max_length+1)
        inputs = np.array([separator]*(list_length))
        inputs[:] = np.random.choice(symbols[1:list_symbols+1], size=list_length, replace=True)
        #print("inputs",inputs)
        targets = [max(inputs)]
        #print("targets", targets)
        return inputs, targets
    
    # # reward calculation from LVD
    # def reward(ghu, targets, outputs):
    #     # Assess reward: negative LVD after separator filtering
    #     outputs_ = [out for out in outputs if out != separator]
    #     r = -lvd(outputs_, targets)
    #     return r

    # reward calculation based on individual steps
    def reward(ghu, targets, outputs):
        #outputs_ = [out for out in outputs if out != separator]
        outputs_ = [outputs[-1]]
        _, d = lvd(outputs_, targets)
        r = np.zeros(len(outputs))
        for i in range(1,d.shape[0]):
            r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        return r
            
    # Optimization settings
    filename = "max"+str(i+1)+".png"

    avg_rewards, grad_norms = reinforce(
    ghu,
    num_epochs = 150,
    episode_duration = 5,
    training_example = training_example,
    reward = reward,
    task = "max",
    learning_rate = .03,
    verbose = 1)

    finalrewards.append(avg_rewards[-1])
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve for max")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.savefig(filename)
    


  
finalrewards = []  


for i in range(30):
    trials(i,finalrewards)

pt.plot(finalreward)
pt.title("final rewards for 30 iterations for max")
pt.ylabel("final avg reward")
pt.xlabel("Trial")
pt.savefig("maxtrials.png")
        
        