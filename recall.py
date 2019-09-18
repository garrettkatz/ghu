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
   
    num_symbols = [str(a) for a in range(5)]
    alpha = ["a","b","c","d"]
    #layer_sizes = {"rinp": 512, "rout":512, "rtemp":512}
    hidden_size = 40
    plastic = []
    #plastic = ["rtemp>rinp"]

    num_episodes = 6000

    symbols = num_symbols+alpha
    length = getsize(max(len(symbols),32))
    layer_sizes = {"rinp": length, "rout":length, "rtemp":length}
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    
    codec = Codec(layer_sizes, symbols, rho=.999, requires_grad=False,ortho=True)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic, batch_size = num_episodes)
    ghu.associate(associations)
    
    # Initialize layers
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
        inputs = np.array([separator]*7)
        #key, val = np.array([separator]*2), np.array([separator]*2)
        key = np.random.choice(num_symbols[1:], size=3, replace=False)
        #key2 = np.random.choice(symbols[3:5], size=1, replace=False)
        val = np.random.choice(alpha[:], size=3, replace=False)
        #val2 = np.random.choice(alpha[2:4], size=1, replace=False)
        lookup = { key[0]:val[0], key[1]:val[1], key[2]:val[2] }
        
        for i in range(3):
            inputs[2*i]=key[i]
            inputs[2*i+1]=val[i]
        new = np.random.choice(key, size=1, replace=False)
        inputs[6] = new[0]
        #print("inputs",inputs)
        targets = [lookup[new[0]]]
        # print("targets", targets)
        # print("lookup",lookup)
        # print("______________________")
        return inputs, targets

    # reward calculation based on individual steps
    def reward(ghu, targets, outputs):
        outputs_ = [outputs[-1]]
        _, d = lvd(outputs_, targets)
        r = np.zeros(len(outputs))
        for i in range(1,d.shape[0]):
            r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        return r
    
    filename = "recall"+str(i+1)+".png"
    # Optimization settings
    avg_rewards, grad_norms = reinforce(
        ghu,
        num_epochs = 10000,
        episode_duration = 7,
        training_example = training_example,
        reward = reward,
        task = "recall",
        learning_rate = .01,
        verbose=1)

    gradnorm[i+1]=grad_norms.tolist()
    avgrew[i+1]=avg_rewards.tolist()

    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve of recall")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.savefig(filename)

    


allgradnorms = {}
allavgrewards = {}  


for i in range(1):
    trials(i,allavgrewards, allgradnorms)

# with open("recallavgrwd.json","w") as fp:
#     json.dump(allavgrewards, fp)

# with open("recallgradnorm.json","w") as fp:
#     json.dump(allgradnorms, fp)


