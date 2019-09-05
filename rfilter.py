import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import *
import json

def trials(i, avgrew, gradnorm):
    print("***************************** Trial ",str(i+1)," *******************************")
    
    # GHU settings
    num_symbols = 4
    layer_sizes = {"rinp": 128, "rout":128, "rtemp1":128}
    hidden_size = 32
    plastic = []
    num_episodes = 1000

    symbols = [str(a) for a in range(num_symbols+1)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    print(symbols)

    codec = Codec(layer_sizes, symbols, rho=.9999)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec, plastic=plastic, batch_size=num_episodes)
    ghu.associate(associations)
    

    separator = symbols[0]
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
        list_symbols = 4
        min_length = 4
        max_length = 4
        list_length = np.random.randint(min_length, max_length+1)
        inputs = np.array([separator]*(list_length))
        inputs[:] = np.random.choice(symbols[1:list_symbols+1], size=list_length, replace=True)
        #print("inputs",inputs)
        targets = [s for s in inputs if int(s)>2]
        return inputs, targets
    
    # # reward calculation from LVD
    # def reward(ghu, targets, outputs):
    #     # Assess reward: negative LVD after separator filtering
    #     outputs_ = [out for out in outputs if out != separator]
    #     r = -lvd(outputs_, targets)
    #     return r

    # reward calculation based on individual steps
    # def reward(ghu, targets, outputs):
    #     outputs_ = [out for out in outputs if out != separator]
    #     _, d = lvd(outputs_, targets)
    #     r = np.zeros(len(outputs))
    #     for i in range(1,d.shape[0]):
    #         r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
    #     return r

    def reward(ghu, targets, outputs):
        outputs_ = [out for out in outputs if out!=separator]
        zeros = [o for o in outputs if o==separator]
        totzeros = len(zeros)
        r = np.zeros(len(outputs))
        if len(outputs_)==0:
            r[-1] -= (len(outputs)+1)
        else:
            _,d = lvd(outputs_,targets) 
            for i in range(1,d.shape[0]):
                r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
            r[-1] -= 0.1*totzeros
        return r
    
    
    filename = "rfilter"+str(i+1)+".png"

    #Optimization settings
    avg_rewards, grad_norms = reinforce(
        ghu,
        num_epochs = 500,
        episode_duration = 4,
        training_example = training_example,
        reward = reward,
        task = "filter with repeat",
        learning_rate = .005,
        verbose=1)
        
    gradnorm[i+1]=grad_norms.tolist()
    avgrew[i+1]=avg_rewards.tolist()
    
    
allgradnorms = {}
allavgrewards = {}  


for i in range(30):
    trials(i,allavgrewards, allgradnorms)

with open("rfilteravgrwd.json","w") as fp:
    json.dump(allavgrewards, fp)

with open("rfiltergradnorm.json","w") as fp:
    json.dump(allgradnorms, fp)