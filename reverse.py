"""
Echo input (rinp) at output (rout)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import reinforce
import json 

def trials(i, avgrew, gradnorm):
    print("***************************** Trial ",str(i+1),"*******************************")
   
    num_addresses = 4
    register_names = ["rinp","rout"]
    num_episodes = 5000
    
    layer_sizes = {q: 128 for q in register_names+["m"]}
    hidden_size = 32
    # plastic = ["%s<m"%q for q in register_names]
    plastic = ["rinp<m"]

    symbols = [str(a) for a in range(num_addresses)]
    pathways, associations = turing_initializer(
        register_names, num_addresses)

    # constrain pathways inductive bias
    remove_pathways = ["rout<m", "m<rout"]
    for p in remove_pathways: pathways.pop(p)
    associations = list(filter(lambda x: x[0] not in remove_pathways, associations))
    
    codec = Codec(layer_sizes, symbols, rho=.999)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic, batch_size = num_episodes)
    ghu.associate(associations)
    
    # Initialize layers
    separator = "0"
    ghu.fill_layers(separator)

    # training example generation
    list_symbols = 4
    min_length = 3
    max_length = 3
    def training_example():
        
        list_length = np.random.randint(min_length, max_length+1)
        inputs = np.array([separator]*(list_length))
        inputs[:] = np.random.choice(symbols[1:list_symbols], size=list_length, replace=False)
        targets = inputs[::-1]
        return inputs, targets
    # # reward calculation from LVD
    # def reward(ghu, targets, outputs):
    #     # Assess reward: negative LVD after separator filtering
    #     outputs_ = [out for out in outputs if out != separator]
    #     l, _ = lvd(outputs_, targets)
    #     return -l

    def reward(ghu, targets, outputs):
        outputs_ = outputs[len(targets)-1:]
        #zeros = [o for o in outputs[len(targets)-1:] if o==separator]
        #totzeros = len(zeros)
        r = np.zeros(len(outputs))
        # if len(outputs_)==0:
        #     r[-1] -= 2*(len(outputs[len(targets)-1:])+1)
        # else:
        _,d = lvd(outputs_,targets) 
        for i in range(1,d.shape[0]):
            r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -2.
            #r[-1] -= 0.1*totzeros
        return r
    
    filename = "reverse"+str(i+1)+".png"
    avg_rewards, grad_norms = reinforce(
        ghu,
        num_epochs = 1000,
        episode_duration = 2*max_length-1,
        training_example = training_example,
        reward = reward,
        task = "reverse",
        learning_rate = .03,
        verbose=2)

    gradnorm[i+1]=grad_norms.tolist()
    avgrew[i+1]=avg_rewards.tolist()

    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve of reverse")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.savefig(filename)

    


allgradnorms = {}
allavgrewards = {}  


for i in range(20):
    trials(i,allavgrewards, allgradnorms)

with open("reverseavgrwd.json","w") as fp:
    json.dump(allavgrewards, fp)

with open("reversegradnorm.json","w") as fp:
    json.dump(allgradnorms, fp)


