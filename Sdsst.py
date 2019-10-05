"""
Echo input (rinp) at output (rout)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
import pickle as pk
from ghu import *
from codec import *
from controller import *
from supervised import supervise
import json


if __name__ == "__main__":
    print("********************Supervised Echo************************")
    
    letters = list('abcd')
    digits = list(map(str,range(len(letters))))
    #alpha = ["a","b","c"]
    #layer_sizes = {"rinp": 512, "rout":512, "rtemp":512}
    hidden_size = 32
    plastic = []
    #plastic = ["rtemp>rinp"]

    num_episodes = 500

    symbols = ["up","left","down","right","_","+","&"] + letters + digits
    length = getsize(max(len(symbols),32))
    layer_sizes = {"rinp": length, "rout":length, "rt1":length, "rt2":length}
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=0.99, requires_grad=False,ortho=True)
    #codec.show()
    controller = SController(layer_sizes, pathways, hidden_size, plastic)
    ghu = SGatedHebbianUnit(
        layer_sizes, pathways, controller, codec,
        batch_size = num_episodes, plastic = plastic)
    ghu.associate(associations)

    # Initialize layers
    separator = "&"
    for k in layer_sizes.keys():
        ghu.v[0][k] = tr.repeat_interleave(
            codec.encode(k, separator).view(1,-1),
            num_episodes, dim=0)

    # training example generation
    def training_example():
        
        with open("datadsst.json", "r") as file:
            result1 = json.load(file)
        choice = np.random.randint(1,500)
        diff = 160 - len(result1[str(choice)][0])
        if diff!=0:
            #print("INSIDE")
            inp = result1[str(choice)][0]
            tar = result1[str(choice)][1]
            for k in range(diff):
                inp.append("&")
                tar.append("&")
            inputs = np.array(inp)
            targets = np.array(tar)
        else:
            inputs = np.array(result1[str(choice)][0])
            targets = np.array(result1[str(choice)][1])
        
        return inputs, targets


    # Run optimization
    loss = supervise(ghu,
        num_epochs = 2000,
        training_example = training_example,
        task = "dsst",
        learning_rate = .1,
        Optimizer = tr.optim.SGD ,
        verbose = 1,
        save_file = "Sdsst.pkl")
    
    with open("Sdsst.pkl","rb") as f:
        config, loss = pk.load(f)

    print(config)
    print(loss[-10:])
    #print(grad_norms[-10:])
    
    
    pt.plot(loss)
    pt.title("Learning curve")
    pt.ylabel("Loss")

    pt.xlabel("Epoch")
    #pt.tight_layout()
    pt.show()
