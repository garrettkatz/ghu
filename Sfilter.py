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


if __name__ == "__main__":
    print("********************Supervised filter************************")
    
    # Configuration
    rho = .99
    num_symbols = 9
    
    hidden_size = 24
    plastic = []
    num_episodes = 500

    # Setup GHU
    symbols = [str(a) for a in range(num_symbols)]
    length = max(getsize(len(symbols)),32)
    layer_sizes = {"rinp": length, "rout":length}
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=rho, requires_grad=False,ortho=True)
    #codec.show()
    controller = SController(layer_sizes, pathways, hidden_size, plastic)
    ghu = SGatedHebbianUnit(
        layer_sizes, pathways, controller, codec,
        batch_size = num_episodes, plastic = plastic)
    ghu.associate(associations)

    # Initialize layers
    separator = "0"
    for k in layer_sizes.keys():
        ghu.v[0][k] = tr.repeat_interleave(
            codec.encode(k, separator).view(1,-1),
            num_episodes, dim=0)

    # training example generation
    def training_example():
        # Randomly choose echo symbol (excluding 0 separator)
        #max_time = 6
        list_symbols = 8
        min_length = 8
        max_length = 8
        list_length = np.random.randint(min_length, max_length+1)
        inputs = np.array([separator]*(list_length))
        inputs[:] = np.random.choice(symbols[1:], size=list_length, replace=False)
        #print("inputs",inputs)
        targets = [s for s in inputs if int(s)>4]
        return inputs, targets
    

    # Run optimization
    loss = supervise(ghu,
        num_epochs = 100,
        training_example = training_example,
        task = "filter",
        learning_rate = .01,
        verbose = 1,
        save_file = "tmp.pkl")
    
    with open("tmp.pkl","rb") as f:
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
