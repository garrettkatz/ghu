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
    print("********************Supervised Echo************************")
    
    # Configuration
    num_symbols = 4
    #layer_sizes = {"rinp": 3, "rout":3}
    hidden_size = 16
    rho = .99
    plastic = []
    num_episodes = 1000

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
        # Randomly choose swap symbols (excluding 0 separator)
        inputs = np.random.choice(symbols[1:], size=2, replace=False)
        targets = inputs[::-1]
        return inputs, targets
    

    # Run optimization
    loss = supervise(ghu,
        num_epochs = 1000,
        training_example = training_example,
        task = "swap",
        learning_rate = 0.001,
        Optimizer = tr.optim.Adam,
        verbose = 1,
        save_file = "swap.pkl")
    
    with open("swap.pkl","rb") as f:
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
