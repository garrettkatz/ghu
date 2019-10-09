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
    print("********************Supervised Echo************************",)
    
    # Configuration
    num_symbols = 4
    #layer_sizes = {"rinp": 3, "rout":3}
    hidden_size = 20
    rho = .99
    plastic = []
    num_episodes = 100

    # Setup GHU
    symbols = [str(a) for a in range(num_symbols)]
    length = 64 #max(getsize(len(symbols)),32)
    layer_sizes = {"rinp": length, "rout":length, "rt1":length, "rt2":length}
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=rho, requires_grad=True,ortho=False)
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
    
    def sloss(pred,y):
        # if tr.abs(tr.mean(pred-y))<1:
        loss = (tr.mean(tr.pow(pred-y, 2.0)))
        # else:
        #     loss = (tr.abs(tr.mean(pred-y))-0.5)
        return loss
    # Run optimization
    loss = supervise(ghu,
        num_epochs = 10000,
        training_example = training_example,
        task = "swap",
        episode_len=2,
        loss_fun = sloss,
        learning_rate = 0.001,
        Optimizer = tr.optim.SGD,
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
