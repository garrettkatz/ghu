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

if __name__ == "__main__":
    print("*******************************************************")
    
    # GHU settings
    num_symbols =6
    layer_sizes = {"rinp": 512, "rout":512}
    hidden_size = 128
    plastic = []
    num_episodes = 1000 # !! This is now a batch-size parameter to the GHU

    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    codec = Codec(layer_sizes, symbols, rho=.9)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec, plastic=plastic,
        batch_size = num_episodes) # !! This is where the GHU gets its batch-size
    ghu.associate(associations)
    # !! The following is now checked by default in ghu.associate
    # for p,s,t in associations:
    #     q,r = ghu.pathways[p]
    #     assert(codec.decode(q, tr.mv( ghu.W[p], codec.encode(r, s))) == t)
    # ghu_init = ghu # !! unnecessary (encapsulated in reinforce)

    # Initialize layers
    separator = "0"
    for k in layer_sizes.keys():
        # ghu_init.v[0][k] = codec.encode(k, separator) # !! no good anymore
        # !! Now we have to repeat the separator for each episode in the batch
        # !! v[t][k][e,:] is time t, layer k activity for episode e
        ghu.v[0][k] = tr.repeat_interleave(
            codec.encode(k, separator).view(1,-1),
            num_episodes, dim=0)

    # training example generation
    def training_example():
        # Randomly choose echo symbol (excluding 0 separator)
        inputs = np.random.choice(symbols[1:], size=1)
        targets = [inputs[0] for i in range(int(inputs[0]))]
        return inputs, targets
    
    # reward calculation based on individual steps
    def reward(ghu, targets, outputs):
        outputs_ = [out for out in outputs if out != separator]
        _, d = lvd(outputs_, targets)
        r = np.zeros(len(outputs))
        for i in range(1,d.shape[0]):
            r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        return r

    # Optimization settings
    avg_rewards, grad_norms = reinforce(
        ghu, # !! no more ghu_init, no more num_episodes (ghu.batch_size provides it)
        num_epochs = 1200,
        episode_duration = 6,
        training_example = training_example,
        reward = reward,
        task = "echov2",
        learning_rate = 0.008,
        verbose = 1)
    
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve of echov2")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.savefig("echov2.png")
    pt.show()
