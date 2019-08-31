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
    
    # Configuration
    num_symbols = 3
    layer_sizes = {"rinp": 64, "rout":64}
    hidden_size = 16
    rho = .99
    plastic = []
    num_episodes = 200

    # Setup GHU
    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=rho)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)
    ghu = GatedHebbianUnit(
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
        inputs = np.random.choice(symbols[1:], size=1)
        targets = inputs
        return inputs, targets

    def set_choices(inputs, targets):
        # pd[t,b,p] is a bernoulli distribution for plastic[p] at time t in batch b
        # pc and pl are corresponding choices/likelihoods of the same shape
        # ad[q][t,b,p] is a softmax distribution for activity in incoming[q][p] at time t in batch b
        # ac[q][t,b,0] and al[q][t,b,0] are the corresponding choices/likelihoods from the softmax
        oi = controller.incoming["rout"].index("rout<rinp")
        ii = controller.incoming["rinp"].index("rinp<rinp")
        io = controller.incoming["rinp"].index("rinp<rout")
        oo = controller.incoming["rout"].index("rout<rout")
        ac0 = {"rout": oi*tr.ones(1, num_episodes, 1), "rinp": io*tr.ones(1, num_episodes, 1)}
        ac1 = {"rout": oi*tr.ones(1, num_episodes, 1), "rinp": ii*tr.ones(1, num_episodes, 1)}
        ac2 = {"rout": oo*tr.ones(1, num_episodes, 1), "rinp": ii*tr.ones(1, num_episodes, 1)}
        pc = tr.zeros(1, num_episodes, 0)
        choices = [(
        return choices
    
    # reward calculation based on leading LVD at individual steps
    def reward(ghu, targets, outputs):
        idx = [i for i, out in enumerate(outputs) if out != separator]
        outputs_ = [out for out in outputs if out != separator]
        _, d = lvd(outputs_, targets)
        r = np.zeros(len(outputs))
        for i in range(1,d.shape[0]):
            r[idx[i-1]] = +1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        return r

    # Set up optimal choices

    # Run optimization
    avg_rewards, grad_norms = reinforce(ghu,
        num_epochs = 50,
        episode_duration = 5,
        training_example = training_example,
        reward = reward,
        task = "echo",
        learning_rate = .1,
        verbose = 1)
    
    pt.figure(figsize=(4,3))
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.tight_layout()
    pt.show()
