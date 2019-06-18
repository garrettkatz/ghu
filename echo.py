"""
Echo input (rinp) at output (rout)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller

if __name__ == "__main__":
    
    num_symbols = 3
    layer_sizes = {"rinp": 64, "rout":64}
    hidden_size = 16

    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer(
        layer_sizes.keys(), symbols)

    codec = Codec(layer_sizes, symbols)
    controller = Controller(layer_sizes, pathways, hidden_size)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec)
    ghu.associate(associations)
    for p,s,t in associations:
        q,r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[0][p], codec.encode(r, s))) == t)
    
    # Optimization settings
    num_epochs = 50
    num_episodes = 100
    max_time = 3
    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    learning_rate = 1
    
    # Train
    for epoch in range(num_epochs):

        # Record rewards
        rewards = np.empty(num_episodes)

        for episode in range(num_episodes):

            # Randomly choose echo symbol (excluding 0 separator)
            separator = symbols[0]
            echo_symbol = np.random.choice(symbols[1:])
            
            # Initialize a GHU with controller/codec and default associations
            ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec)
            ghu.associate(associations)

            # Initialize layers
            ghu.v[0]["rinp"] = codec.encode("rinp", echo_symbol)
            ghu.v[0]["rout"] = codec.encode("rout", separator)

            # Run GHU
            outputs = []
            for t in range(max_time):

                ghu.tick() # Take a step
                out = codec.decode("rout", ghu.v[t]["rout"]) # Read output
                outputs.append(out)

            # Assess reward
            outputs = np.array(outputs)
            reward = -(1. - (outputs == echo_symbol).sum())**2
            reward -= (len(outputs)-1 - (outputs == separator).sum())**2 / (len(outputs))
            rewards[episode] = reward
            
            if episode < 5:
                print("Epoch %d, episode %d: echo %s -> %s" % (epoch, episode, echo_symbol, outputs))

        # Compute returns (reward - average)
        avg_rewards[epoch] = rewards.mean()
        returns = tr.tensor(rewards - rewards.mean()).float()
        
        # Accumulate policy gradient
        J = 0.
        for e in range(num_episodes):
            r = returns[e]
            for t in range(max_time):
                for i in range(len(ghu.g[t])):
                    if ghu.a[t][i] > .5: p = ghu.g[t][i]
                    if ghu.a[t][i] < .5: p = 1. - ghu.g[t][i]
                    J += r * tr.log(p)
        J.backward(retain_graph=True)
        
        # Policy update
        for model in [controller]:
            for p in model.parameters():
                grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
                p.data += p.grad * learning_rate # Take ascent step
                p.grad *= 0 # Clear gradients for next epoch
        print("|grad| = %f" % grad_norms[epoch])
    
    print(avg_rewards)
    print(grad_norms)
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.show()

