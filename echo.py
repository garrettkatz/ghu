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

if __name__ == "__main__":
    print("*******************************************************")
    
    # GHU settings
    num_symbols = 3
    layer_sizes = {"rinp": 64, "rout":64}
    hidden_size = 16
    plastic = []

    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer(
        layer_sizes.keys(), symbols)

    codec = Codec(layer_sizes, symbols, rho=.9)
    controller = Controller(layer_sizes, pathways, hidden_size)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec)
    ghu.associate(associations)
    for p,s,t in associations:
        q,r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[0][p], codec.encode(r, s))) == t)
    
    # Optimization settings
    num_epochs = 100
    num_episodes = 200
    max_time = 3
    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    learning_rate = .005
    
    # Train
    for epoch in range(num_epochs):

        # Record episodes and rewards
        ghus = []
        rewards = np.empty(num_episodes)

        for episode in range(num_episodes):

            # Randomly choose echo symbol (excluding 0 separator)
            separator = symbols[0]
            echo_symbol = np.random.choice(symbols[1:])
            
            # Initialize a GHU with controller/codec and default associations
            ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec)
            ghu.associate(associations)
            ghus.append(ghu)

            # Initialize layers
            ghu.v[0]["rinp"] = codec.encode("rinp", echo_symbol)
            ghu.v[0]["rout"] = codec.encode("rout", separator)

            # Run GHU
            outputs = []
            for t in range(max_time):

                ghu.tick(plastic=plastic) # Take a step
                out = codec.decode("rout", ghu.v[t+1]["rout"]) # Read output
                outputs.append(out)

            # Assess reward: negative LVD after separator filtering
            outputs = [out for out in outputs if out != separator]
            reward = -lvd(outputs, [echo_symbol])
            rewards[episode] = reward

            if episode < 5:
                print("Epoch %d, episode %d: echo %s -> %s, R=%f" % (
                    epoch, episode, echo_symbol, outputs, reward))

        # Compute baselined returns (reward - average)
        avg_rewards[epoch] = rewards.mean()
        returns = tr.tensor(rewards - avg_rewards[epoch]).float()
        
        # Accumulate policy gradient
        J = 0.
        saturation = 0.
        for e in range(num_episodes):
            r = returns[e]
            for t in range(max_time):
                for i in range(len(ghu.g[t])):
                    if ghus[e].a[t][i] > .5: p = ghus[e].g[t][i]
                    if ghus[e].a[t][i] < .5: p = 1. - ghus[e].g[t][i]
                    J += r * tr.log(p)
                    saturation += min(p, 1-p)
        J.backward(retain_graph=True)
        saturation /= num_episodes * max_time * len(ghus[0].g[0])
        
        # Policy update
        models = [controller]
        # grad_max = max([p.grad.abs().max() for m in models for p in m.parameters()])
        grad_max = 1.
        for model in models:
            for p in model.parameters():
                grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
                p.data += p.grad * learning_rate / grad_max # Take ascent step
                p.grad *= 0 # Clear gradients for next epoch
        print("Avg reward = %f, |grad| = %f, saturation=%f" % 
            (avg_rewards[epoch], grad_norms[epoch], saturation))
    
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.show()

