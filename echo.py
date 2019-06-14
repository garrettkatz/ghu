"""
Echo input (rinp) at output (rout)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *

if __name__ == "__main__":
    
    num_symbols = 5
    layer_sizes = {"rinp": 5, "rout":5}
    hidden_size = 5

    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer(
        layer_sizes.keys(), symbols)

    c = Codec(layer_sizes, symbols)
    dc = DefaultController(layer_sizes, pathways, hidden_size)

    # Optimization settings
    num_epochs = 20
    num_episodes = 100
    max_time = 5
    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    learning_rate = 0.01
    
    # Train
    for epoch in range(num_epochs):

        # Record "actions" and rewards
        gate_streams = []
        rewards = np.empty((num_episodes, max_time))

        for episode in range(num_episodes):

            # Set up records
            gate_streams.append([])
        
            # Randomly choose echo symbol (excluding 0 separator)
            separator = symbols[0]
            echo_symbol = np.random.choice(symbols[1:])
            
            # Initialize a GHU with controller/codec and default associations
            ghu = GatedHebbianUnit(layer_sizes, pathways, dc, c)
            ghu.associate(associations)
            dc.reset()

            # Stream in echo symbol followed by separator
            ghu.v["rinp"] = c.encode("rinp", separator)
            ghu.v["rout"] = c.encode("rout", separator)
            did_echo = False
            output_stream = []
            for t in range(max_time):

                ghu.tick() # Take a step
                ghu.v["rinp"] = c.encode("rinp",
                    echo_symbol if t == 0 else separator) # Force input
                out = c.decode("rout", ghu.v["rout"]) # Read output
                output_stream.append(out)
                # print(dc.p)
                # print(dc.a)
                # print(ghu.v)
                
                # Assess reward
                if not did_echo:
                    if out == separator: r = .5
                    elif out == echo_symbol: r = 1.
                    else: r = -.5
                else:
                    if out == separator: r = .5
                    else: r = -.5
                if out == echo_symbol: did_echo = True
                
                # Update records
                gate_streams[episode].append((dc.g, dc.p, dc.a))
                rewards[episode, t] = r
            
            if episode < 5:
                print("Epoch %d, episode %d: echo %s -> %s" % (epoch, episode, echo_symbol, output_stream))

        # Compute returns
        returns = rewards.sum(axis=1)[:,np.newaxis] - rewards.cumsum(axis=1)
        returns -= returns.mean(axis=0) # baseline
        avg_rewards[epoch] = rewards[:,-1].sum()
        
        # Accumulate policy gradient
        for e in range(num_episodes):
            for t in range(max_time):
                r = tr.tensor(returns[e,t])
                g, p, a = gate_streams[e][t]
                pr = a - p.detach()
                (g * pr * r).sum().backward(retain_graph=True)
        
        # Policy update
        for model in [c, dc]:
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

