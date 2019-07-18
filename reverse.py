"""
Reverse input (rinp) on output (rout) with turing layer m
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
    
    num_addresses = 4
    # layer_sizes = {"rinp": 64, "rout":64, "r0": 64, "r1": 64}
    # layer_sizes = {"rinp": 256, "rout": 256, "m": 256}
    layer_sizes = {"rinp": 64, "rout": 64, "m": 64}
    hidden_size = 16
    plastic = ["rinp<m", "rout<m"]

    symbols = [str(a) for a in range(num_addresses)]
    pathways, associations = turing_initializer(
        ["rinp","rout"], num_addresses)

    codec = Codec(layer_sizes, symbols, rho=.9)
    controller = Controller(layer_sizes, pathways, hidden_size)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec)
    ghu.associate(associations)
    for p, s, t in associations:
        q, r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[0][p], codec.encode(r, s))) == t)
    
    # Optimization settings
    num_epochs = 200
    num_episodes = 250
    list_symbols = 3
    min_length = 2
    max_length = 2
    max_time = 2*max_length+1
    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    learning_rate = .001
    
    # Train
    for epoch in range(num_epochs):

        # Record episodes and rewards
        ghus = []
        rewards = np.empty(num_episodes)

        for episode in range(num_episodes):

            # Get random example
            separator = symbols[0]
            list_length = np.random.randint(min_length, max_length+1)
            inputs = np.array([separator]*(list_length+1))
            inputs[:-1] = np.random.choice(symbols[1:list_symbols], size=list_length, replace=False)
            targets = inputs[:-1][::-1]
            
            # Initialize a GHU with controller/codec and default associations
            ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec)
            ghu.associate(associations)
            ghus.append(ghu)

            # Initialize layers
            for k in layer_sizes.keys():
                ghu.v[0][k] = codec.encode(k, separator)

            # Run GHU
            outputs = []
            for t in range(max_time):

                if t < len(inputs):
                    ghu.v[t]["rinp"] = codec.encode("rinp", inputs[t])                
                ghu.tick(plastic=plastic) # Take a step
                out = codec.decode("rout", ghu.v[t+1]["rout"]) # Read output
                outputs.append(out)

            # Assess reward: negative LVD after separator filtering
            outputs_ = [out for out in outputs if out != separator]
            reward = -lvd(outputs_, targets)
            # reward -= sum([ghu.a[t].sum() for t in range(max_time)]) / (max_time*len(ghu.a[0])) # favor sparse gating
            rewards[episode] = reward

            if episode < 5:
                print("Epoch %d, episode %d: reverse %s -> %s vs %s, R=%f" % (
                    epoch, episode, list(inputs), list(outputs), list(targets), reward))
            # if episode == 4:
            #     for t in range(max_time):
            #         print(t,{k: codec.decode(k,ghu.v[t][k]) for k in ghu.layer_sizes})
            #         # print(ghu.a[t].numpy())
            #         hrs, hrl = [], []
            #         for i, p in enumerate(ghu.controller.pathway_keys):
            #             j = i + len(ghu.controller.pathway_keys)
            #             if ghu.a[t][i] > .5: hrs.append("s[%s]" % p)
            #             if ghu.a[t][j] > .5: hrl.append("l[%s]" % p)
            #         print(t,str(hrs))
            #         print(t,str(hrl))
            #     print(t,{k: codec.decode(k,ghu.v[max_time][k]) for k in ghu.layer_sizes})

        # Compute baselined returns (reward - average)
        avg_rewards[epoch] = rewards.mean()
        returns = tr.tensor(rewards - avg_rewards[epoch]).float()
        
        # Accumulate policy gradient
        J = 0.
        saturation = 0.
        for e in range(num_episodes):
            r = returns[e]
            for t in range(max_time):
                for g in [ghus[e].ag[t], ghus[e].pg[t]]:
                    for _, (_, _, prob) in g.items():
                        J += r * tr.log(prob)
                        saturation += min(prob, 1. - prob)
                # avg_a[t] += ghus[e].a[t]
        J.backward(retain_graph=True)
        saturation /= num_episodes * max_time * (len(ghus[0].ag[0]) + len(ghus[0].pg[0]))
        
        # Policy update
        models = [controller]
        for model in models:
            for p in model.parameters():
                grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
                p.data += p.grad * learning_rate # Take ascent step
                p.grad *= 0 # Clear gradients for next epoch

        print("Avg reward = %f, |grad| = %f, saturation=%f" % 
            (avg_rewards[epoch], grad_norms[epoch], saturation))
    
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.show()


