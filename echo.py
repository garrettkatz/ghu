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
    batch_size = 200
    num_episodes = 1
    plastic = []

    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    codec = Codec(layer_sizes, symbols, rho=.9)
    controller = Controller(layer_sizes, pathways, hidden_size)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, batch_size)
    ghu.associate(associations)
    for p,s,t in associations:
        q,r = ghu.pathways[p]
        for b in range(batch_size):
            assert(codec.decode(q, tr.mv( ghu.W[0][p][b], codec.encode(r, s))) == t)
    
    # Optimization settings
    num_epochs = 50
    max_time = 5
    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    learning_rate = .005
    
    # Train
    for epoch in range(num_epochs):

        # Record episodes and rewards
        ghus = []
        rewards = np.empty((num_episodes, batch_size))

        for episode in range(num_episodes):

            # Initialize a GHU with controller/codec and default associations
            ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, batch_size)
            ghu.associate(associations)
            ghus.append(ghu)

            # Randomly choose echo symbols (excluding 0 separator)
            separator = symbols[0]
            echo_symbols = np.random.choice(symbols[1:], size=(batch_size,))

            # Initialize layers
            ghu.v[0]["rinp"] = tr.stack([
                codec.encode("rinp", echo_symbols[b])
                for b in range(batch_size)])
            ghu.v[0]["rout"] = tr.stack([
                codec.encode("rout", separator)
                for b in range(batch_size)])

            # Run GHU
            outputs = [[] for b in range(batch_size)]
            for t in range(max_time):

                ghu.tick(plastic=plastic) # Take a step
                for b in range(batch_size):
                    s = codec.decode("rout", ghu.v[t+1]["rout"][b]) # read output
                    if s != separator: outputs[b].append(s) # filter out separator

            # Assess reward: negative LVD after separator filtering
            for b in range(batch_size):
                rewards[episode, b] = -lvd(outputs[b], echo_symbols[[b]])

                if episode < 2 and b < 5:
                    print("Epoch %d, episode %d, batch %d: echo %s -> %s, R=%f" % (
                        epoch, episode, b, echo_symbols[b], outputs[b], rewards[episode, b]))
            # if episode == 0:
            #     for t in range(max_time):
            #         print(t,{k: codec.decode(k,ghu.v[t][k][0]) for k in ghu.layer_sizes})
            #         hrs, hrl = [], []
            #         for q, (gate, action, prob) in ghu.ag[t].items():
            #             hrs.append("%s(%.3f~%.3f)" % (action[0], gate[0].max(), prob[0]))
            #         for p, (gate, action, prob) in ghu.pg[t].items():
            #             if action[0] > .5: hrl.append("%s(%.3f~%.3f)" % (p, gate[0], prob[0]))
            #         print(t,"act",str(hrs))
            #         print(t,"pla",str(hrl))
            #     print(t,{k: codec.decode(k,ghu.v[max_time][k][0]) for k in ghu.layer_sizes})
            
        # Compute baselined returns (reward - average)
        avg_rewards[epoch] = rewards.mean()
        returns = tr.tensor(rewards - avg_rewards[epoch]).float()
        
        # Accumulate policy gradient
        J = 0.
        saturation = 0.
        for e in range(num_episodes):
            for t in range(max_time):
                for g in [ghus[e].ag[t], ghus[e].pg[t]]:
                    for _, (_, _, prob) in g.items():
                        for b,p in enumerate(prob): # over batch
                            J += returns[e,b] * tr.log(p)
                            saturation += min(p, 1. - p)
        # J.backward(retain_graph=True)
        J.backward()
        saturation /= num_episodes * batch_size * max_time * (len(ghus[0].ag[0]) + len(ghus[0].pg[0]))
        
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
    pt.title("Learning curve")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.show()
