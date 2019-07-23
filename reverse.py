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
    
    num_addresses = 5
    # register_names = ["rinp","rout","r0"]
    register_names = ["rinp","rout"]
    # layer_sizes = {"rinp": 64, "rout":64, "r0": 64, "r1": 64}
    # layer_sizes = {"rinp": 256, "rout": 256, "m": 256}
    layer_sizes = {q: 128 for q in register_names+["m"]}
    hidden_size = 16
    plastic = ["%s<m"%q for q in register_names]

    symbols = [str(a) for a in range(num_addresses)]
    pathways, associations = turing_initializer(
        register_names, num_addresses)

    codec = Codec(layer_sizes, symbols, rho=.9)
    controller = Controller(layer_sizes, pathways, hidden_size)
    # controller = Controller(layer_sizes, pathways, hidden_size, input_keys=["m","rinp"])

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec)
    ghu.associate(associations)
    for p, s, t in associations:
        q, r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[0][p], codec.encode(r, s))) == t)
    
    # Optimization settings
    num_epochs = 600
    num_episodes = 500
    list_symbols = 4
    min_length = 3
    max_length = 3
    max_time = 2*max_length+1
    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    learning_rate = .00001
    
    # Train
    for epoch in range(num_epochs):

        # Record episodes and rewards
        ghus = []
        rewards = np.empty(num_episodes)
        best_reward = -2.

        for episode in range(num_episodes):

            # Get random example
            separator = symbols[0]
            list_length = np.random.randint(min_length, max_length+1)
            inputs = np.array([separator]*(list_length+1))
            inputs[:-1] = np.random.choice(symbols[1:list_symbols], size=list_length, replace=False)
            targets = inputs[:-1][::-1]
            
            # Initialize a GHU with controller/codec and default associations
            ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic)
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
                ghu.tick() # Take a step
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
            if reward > best_reward:
                print("Epoch %d, episode %d: reverse %s -> %s vs %s, R=%f" % (
                    epoch, episode, list(inputs), list(outputs), list(targets), reward))
                best_reward = reward
                for t in range(max_time):
                    print(t,{k: codec.decode(k,ghu.v[t][k]) for k in ghu.layer_sizes})
                    hrs, hrl = [], []
                    for q, (gate, action, prob) in ghu.ag[t].items():
                        hrs.append("%s(%.1f~%.1f)" % (action, gate.max(), prob))
                    for p, (gate, action, prob) in ghu.pg[t].items():
                        if action > .5: hrl.append("%s(%.1f~%.1f)" % (p, gate, prob))
                    print(t,"act",str(hrs))
                    print(t,"pla",str(hrl))
                print(t,{k: codec.decode(k,ghu.v[max_time][k]) for k in ghu.layer_sizes})

        # Compute baselined returns (reward - average)
        avg_rewards[epoch] = rewards.mean()
        returns = tr.tensor(rewards - avg_rewards[epoch]).float()
        
        # Accumulate policy gradient
        print("Gradient calculation...")
        J = 0.
        saturation = []
        for e in range(num_episodes):
            r = returns[e]
            for t in range(max_time):
                for g in [ghus[e].ag[t], ghus[e].pg[t]]:
                    for _, (_, _, prob) in g.items():
                        J += r * tr.log(prob)
                # avg_a[t] += ghus[e].a[t]
            saturation.extend(ghus[e].saturation())
        J.backward(retain_graph=True)
        print("Done.")
        
        # Policy update
        models = [controller]
        for model in models:
            for p in model.parameters():
                grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
                p.data += p.grad * learning_rate # Take ascent step
                p.grad *= 0 # Clear gradients for next epoch

        print("Avg reward = %f, |grad| = %f, saturation=%f (%f,%f)" %
            (avg_rewards[epoch], grad_norms[epoch],
            np.mean(saturation),np.min(saturation),np.max(saturation)))
    
    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.show()


