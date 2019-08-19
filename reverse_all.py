"""
Reverse input (rinp) on output (rout) with turing layer m
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
import itertools as it
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd

if __name__ == "__main__":
    print("*******************************************************")
    
    verbose = 1
    
    num_addresses = 4
    # register_names = ["rinp","rout","r0"]
    register_names = ["rinp","rout"]
    # layer_sizes = {"rinp": 64, "rout":64, "r0": 64, "r1": 64}
    # layer_sizes = {"rinp": 256, "rout": 256, "m": 256}
    layer_sizes = {q: 64 for q in register_names+["m"]}
    hidden_size = 16
    # plastic = ["%s<m"%q for q in register_names]
    plastic = ["rinp<m"]

    symbols = [str(a) for a in range(num_addresses)]
    pathways, associations = turing_initializer(
        register_names, num_addresses)

    # constrain pathways inductive bias
    remove_pathways = ["rout<m", "m<rout"]
    for p in remove_pathways: pathways.pop(p)
    associations = list(filter(lambda x: x[0] not in remove_pathways, associations))
    # print(pathways)
    # print(associations)

    codec = Codec(layer_sizes, symbols, rho=.999)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)
    # controller = Controller(layer_sizes, pathways, hidden_size, input_keys=["m","rinp"])

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic)
    ghu.associate(associations)
    for p, s, t in associations:
        q, r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[0][p], codec.encode(r, s))) == t)
    
    # Pre-train for uniform action distribution
    tol=.25
    max_time=4
    max_iters=500
    learning_rate=0.01
    for itr in range(max_iters):
        ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic)
        ghu.associate(associations)
        loss = 0.
        for t in range(max_time):
            for k in layer_sizes.keys():
                ghu.v[t][k] = codec.encode(k, np.random.choice(symbols))
            ghu.tick()
            for (gates,_,_) in ghu.pg[t].values():
                loss += tr.sum((gates-.5)**2)
            for (gates,_,_) in ghu.ag[t].values():
                loss += tr.sum((gates - 1./gates.numel())**2)
        if itr % 100 == 0: print("pretrain %d: %f" % (itr, loss.item()))
        loss.backward()
        for p in ghu.controller.parameters():
            if p.data.numel() == 0: continue # happens for plastic = []
            p.data -= p.grad * learning_rate # Take ascent step
            p.grad *= 0 # Clear gradients for next epoch
    
    
    # Optimization settings
    num_epochs = 2000
    list_symbols = 4
    min_length = 3
    max_length = 3
    max_time = 2*max_length+1
    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    learning_rate = .0001
    
    # Train
    for epoch in range(num_epochs):

        # Record episodes and rewards
        ghus = []
        rewards = []
        best_reward = -2.

        episode = -1
        for list_length in range(min_length, max_length+1):
            for list_perm in it.permutations(symbols[1:list_symbols], list_length):

                # Number of possible action sequences:
                # ( 2**len(plastic) * product([len(incoming[q]) for q in layer_sizes] )**list_length
                # incoming:
                    # rinp: <m, <rout, <rinp (3)
                    # rout: <rinp, <rout (2)
                    # m: <rinp, <m, inc, dec (4)
                # (2**1 * product([3,2,4])) ** 3 = 110592

                episode += 1

                # Get current example
                if verbose > 1 and episode < 5: print("Sampling problem instance...")
                separator = symbols[0]
                inputs = np.array([separator]*(list_length+1))
                inputs[:-1] = list_perm
                targets = inputs[:-1][::-1]
                
                # Initialize a GHU with controller/codec and default associations
                if verbose > 1 and episode < 5: print("Initializing GHU with associations...")
                ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic)
                ghu.associate(associations)
                ghus.append(ghu)
    
                # Initialize layers
                for k in layer_sizes.keys():
                    ghu.v[0][k] = codec.encode(k, separator)
    
                # Run GHU
                if verbose > 1 and episode < 5: print("Running GHU...")
                outputs = []
                for t in range(max_time):
    
                    if t < len(inputs):
                        ghu.v[t]["rinp"] = codec.encode("rinp", inputs[t])
                    ghu.tick() # Take a step
                    out = codec.decode("rout", ghu.v[t+1]["rout"]) # Read output
                    outputs.append(out)
    
                # Assess reward: negative LVD after separator filtering
                if verbose > 1 and episode < 5: print("Assessing reward...")
                outputs_ = [out for out in outputs if out != separator]
                reward = -lvd(outputs_, targets)
                rewards.append( reward )
    
                if episode < 5:
                    print("Epoch %d, episode %d: reverse %s -> %s vs %s, R=%f" % (
                        epoch, episode, list(inputs), list(outputs), list(targets), reward))
                # if reward > best_reward:
                #     print("Epoch %d, episode %d: reverse %s -> %s vs %s, R=%f" % (
                #         epoch, episode, list(inputs), list(outputs), list(targets), reward))
                #     best_reward = reward
                #     for t in range(max_time):
                #         print(t,{k: codec.decode(k,ghu.v[t][k]) for k in ghu.layer_sizes})
                #         hrs, hrl = [], []
                #         for q, (gate, action, prob) in ghu.ag[t].items():
                #             hrs.append("%s(%.1f~%.1f)" % (action, gate.max(), prob))
                #         for p, (gate, action, prob) in ghu.pg[t].items():
                #             if action > .5: hrl.append("%s(%.1f~%.1f)" % (p, gate, prob))
                #         print(t,"act",str(hrs))
                #         print(t,"pla",str(hrl))
                #     print(t,{k: codec.decode(k,ghu.v[max_time][k]) for k in ghu.layer_sizes})

        # Compute baselined returns (reward - average)
        num_episodes = episode
        rewards = np.array(rewards)
        avg_rewards[epoch] = rewards.mean()
        returns = tr.tensor(rewards - avg_rewards[epoch]).float()
        
        # Accumulate policy gradient
        print("Calculating pre-gradient...")
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
        print("Autodiff...")
        J.backward()
        
        # Policy update
        print("Updating model...")
        models = [controller]
        for model in models:
            for p in model.parameters():
                if p.data.numel() == 0: continue # happens for plastic = []
                grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
                p.data += p.grad * learning_rate # Take ascent step
                p.grad *= 0 # Clear gradients for next epoch

        print("Avg reward = %.2f (%.2f, %.2f), |grad| = %f, saturation=%f (%f,%f)" %
            (avg_rewards[epoch], rewards.min(), rewards.max(), grad_norms[epoch],
            np.mean(saturation),np.min(saturation),np.max(saturation)))
        
        # if epoch > 0 and epoch % 100 == 0:
        #     yn = input("Continue? [y/n]")
        #     if yn == "n": break
    
    pt.subplot(2,1,1)
    pt.plot(avg_rewards[:epoch+1])
    pt.title("Learning curve")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms[:epoch+1])
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.show()


