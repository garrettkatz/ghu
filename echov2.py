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

    symbols = [str(a) for a in range(num_symbols)]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    codec = Codec(layer_sizes, symbols, rho=.9)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec, plastic=plastic)
    ghu.associate(associations)
    for p,s,t in associations:
        q,r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[p], codec.encode(r, s))) == t)
    ghu_init = ghu

    # Initialize layers
    separator = "0"
    for k in layer_sizes.keys():
        ghu_init.v[0][k] = codec.encode(k, separator)

    # training example generation
    def training_example():
        # Randomly choose echo symbol (excluding 0 separator)
        inputs = np.random.choice(symbols[1:], size=1)
        targets = [inputs[0] for i in range(int(inputs[0]))]
        return inputs, targets
    
    # # reward calculation from LVD
    # def reward(ghu, targets, outputs):
    #     # Assess reward: negative LVD after separator filtering
    #     outputs_ = [out for out in outputs if out != separator]
    #     l, _ = lvd(outputs_, targets)
    #     return -l

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
        ghu_init,
        num_epochs = 1200,
        num_episodes = 1000,
        episode_duration = 6,
        training_example = training_example,
        reward = reward,
        task = "echov2",
        learning_rate = .008,
        verbose = 1)

    
    # # Optimization settings
    # num_epochs = 50
    # num_episodes = 100
    # max_time = 5
    # avg_rewards = np.empty(num_epochs)
    # grad_norms = np.zeros(num_epochs)
    # learning_rate = .001
    
    # # Train
    # for epoch in range(num_epochs):

    #     # Record episodes and rewards
    #     ghus = []
    #     rewards = np.empty(num_episodes)

    #     for episode in range(num_episodes):

    #         # Randomly choose echo symbol (excluding 0 separator)
    #         separator = symbols[0]
    #         echo_symbol = np.random.choice(symbols[1:])
            
    #         # Initialize a GHU with controller/codec and default associations
    #         ghu = ghu_init.clone()
    #         ghus.append(ghu)

    #         # Initialize layers
    #         ghu.v[0]["rinp"] = codec.encode("rinp", echo_symbol)
    #         ghu.v[0]["rout"] = codec.encode("rout", separator)

    #         # Run GHU
    #         outputs = []
    #         for t in range(max_time):

    #             ghu.tick() # Take a step
    #             out = codec.decode("rout", ghu.v[t+1]["rout"]) # Read output
    #             outputs.append(out)

    #         # Assess reward: negative LVD after separator filtering
    #         outputs = [out for out in outputs if out != separator]
    #         reward = -lvd(outputs, [echo_symbol])
    #         rewards[episode] = reward

    #         if episode < 5:
    #             print("Epoch %d, episode %d: echo %s -> %s, R=%f" % (
    #                 epoch, episode, echo_symbol, outputs, reward))
    #         # if episode == 4:
    #         #     for t in range(max_time):
    #         #         print(t,{k: codec.decode(k,ghu.v[t][k]) for k in ghu.layer_sizes})
    #         #         hrs, hrl = [], []
    #         #         for q, (gate, action, prob) in ghu.ag[t].items():
    #         #             hrs.append("%s(%.3f~%.3f)" % (action, gate.max(), prob))
    #         #         for p, (gate, action, prob) in ghu.pg[t].items():
    #         #             if action > .5: hrl.append("%s(%.3f~%.3f)" % (p, gate, prob))
    #         #         print(t,"act",str(hrs))
    #         #         print(t,"pla",str(hrl))
    #         #     print(t,{k: codec.decode(k,ghu.v[max_time][k]) for k in ghu.layer_sizes})
            
    #     # Compute baselined returns (reward - average)
    #     avg_rewards[epoch] = rewards.mean()
    #     returns = tr.tensor(rewards - avg_rewards[epoch]).float()
        
    #     # Accumulate policy gradient
    #     J = 0.
    #     saturation = []
    #     for e in range(num_episodes):
    #         r = returns[e]
    #         for t in range(max_time):
    #             for g in [ghus[e].ag[t], ghus[e].pg[t]]:
    #                 for _, (_, _, prob) in g.items():
    #                     J += r * tr.log(prob)
    #         saturation.extend(ghus[e].saturation())
    #     J.backward()
        
    #     # Policy update
    #     models = [controller]
    #     # grad_max = max([p.grad.abs().max() for m in models for p in m.parameters()])
    #     grad_max = 1.
    #     for model in models:
    #         for p in model.parameters():
    #             if p.data.numel() == 0: continue # happens for plastic = []
    #             grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
    #             p.data += p.grad * learning_rate / grad_max # Take ascent step
    #             p.grad *= 0 # Clear gradients for next epoch
    #     print("Avg reward = %f, |grad| = %f, saturation=%f (%f,%f)" %
    #         (avg_rewards[epoch], grad_norms[epoch],
    #         np.mean(saturation),np.min(saturation),np.max(saturation)))
    
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
