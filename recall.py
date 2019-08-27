import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import Codec
from controller import Controller
from lvd import lvd
from reinforce import *

if __name__ == "__main__":
    print("*******************************************************")
    
    # GHU settings
    num_symbols = [str(a) for a in range(4)]
    alpha = ["a","b","c",]
    layer_sizes = {"rinp": 512, "rout":512, "rtemp":512}
    hidden_size = 128
    plastic = []

    symbols = num_symbols+alpha+[":"]
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    print(symbols)

    codec = Codec(layer_sizes, symbols, rho=.9999)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(
        layer_sizes, pathways, controller, codec, plastic=plastic)
    ghu.associate(associations)
    for p,s,t in associations:
        q,r = ghu.pathways[p]
        assert(codec.decode(q, tr.mv( ghu.W[p], codec.encode(r, s))) == t)
    ghu_init = ghu

    separator = symbols[0]
    for k in layer_sizes.keys():
        ghu_init.v[0][k] = codec.encode(k, separator)

    def training_example():
        # Randomly choose echo symbol (excluding 0 separator)
        inputs = np.array([separator]*6)
        #key, val = np.array([separator]*2), np.array([separator]*2)
        key = np.random.choice(num_symbols[1:], size=2, replace=False)
        #key2 = np.random.choice(symbols[3:5], size=1, replace=False)
        val = np.random.choice(alpha[:], size=2, replace=False)
        #val2 = np.random.choice(alpha[2:4], size=1, replace=False)
        lookup = { key[0]:val[0], key[1]:val[1] }
        inputs[0]= key[0]
        inputs[1]= val[0]
        inputs[2]= key[1]
        inputs[3]= val[1]
        inputs[4]=":"
        new = np.random.choice(key, size=1, replace=False)
        inputs[5] = new[0]
        #print("inputs",inputs)
        targets = ['0','0','0','0','0',lookup[new[0]]]
        # print("targets", targets)
        # print("lookup",lookup)
        # print("______________________")
        return inputs, targets
    
    # # reward calculation from LVD
    # def reward(ghu, targets, outputs):
    #     # Assess reward: negative LVD after separator filtering
    #     outputs_ = [out for out in outputs if out != separator]
    #     r = -lvd(outputs_, targets)
    #     return r

    # reward calculation based on individual steps
    def reward(ghu, targets, outputs):
        #outputs_ = [out for out in outputs if out != separator]
        _, d = lvd(outputs, targets)
        r = np.zeros(len(outputs))
        for i in range(1,d.shape[0]):
            r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        return r
            
    # Optimization settings
    avg_rewards, grad_norms = reinforce(
        ghu_init,
        num_epochs = 1000,
        num_episodes = 4000,
        episode_duration = 6,
        training_example = training_example,
        reward = reward,
        task = "recall",
        learning_rate = .003,
        verbose=1)
    
    # # Optimization settings
    # num_epochs = 10000
    # num_episodes = 48	

    # avg_rewards = np.empty(num_epochs)
    # grad_norms = np.zeros(num_epochs)
    # learning_rate = .001

    # #Training
    # for epoch in range(num_epochs):

    #     ghus = []
    #     rewards = np.empty(num_episodes)

    #     for episode in range(num_episodes):
    #         separator = symbols[0]
    #         list_length = np.random.randint(min_length, max_length+1)
    #         inputs = np.array([separator]*(list_length+1))
    #         inputs[:-1] = np.random.choice(symbols[1:list_symbols+1], size=list_length, replace=False)
    #         #print("inputs",inputs)
    #         targets = [s for s in inputs if int(s)>2]
    #         #print("targets", targets)
    #         ghu = ghu_init.clone()
    #         ghus.append(ghu)

    #         for k in layer_sizes.keys():
    #             ghu.v[0][k] = codec.encode(k, separator)

    #         #if verbose > 1 and episode < 5: print("Running GHU...")
    #         outputs = []
    #         for t in range(max_time):

    #             if t < len(inputs):
    #                 ghu.v[t]["rinp"] = codec.encode("rinp", inputs[t])
    #                 #print("IN",inputs[t])
    #             ghu.tick() # Take a step
    #             out = codec.decode("rout", ghu.v[t+1]["rout"]) # Read output
    #             outputs.append(out)
                
    #             #print("OUT",out)
    #         outputs_ = [out for out in outputs if out not in {separator, None}]
    #         reward = -lvd(outputs_, targets)
    #         rewards[episode] = reward

    #         # if episode < 5:
    #         #     print("Epoch %d, episode %d: filtered output %s -> %s vs %s, R=%f" % (
    #         #         epoch, episode, list(inputs), list(outputs), list(targets), reward))


    #     avg_rewards[epoch] = rewards.mean()
    #     returns = tr.tensor(rewards - avg_rewards[epoch]).float()
        
    #     # Accumulate policy gradient
    #     #print("Calculating pre-gradient...")
    #     J = 0.
    #     saturation = []
    #     for e in range(num_episodes):
    #         r = returns[e]
    #         for t in range(max_time):
    #             for g in [ghus[e].ag[t], ghus[e].pg[t]]:
    #                 for _, (_, _, prob) in g.items():
    #                     J += r * tr.log(prob)
    #         saturation.extend(ghus[e].saturation())
    #     J /= num_episodes
    #     #print("Autodiff...")
    #     J.backward()

    #     #print("Updating model...")
    #     models = [controller]
    #     for model in models:
    #         for p in model.parameters():
    #             if p.data.numel() == 0: continue # happens for plastic = []
    #             grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
    #             p.data += p.grad * learning_rate # Take ascent step
    #             p.grad *= 0 # Clear gradients for next epoch

    #     print("Avg reward = %.2f (%.2f, %.2f), |grad| = %f, saturation=%f (%f,%f)" %
    #         (avg_rewards[epoch], rewards.min(), rewards.max(), grad_norms[epoch],
    #         np.mean(saturation),np.min(saturation),np.max(saturation)))

    #     if avg_rewards[epoch]> -0.1:
    #     	break


    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve for recall")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.savefig('recall.png')
    pt.show()
