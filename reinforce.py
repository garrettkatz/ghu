import numpy as np
import torch as tr

def pretrain(max_time, max_iters, learning_rate, ghu_init, symbols, verbose=True):
    # Pre-train controller for uniform action distribution
    for itr in range(max_iters):
        ghu = ghu_init.clone()
        loss = 0.
        for t in range(max_time):
            for k in ghu.layer_sizes.keys():
                ghu.v[t][k] = ghu.codec.encode(k, np.random.choice(symbols))
            ghu.tick()
            for (gates,_,_) in ghu.pg[t].values():
                loss += tr.sum((gates-.5)**2)
            for (gates,_,_) in ghu.ag[t].values():
                loss += tr.sum((gates - 1./gates.numel())**2)
        if itr % 100 == 0 and verbose: print("pretrain %d: %f" % (itr, loss.item()))
        loss.backward()
        for p in ghu.controller.parameters():
            if p.data.numel() == 0: continue # happens for plastic = []
            p.data -= p.grad * learning_rate # Take ascent step
            p.grad *= 0 # Clear gradients for next epoch


def reinforce(ghu_init, num_epochs, num_episodes, episode_duration, training_example, reward, learning_rate, verbose=2):
    # ghu_init: initial ghu cloned for each episode
    # training_example: function that produces an example
    # reward: function of ghu, target/actual output

    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    learning_rate = .1
    
    # Train
    for epoch in range(num_epochs):

        # Record episodes and rewards
        ghus = []
        rewards = []
        best_reward = -np.inf

        for episode in range(num_episodes):

            # Get random example
            if verbose > 1 and episode < 5: print("Sampling problem instance...")
            inputs, targets = training_example()
            
            # Clone initial GHU with controller/codec and associations
            if verbose > 1 and episode < 5: print("Initializing GHU...")
            ghu = ghu_init.clone()
            ghus.append(ghu)

            # Run GHU
            if verbose > 1 and episode < 5: print("Running GHU...")
            outputs = []
            for t in range(episode_duration):

                if t < len(inputs):
                    ghu.v[t]["rinp"] = ghu.codec.encode("rinp", inputs[t])
                ghu.tick() # Take a step
                out = ghu.codec.decode("rout", ghu.v[t+1]["rout"]) # Read output
                outputs.append(out)

            # Assess reward: negative LVD after separator filtering
            if verbose > 1 and episode < 5: print("Assessing reward...")
            r = reward(ghu, targets, outputs)
            rewards.append(r)

            if verbose > 1 and episode < 5:
                print("Epoch %d, episode %d: reverse %s -> %s vs %s, R=%f" % (
                    epoch, episode, list(inputs), list(outputs), list(targets), r))
            elif verbose > 2 and reward > best_reward:
                print("Epoch %d, episode %d: reverse %s -> %s vs %s, R=%f" % (
                    epoch, episode, list(inputs), list(outputs), list(targets), reward))
                best_reward = reward
                for t in range(episode_duration):
                    print(t,{k: codec.decode(k,ghu.v[t][k]) for k in ghu.layer_sizes})
                    hrs, hrl = [], []
                    for q, (gate, action, prob) in ghu.ag[t].items():
                        hrs.append("%s(%.1f~%.1f)" % (action, gate.max(), prob))
                    for p, (gate, action, prob) in ghu.pg[t].items():
                        if action > .5: hrl.append("%s(%.1f~%.1f)" % (p, gate, prob))
                    print(t,"act",str(hrs))
                    print(t,"pla",str(hrl))
                print(t,{k: codec.decode(k,ghu.v[episode_duration][k]) for k in ghu.layer_sizes})

            if r > best_reward: best_reward = r

        # Compute baselined returns (reward - average)
        rewards = np.array(rewards)
        avg_rewards[epoch] = rewards.mean()
        returns = tr.tensor(rewards - avg_rewards[epoch]).float()
        
        # Accumulate policy gradient
        print("Calculating pre-gradient...")
        J = 0.
        saturation = []
        for e in range(num_episodes):
            r = returns[e]
            for t in range(episode_duration):
                for g in [ghus[e].ag[t], ghus[e].pg[t]]:
                    for _, (_, _, prob) in g.items():
                        J += r * tr.log(prob)
            saturation.extend(ghus[e].saturation())
        J *= 1./num_episodes
        print("Autodiff...")
        J.backward()
        
        # Policy update
        print("Updating model...")
        for p in ghu.controller.parameters():
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

    return avg_rewards, grad_norms
