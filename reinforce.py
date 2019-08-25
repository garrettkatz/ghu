import numpy as np
import torch as tr
from controller import get_likelihoods

def pretrain(max_time, max_iters, learning_rate, ghu_init, symbols, verbose=True):
    # Pre-train controller for uniform action distribution
    for itr in range(max_iters):
        ghu = ghu_init.clone()
        loss = 0.
        for t in range(max_time):
            for k in ghu.layer_sizes.keys():
                ghu.v[t][k] = ghu.codec.encode(k, np.random.choice(symbols))
            ghu.tick()
            for (gates,_,_) in ghu.pl[t].values():
                loss += tr.sum((gates-.5)**2)
            for (gates,_,_) in ghu.al[t].values():
                loss += tr.sum((gates - 1./gates.numel())**2)
        if itr % 100 == 0 and verbose: print("pretrain %d: %f" % (itr, loss.item()))
        loss.backward()
        for p in ghu.controller.parameters():
            if p.data.numel() == 0: continue # happens for plastic = []
            p.data -= p.grad * learning_rate # Take ascent step
            p.grad *= 0 # Clear gradients for next epoch


def reinforce(ghu_init, num_epochs, num_episodes, episode_duration, training_example, reward, task, learning_rate=0.1, verbose=2):
    # ghu_init: initial ghu cloned for each episode
    # training_example: function that produces an example
    # reward: function of ghu, target/actual output

    num_inner = 1
    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros((num_epochs, num_inner))
    
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

            # Delete weight matrix to save memory
            del ghu.W

            # Assess reward: negative LVD after separator filtering
            if verbose > 1 and episode < 5: print("Assessing reward...")
            r = reward(ghu, targets, outputs)
            R = r.sum()
            rewards.append(r)

            if verbose > 1 and episode < 5:
                print("Epoch %d, episode %d: task: %s %s -> %s vs %s, R=%f" % (
                    epoch, episode, task, list(inputs), list(outputs), list(targets), R))
            elif verbose > 2 and R > best_reward:
                print("Epoch %d, episode %d: task: %s %s -> %s vs %s, R=%f" % (
                    epoch, episode, task, list(inputs), list(outputs), list(targets), R))
                best_reward = reward
                for t in range(episode_duration):
                    print(t,{k: codec.decode(k,ghu.v[t][k]) for k in ghu.layer_sizes})
                    hrs, hrl = [], []
                    for q, (gate, action, prob) in ghu.al[t].items():
                        hrs.append("%s(%.1f~%.1f)" % (action, gate.max(), prob))
                    for p, (gate, action, prob) in ghu.pl[t].items():
                        if action > .5: hrl.append("%s(%.1f~%.1f)" % (p, gate, prob))
                    print(t,"act",str(hrs))
                    print(t,"pla",str(hrl))
                print(t,{k: codec.decode(k,ghu.v[episode_duration][k]) for k in ghu.layer_sizes})

            if R > best_reward: best_reward = R

        # Compute baselined rewards-to-go
        rewards = np.array(rewards)
        rewards_to_go = rewards.sum(axis=1)[:,np.newaxis] - rewards.cumsum(axis=1) + rewards
        avg_rewards_to_go = rewards_to_go.mean(axis=0)
        baselined_rewards_to_go = tr.tensor(rewards_to_go - avg_rewards_to_go[np.newaxis,:]).float()
        avg_rewards[epoch] = avg_rewards_to_go[0]

        # Re-organize for batch processing
        print("Re-organizing for batch processing...")
        V = tr.cat([
            tr.cat(
                [ghus[e].v[t][q]
                    for t in range(episode_duration)
                        for e in range(num_episodes)
                ]).view(episode_duration, num_episodes, -1)
            for q in ghu_init.controller.input_keys], dim=2)
        H_0 = tr.zeros(1, num_episodes, ghu_init.controller.hidden_size)
        
        AC = {q: tr.cat(
            [ghus[e].ac[t][q]
                for t in range(episode_duration)
                    for e in range(num_episodes)
            ]).view(episode_duration, num_episodes, -1)
            for q in ghu_init.layer_sizes}
        PC = tr.cat(
            [ghus[e].pc[t]
                for t in range(episode_duration)
                    for e in range(num_episodes)
            ]).view(episode_duration, num_episodes, -1)
        AL0 = {q: tr.cat(
            [ghus[e].al[t][q]
                for t in range(episode_duration)
                    for e in range(num_episodes)
            ]).view(episode_duration, num_episodes, -1)
            for q in ghu_init.layer_sizes}
        PL0 = tr.cat(
            [ghus[e].pl[t]
                for t in range(episode_duration)
                    for e in range(num_episodes)
            ]).view(episode_duration, num_episodes, -1)
        
        
        for inr in range(1):
            AD, PD, _ = ghu_init.controller.forward(V, H_0)
            AL, PL = get_likelihoods(AC, PC, AD, PD)
            
            print(" Inner itr %d" % inr)
            print(" ",tr.abs(PL0 - PL).mean(), {q: tr.abs(AL0[q]-AL[q]).mean() for q in AC.keys()})
    
            # Accumulate policy gradient
            print(" Calculating pre-gradient...")
            J = 0.
            if len(ghu_init.plastic) > 0:
                J += tr.sum(baselined_rewards_to_go.t() * tr.log(PL).squeeze())
            for ALq in AL.values():
                J += tr.sum(baselined_rewards_to_go.t() * tr.log(ALq).squeeze())
            J *= 1./num_episodes
            print(" Autodiff...")
            J.backward()
            
            # Policy update
            print(" Updating model...")
            for p in ghu.controller.parameters():
                if p.data.numel() == 0: continue # happens for plastic = []
                grad_norms[epoch, inr] += (p.grad**2).sum() # Get gradient norm
                p.data += p.grad * learning_rate # Take ascent step
                p.grad *= 0 # Clear gradients for next epoch

            saturation = tr.cat([l.flatten() for l in list(AL.values()) + [PL]])
            print(" Avg reward = %.2f (%.2f, %.2f), |grad| = %f, saturation=%f (%f,%f)" %
                (avg_rewards[epoch], rewards.sum(axis=1).min(), rewards.sum(axis=1).max(),
                grad_norms[epoch, inr], saturation.mean(),saturation.min(),saturation.max()))
        
        # if epoch > 0 and epoch % 100 == 0:
        #     yn = input("Continue? [y/n]")
        #     if yn == "n": break

    return avg_rewards, grad_norms.mean(axis=1)

