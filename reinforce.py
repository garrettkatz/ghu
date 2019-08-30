import numpy as np
import torch as tr
from controller import get_likelihoods

def reinforce(ghu_init, num_epochs, num_episodes, episode_duration, training_example, reward, task, learning_rate=0.1, verbose=3):
    # ghu_init: initial ghu cloned for each episode
    # training_example: function that produces an example
    # reward: function of ghu, target/actual output

    controller = ghu_init.controller
    codec = ghu_init.codec

    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    
    # Train
    for epoch in range(num_epochs):

        # Record rewards
        rewards = []
        best_reward = -np.inf

        # Clone initial GHU with controller/codec and associations
        if verbose > 1: print("Initializing GHU...")
        ghu = ghu_init.clone()

        # Get random examples
        if verbose > 1: print("Sampling problem instances...")
        inputs, targets = zip(*[training_example() for b in range(ghu.batch)])

        # Run GHU
        if verbose > 1: print("Running GHU...")
        outputs = []
        for t in range(episode_duration):

            if t < len(inputs[0]):
                ghu.v[t]["rinp"] = tr.stack([
                    codec.encode("rinp", inputs[b][t])
                    for b in range(ghu.batch)])
            ghu.tick() # Take a step
            outputs.append([
                codec.decode("rout", ghu.v[t+1]["rout"][b,:])
                for b in range(ghu.batch)])

        # Rearrange outputs by batch
        outputs = [[outputs[t][b]
            for t in range(episode_duration)]
                for b in range(ghu.batch)]

        # Assess rewards
        if verbose > 1: print("Assessing reward...")
        rewards = np.array([
            reward(ghu, targets[b], outputs[b])
            for b in range(ghu.batch)])
        R = rewards.sum(axis=1)

        # Show episode results
        if verbose > 0:
            for b in range(min(ghu.batch, 5)):
                print(" Epoch %d, episode %d: task: %s %s -> %s vs %s, R=%f" % (
                    epoch, b, task, list(inputs[b]), list(outputs[b]), list(targets[b]), R[b]))
                # if verbose > 2 and R > best_reward:
                #     for t in range(episode_duration+1):
                #         print(" Step %d, reward = %f" % (t, 0 if t==0 else r[t-1]))
                #         print(" layers: ",
                #             {k: codec.decode(k,ghu.v[t][k]) for k in ghu.layer_sizes.keys()})
                #         if t == episode_duration: break
                #         print(" actions: ", ghu.action[t])
                #         print(" likelihoods: ",
                #             {q: al.item() for q,al in ghu.al[t].items()},
                #             [pl.item() for pl in ghu.pl[t] if len(ghu.plastic) > 0])
            b = R.argmax()
            if b >= 5:
                print(" Epoch %d, episode %d: task: %s %s -> %s vs %s, R=%f" % (
                    epoch, b, task, list(inputs[b]), list(outputs[b]), list(targets[b]), R[b]))

        # Compute baselined rewards-to-go
        rewards_to_go = rewards.sum(axis=1)[:,np.newaxis] - rewards.cumsum(axis=1) + rewards
        avg_rewards_to_go = rewards_to_go.mean(axis=0)
        baselined_rewards_to_go = tr.tensor(rewards_to_go - avg_rewards_to_go[np.newaxis,:]).float()
        avg_rewards[epoch] = avg_rewards_to_go[0]

        if verbose > 0:
            print(" Avg reward = %.2f +/-%.2f (%.2f, %.2f)" %
                (avg_rewards[epoch], R.std(), R.min(), R.max()))

        # Re-organize for batch processing
        print("Re-organizing for batch processing...")
        V = tr.stack([
            tr.cat([ghu.v[t][q] for q in controller.input_keys], dim=1)
            for t in range(episode_duration)])
        H_0 = tr.zeros(1, ghu.batch, controller.hidden_size)
        AC = {q: tr.cat([ghu.ac[t][q] for t in range(episode_duration)])
            for q in ghu.layer_sizes}
        PC = tr.cat([ghu.pc[t] for t in range(episode_duration)])
        AL0 = {q: tr.cat([ghu.al[t][q] for t in range(episode_duration)])
            for q in ghu.layer_sizes}
        PL0 = tr.cat([ghu.pl[t] for t in range(episode_duration)])
        
        AD, PD, _ = controller.forward(V, H_0)
        AL, PL = get_likelihoods(AC, PC, AD, PD)
        
        # print(" ", tr.abs(PL0 - PL).mean().item(),
        #     {q: tr.abs(AL0[q]-AL[q]).mean().item() for q in AC.keys()})

        # Calculate policy gradient
        print(" Calculating pre-gradient...")
        J = 0.
        if len(ghu_init.plastic) > 0:
            J += tr.sum(baselined_rewards_to_go.t() * tr.log(PL).squeeze())
        for AL_q in AL.values():
            J += tr.sum(baselined_rewards_to_go.t() * tr.log(AL_q).squeeze())
        J *= 1./ghu.batch
        print(" Autodiff...")
        J.backward()
        
        # Policy update
        print(" Updating model...")
        for p in controller.parameters():
            if p.data.numel() == 0: continue # happens for plastic = []
            grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
            p.data += p.grad * learning_rate # Take ascent step
            p.grad *= 0 # Clear gradients for next epoch

        saturation = tr.cat([l.flatten() for l in list(AL.values()) + [PL]])
        print(" Avg reward = %.2f (%.2f, %.2f), |grad| = %f, saturation=%f (%f,%f)" %
            (avg_rewards[epoch], rewards.sum(axis=1).min(), rewards.sum(axis=1).max(),
            grad_norms[epoch], saturation.mean(),saturation.min(),saturation.max()))
        
        # Delete ghu clone to save memory
        del ghu

        # if epoch > 0 and epoch % 100 == 0:
        #     yn = input("Continue? [y/n]")
        #     if yn == "n": break

    return avg_rewards, grad_norms

