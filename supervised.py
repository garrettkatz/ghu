import numpy as np
import torch as tr
import pickle as pk
from controller import *

def supervise(ghu_init, num_epochs, episode_duration, training_example, task,
    learning_rate=0.1, verbose=3, save_file=None):
    # ghu_init: initial ghu cloned for each episode
    # training_example: function that produces an example
    # reward: function of ghu, target/actual output

    controller = ghu_init.controller
    codec = ghu_init.codec

    #avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    #dist_change = np.zeros(num_epochs)
    #dist_vars = np.zeros(num_epochs)
    
    # Train
    for epoch in range(num_epochs):

        # Clone initial GHU with controller/codec and associations
        if verbose > 1: print("Cloning GHU...")
        ghu = ghu_init.clone()

        # Get random examples
        if verbose > 1: print("Sampling problem instances...")
        inputs, targets = zip(*[training_example() for b in range(ghu.batch_size)])

        # Run GHU
        if verbose > 1: print("Running GHU...")
        outputs = []
        for t in range(episode_duration):
            if verbose > 1: print(" t=%d..." % t)

            if t < len(inputs[0]):
                ghu.v[t]["rinp"] = tr.stack([
                    codec.encode("rinp", inputs[b][t])
                    for b in range(ghu.batch_size)])
            ghu.tick() # Take a step
            outputs.append([
                codec.decode("rout", ghu.v[t+1]["rout"][b,:])
                for b in range(ghu.batch_size)])

        # Rearrange outputs by batch
        outputs = [[outputs[t][b]
            for t in range(episode_duration)]
                for b in range(ghu.batch_size)]

        # # Assess rewards
        # if verbose > 1: print("Assessing reward...")
        # rewards = np.array([
        #     reward(ghu, targets[b], outputs[b])
        #     for b in range(ghu.batch_size)])
        # R = rewards.sum(axis=1)

        # Show episode results
        if verbose > 0:
            for b in range(min(ghu.batch_size, 5)):
                print(" Epoch %d, episode %d: task: %s %s -> %s vs %s" % (
                    epoch, b, task, list(inputs[b]), list(outputs[b]), list(targets[b])))
            b = R.argmax()
            if b >= 5:
                print(" Epoch %d, episode %d: task: %s %s -> %s vs %s, R=%f" % (
                    epoch, b, task, list(inputs[b]), list(outputs[b]), list(targets[b]), R[b]))
        
        # Re-organize for batch processing
        print("Re-organizing for batch processing...")
        V = tr.stack([
            tr.cat([ghu.v[t][q] for q in controller.input_keys], dim=1)
            for t in range(episode_duration)])
        H_0 = tr.zeros(1, ghu.batch_size, controller.hidden_size)
        AC = {q: tr.cat([ghu.ac[t][q] for t in range(episode_duration)])
            for q in ghu.layer_sizes}
        PC = tr.cat([ghu.pc[t] for t in range(episode_duration)])

        # Calculate policy gradient
        print(" Calculating pre-gradient...")
        AD, PD, _ = controller.forward(V, H_0)
        AL, PL = get_likelihoods(AC, PC, AD, PD)

        J = 0.
        if len(ghu_init.plastic) > 0:
            J += tr.sum(baselined_rewards_to_go.t() * tr.log(PL).squeeze())
            J -= tr.sum(tr.masked_select(PL, PL > likelihood_cap))
        for AL_q in AL.values():
            J += tr.sum(baselined_rewards_to_go.t() * tr.log(AL_q).squeeze())
            J -= tr.sum(tr.masked_select(AL_q, AL_q > likelihood_cap))
        J *= 1./ghu.batch_size

        for D in list(AD.values()) + ([PD] if len(ghu.plastic) > 0 else []):
            variance = ((D - D.mean(dim=1).unsqueeze(1))**2).mean()
            dist_vars[epoch] += variance.item()
            J -= distribution_variance_coefficient * variance

        print(" Autodiff...")
        J.backward()
        
        # Policy update
        print(" Updating model...")
        for p in controller.parameters():
            if p.data.numel() == 0: continue # happens for plastic = []
            grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
            p.data += p.grad * learning_rate # Take ascent step

        # Line-search
        lr = learning_rate
        AD0, PD0 = AD, PD
        D0 = list(AD0.values()) + ([PD0] if len(ghu.plastic) > 0 else [])
        for itr in range(line_search_iterations):

            AD, PD, _ = controller.forward(V, H_0)
            D = list(AD.values()) + ([PD] if len(ghu.plastic) > 0 else [])
            AL, PL = get_likelihoods(AC, PC, AD, PD)
            L = list(AL.values()) + ([PL] if len(ghu.plastic) > 0 else [])
            # if all([l.mean() < likelihood_cap for l in [PL] + list(AL.values())]): break
            max_likelihood = max([l.max() for l in L])
            changes = [(d-d0).abs().max() for (d,d0) in zip(D, D0)]
            dist_change[epoch] = max(changes)
            if max(changes) < distribution_cap and max_likelihood < likelihood_cap: break

            lr *= .5
            for p in controller.parameters():
                if p.data.numel() == 0: continue # happens for plastic = []
                p.data -= p.grad * lr # Descent halfway back

        for p in controller.parameters():
            if p.data.numel() == 0: continue # happens for plastic = []
            p.grad *= 0 # Clear gradients for next epoch

        saturation = tr.cat([l.flatten() for l in [PL] + list(AL.values())])
        if verbose > 0:
            print(" Avg reward = %.2f +/- %.2f (%.2f, %.2f), |~D| = %f, Var D = %f" %
                (avg_rewards[epoch], R.std(), R.min(), R.max(), dist_change[epoch], dist_vars[epoch]))
            print(" saturation=%f +/- %.2f (%f, %f), |grad| = %f" %
                (saturation.mean(), saturation.std(), saturation.min(), saturation.max(),
                grad_norms[epoch]))
        
        # Delete ghu clone to save memory
        del ghu

        # if epoch > 0 and epoch % 100 == 0:
        #     yn = input("Continue? [y/n]")
        #     if yn == "n": break

        if save_file is None: continue
        config = {'num_episodes': ghu_init.batch_size,
            'layer_size': ghu_init.v[-1]["rinp"].shape[-1],
            'hidden_size': ghu_init.h[-1].shape[-1],
            'episode_duration': episode_duration,
            'learning_rate': learning_rate,
            'line_search_iterations': line_search_iterations,
            'distribution_cap': distribution_cap,
            'likelihood_cap': likelihood_cap,
            'distribution_variance_coefficient': distribution_variance_coefficient}
        with open(save_file, "wb") as f:
            pk.dump((config, avg_rewards, grad_norms, dist_vars), f)

    return avg_rewards, grad_norms

