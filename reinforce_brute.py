import numpy as np
import torch as tr
import itertools as it
from controller import get_likelihoods

def reinforce_brute(ghu_init, num_epochs, episode_duration, all_training_examples, reward, task,
    learning_rate=0.1, line_search_iterations=0, distribution_cap=1., likelihood_cap=1., verbose=3):
    # ghu_init: initial ghu cloned for each episode
    # all_training_examples: list of all possible (input, target) examples
    # reward: function of ghu, target/actual output

    controller = ghu_init.controller
    codec = ghu_init.codec

    avg_rewards = np.empty(num_epochs)
    grad_norms = np.zeros(num_epochs)
    dist_change = np.zeros(num_epochs)
    
    # Enumerate all possible episodes
    # pc[0,b,p] is likelihood of plastic[p] in episode b
    # ac[q][0,b,0] is index for incoming[q][p] in episode b
    # number of possible action sequences:
    # ( 2**len(plastic) * product([len(incoming[q]) for q in layer_sizes] )**episode_duration
    num_acts = (
        2**len(ghu_init.plastic) * np.prod(list(map(len, controller.incoming.values())))
        ) ** episode_duration
    num_episodes = num_acts * len(all_training_examples)
    assert(ghu_init.batch_size == num_episodes)
    print("Enumerating all %d possible episodes..." % num_episodes)
    
    incoming_keys = list(controller.incoming.keys())
    pc = [tr.zeros(1, num_episodes, len(ghu_init.plastic)) for t in range(episode_duration)]
    ac = [{q: tr.zeros(1, num_episodes, 1).long() for q in incoming_keys}
        for t in range(episode_duration)]

    inputs, targets = [], []
    b = 0
    for (inp, targ) in all_training_examples:
        for act in it.product(*
            (it.product(
                it.product(*([0,1] for p in ghu_init.plastic)),
                it.product(*(range(len(controller.incoming[q])) for q in incoming_keys)))
            for t in range(episode_duration))):

            for t in range(episode_duration):
                pc[t][0, b, :] = tr.tensor(act[t][0])
                for i,q in enumerate(incoming_keys):
                    ac[t][q][0, b, 0] = act[t][1][i]
                
            inputs.append(inp)
            targets.append(targ)
            b += 0    

    choices = list(zip(ac, pc))
    
    # Train
    for epoch in range(num_epochs):

        # Clone initial GHU with controller/codec and associations
        if verbose > 1: print("Cloning GHU...")
        ghu = ghu_init.clone()

        # Run GHU
        if verbose > 1: print("Running GHU...")
        outputs = []
        for t in range(episode_duration):
            if verbose > 1: print(" t=%d..." % t)

            if t < len(inputs[0]):
                ghu.v[t]["rinp"] = tr.stack([
                    codec.encode("rinp", inputs[b][t])
                    for b in range(ghu.batch_size)])
            ghu.tick(choices = choices[t]) # Take a step
            outputs.append([
                codec.decode("rout", ghu.v[t+1]["rout"][b,:])
                for b in range(ghu.batch_size)])

        # Rearrange outputs by batch
        outputs = [[outputs[t][b]
            for t in range(episode_duration)]
                for b in range(ghu.batch_size)]

        # Assess rewards
        if verbose > 1: print("Assessing reward...")
        rewards = np.array([
            reward(ghu, targets[b], outputs[b])
            for b in range(ghu.batch_size)])
        R = rewards.sum(axis=1)

        # Show episode results
        if verbose > 0:
            for b in range(min(ghu.batch_size, 5)):
                print(" Epoch %d, episode %d: task: %s %s -> %s vs %s, R=%f" % (
                    epoch, b, task, list(inputs[b]), list(outputs[b]), list(targets[b]), R[b]))
            b = R.argmax()
            if b >= 5:
                print(" Epoch %d, episode %d: task: %s %s -> %s vs %s, R=%f" % (
                    epoch, b, task, list(inputs[b]), list(outputs[b]), list(targets[b]), R[b]))
                if verbose > 2:
                    for t in range(episode_duration+1):
                        print(" Step %d, reward = %f" % (t, 0 if t==0 else rewards[b,t-1]))
                        print(" layers: ",
                            {k: codec.decode(k, ghu.v[t][k][b]) for k in ghu.layer_sizes.keys()})
                        if t == episode_duration: break
                        print(" likelihoods: ",
                            {q: al[b].item() for q,al in ghu.al[t].items()},
                            [pl[b].item() for pl in ghu.pl[t] if len(ghu.plastic) > 0])

        # Compute baselined rewards-to-go
        rewards_to_go = rewards.sum(axis=1)[:,np.newaxis] - rewards.cumsum(axis=1) + rewards
        avg_rewards_to_go = rewards_to_go.mean(axis=0)
        baselined_rewards_to_go = tr.tensor(rewards_to_go - avg_rewards_to_go[np.newaxis,:]).float()
        avg_rewards[epoch] = avg_rewards_to_go[0]

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
        
        # True expectation (up to scalar, assuming deterministic E and uniform initial states
        L = list(AL.values()) + ([PL.prod(dim=2).unsqueeze(2)] if len(ghu.plastic) > 0 else [])
        P = tr.stack(L)
        P = P.squeeze().prod(dim=0).prod(dim=0)
        J = (tr.tensor(R).float()*P).sum()
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
            print(" Avg reward = %.2f +/- %.2f (%.2f, %.2f), |~D| = %f, |grad| = %f, saturation=%f (%f,%f)" %
                (avg_rewards[epoch], R.std(), R.min(), R.max(), dist_change[epoch],
                grad_norms[epoch], saturation.mean(),saturation.min(),saturation.max()))
        
        # Delete ghu clone to save memory
        del ghu

        # if epoch > 0 and epoch % 100 == 0:
        #     yn = input("Continue? [y/n]")
        #     if yn == "n": break

    return avg_rewards, grad_norms

