import numpy as np
import torch as tr
import pickle as pk
from controller import *

def supervise(ghu_init, num_epochs, training_example, task,
    learning_rate,Optimizer, verbose=3, save_file=None):
    # ghu_init: initial ghu cloned for each episode
    # training_example: function that produces an example
    # reward: function of ghu, target/actual output
    #parameters = [ghu_init.v,ghu_init.h, ghu_init.WL, ghu_init.WR, ghu_init.controller,ghu_init.codec]
    print("Optimizer "+str(Optimizer)+"learning_rate "+str(learning_rate))
    optimizer = Optimizer(ghu_init.controller.parameters(), lr=learning_rate)
    controller = ghu_init.controller
    codec = ghu_init.codec

    losscur = np.empty(num_epochs)
    #grad_norms = np.zeros(num_epochs)

    # Train
    #print("@@@@",codec.encoder["rout"])
    for epoch in range(num_epochs):

        # Clone initial GHU with controller/codec and associations
        if verbose > 1: print("Cloning GHU...")
        ghu = ghu_init.clone()

        # Get random examples
        if verbose > 1: print("Getting problem instances...")
        inputs, targets = zip(*[training_example() for b in range(ghu.batch_size)])
        #print("TARGETS",targets)
        # Run GHU
        tars = []
        pred1 = []
        if verbose > 1: print("Running GHU...")
        outputs = []
        for t in range(max(len(targets[0]),len(inputs[0]))):
            if verbose > 1: print(" t=%d..." % t)

            if t < len(inputs[0]):
                ghu.v[t]["rinp"] = tr.stack([
                    codec.encode("rinp", inputs[b][t])
                    for b in range(ghu.batch_size)])
            ghu.tick() # Take a step
            tars.append([codec.encoder["rout"][targets[b][t]] for b in range(ghu.batch_size)])
            pred1.append([ghu.v[t+1]["rout"][b,:]for b in range(ghu.batch_size)])
            outputs.append([
                codec.decode("rout", ghu.v[t+1]["rout"][b,:])
                for b in range(ghu.batch_size)])
        # print("OUT",ghu.v[t+1]["rout"].shape)
        # print("TARS",len(tars[0]))
        pred = tr.zeros(ghu.v[t+1]["rout"].shape)
        tt  = tr.zeros(ghu.v[t+1]["rout"].shape)
        #print("TTT",tt.shape)
        for l in range(len(tars[0])):
            tt[l,:] += tars[0][l]
            pred[l,:] += pred1[0][l]
        #print("jbkhjjhb",tt)
        
        # Rearrange outputs by batch
        outputs = [[outputs[t][b] for t in range(len(targets[0]))] for b in range(ghu.batch_size)]
        # Show episode results
        if verbose > 0:
            for b in range(min(ghu.batch_size, 3)):
                print(" Epoch %d, episode %d: task: %s %s -> %s vs %s" % (
                    epoch, b, task, list(inputs[b]), list(outputs[b]), list(targets[b])))
            

        loss = tr.tensor(0., dtype=tr.float32)
        for i in range(tt.shape[0]):
            #print("AT i", pred[i], tt[i])
            #loss += (tr.mean(tr.pow(pred[i]-tt[i], 2.0)))
            #loss = tr.nn.MSELoss(pred[i], tt[i])
            if tr.abs(pred[i]-tt[i])<1:
                loss+= 0.5*(tr.mean(tr.pow(pred[i]-tt[i], 2.0)))
            else:
                loss+= (tr.abs(pred[i]-tt[i])-0.5)

        loss *= (1/ghu.batch_size)
        print("Loss ----------------->>>>> ", loss)
        print("********************************************")
        losscur[epoch]=loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            

        
        # # Re-organize for batch processing
        # print("Re-organizing for batch processing...")
        # V = tr.stack([
        #     tr.cat([ghu.v[t][q] for q in controller.input_keys], dim=1)
        #     for t in range(targets[0])])
        # H_0 = tr.zeros(1, ghu.batch_size, controller.hidden_size)
        
        # # Calculate policy gradient
        # print(" Calculating pre-gradient...")
        # AD, PD, _ = controller.forward(V, H_0)

        # J = 0.
        # if len(ghu_init.plastic) > 0:
        #     J += tr.sum(baselined_rewards_to_go.t() * tr.log(PL).squeeze())
        #     J -= tr.sum(tr.masked_select(PL, PL > likelihood_cap))
        # for AL_q in AL.values():
        #     J += tr.sum(baselined_rewards_to_go.t() * tr.log(AL_q).squeeze())
        #     J -= tr.sum(tr.masked_select(AL_q, AL_q > likelihood_cap))
        # J *= 1./ghu.batch_size

        # for D in list(AD.values()) + ([PD] if len(ghu.plastic) > 0 else []):
        #     variance = ((D - D.mean(dim=1).unsqueeze(1))**2).mean()
        #     dist_vars[epoch] += variance.item()
        #     J -= distribution_variance_coefficient * variance

        # print(" Autodiff...")
        # J.backward()
        
        # # Policy update
        # print(" Updating model...")
        # for p in controller.parameters():
        #     if p.data.numel() == 0: continue # happens for plastic = []
        #     grad_norms[epoch] += (p.grad**2).sum() # Get gradient norm
        #     p.data += p.grad * learning_rate # Take ascent step

        # # Line-search
        # lr = learning_rate
        # AD0, PD0 = AD, PD
        # D0 = list(AD0.values()) + ([PD0] if len(ghu.plastic) > 0 else [])
        # for itr in range(line_search_iterations):

        #     AD, PD, _ = controller.forward(V, H_0)
        #     D = list(AD.values()) + ([PD] if len(ghu.plastic) > 0 else [])
        #     AL, PL = get_likelihoods(AC, PC, AD, PD)
        #     L = list(AL.values()) + ([PL] if len(ghu.plastic) > 0 else [])
        #     # if all([l.mean() < likelihood_cap for l in [PL] + list(AL.values())]): break
        #     max_likelihood = max([l.max() for l in L])
        #     changes = [(d-d0).abs().max() for (d,d0) in zip(D, D0)]
        #     dist_change[epoch] = max(changes)
        #     if max(changes) < distribution_cap and max_likelihood < likelihood_cap: break

        #     lr *= .5
        #     for p in controller.parameters():
        #         if p.data.numel() == 0: continue # happens for plastic = []
        #         p.data -= p.grad * lr # Descent halfway back

        # for p in controller.parameters():
        #     if p.data.numel() == 0: continue # happens for plastic = []
        #     p.grad *= 0 # Clear gradients for next epoch

        # saturation = tr.cat([l.flatten() for l in [PL] + list(AL.values())])
        # if verbose > 0:
        #     print(" Avg reward = %.2f +/- %.2f (%.2f, %.2f), |~D| = %f, Var D = %f" %
        #         (avg_rewards[epoch], R.std(), R.min(), R.max(), dist_change[epoch], dist_vars[epoch]))
        #     print(" saturation=%f +/- %.2f (%f, %f), |grad| = %f" %
        #         (saturation.mean(), saturation.std(), saturation.min(), saturation.max(),
        #         grad_norms[epoch]))
        
        # Delete ghu clone to save memory
        del ghu

        # if epoch > 0 and epoch % 100 == 0:
        #     yn = input("Continue? [y/n]")
        #     if yn == "n": break

        if save_file is None: continue
        config = {'num_episodes': ghu_init.batch_size,
            'layer_size': ghu_init.v[-1]["rinp"].shape[-1],
            'hidden_size': ghu_init.h[-1].shape[-1],
            'learning_rate': learning_rate}
        with open(save_file, "wb") as f:
            pk.dump((config, losscur), f)

    return losscur

