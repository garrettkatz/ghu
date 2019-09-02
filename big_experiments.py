import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

### Reverse

# Load results
num_reps = 2
results = {}
for rep in range(num_reps):
    save_file = "results/reverse/reversegk_%d.pkl" % rep
    with open(save_file,"rb") as f:
        results[rep] = pk.load(f)

# Plot results
pt.figure(figsize=(4.25,2.5))
bg = (.75,.75,.75) # background color
min_num_epochs = np.inf
all_rewards = []
for rep in results.keys():
    _, avg_rewards, grad_norms = results[rep]
    num_epochs = (grad_norms == 0.).argmax()
    min_num_epochs = min(min_num_epochs, num_epochs)
    all_rewards.append(avg_rewards)
    pt.plot(avg_rewards[:num_epochs], c=bg, zorder=0)

fg = 'k' # foreground color
pt.plot(
    np.array(all_rewards)[:,:min_num_epochs].mean(axis=0),
    c=fg, zorder=1)

pt.title("Learning curves")
pt.ylabel("Average Reward")
pt.xlabel("Epoch")
pt.tight_layout()
pt.savefig('reversegk_learning_curves.eps')
pt.show()


