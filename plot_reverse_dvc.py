import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

def normalize(x): return (x - x.min()) / (x.max() - x.min())

# Load results
num_reps = 30
results = {}
for rep in range(num_reps):
    save_file = "results/big_reverse/len4new/run_5000_%d.pkl" % rep
    with open(save_file,"rb") as f:
        results[rep] = pk.load(f)
all_rewards = []
all_dvcs = []
for rep in results.keys():
    avg_rewards, dvcs = results[rep][1], results[rep][4]
    all_rewards.append(avg_rewards)
    all_dvcs.append(dvcs)
bg = (.75,.75,.75) # background color
fg = 'k'

# Perf vs dist var
pt.figure(figsize=(4.25,2.25))
x = (np.array(all_rewards)[:,1:] - np.array(all_rewards)[:,:-1]).flatten()
y = (np.array(all_dvcs)[:,1:] - np.array(all_dvcs)[:,:-1]).flatten()
pt.scatter(x, y, c='k',marker='+')
xlo, xhi = 1.1*x.min(), 1.1*x.max()
ylo, yhi = 1.1*y.min(), 1.1*y.max()
pt.plot([xlo, xhi], [0, 0], 'k--')
pt.plot([0, 0], [ylo, yhi], 'k--')
pt.xlim([xlo,xhi])
pt.ylim([ylo,yhi])
pt.xlabel("$\Delta$ $\mathrm{\mathbb{E}}[R_0]$")
pt.ylabel("$\Delta D$")
pt.tight_layout()
pt.savefig('reverse_dvc_deltas.eps')
pt.show()

# Distribution variance
pt.figure(figsize=(4.25,5.25))
pt.subplot(4,1,1)
for r,rep in enumerate(results.keys()):
    pt.plot(all_dvcs[r], c=bg, zorder=0)
pt.plot(np.array(all_dvcs).mean(axis=0),
    c=fg, zorder=1, label="Avg. of %d trials" % len(results))
pt.legend(loc="upper left")
pt.ylabel("D")
pt.title("Distribution variance")
num_exs = 3
for r,rep in enumerate(np.random.choice(num_reps, num_exs)):
    pt.subplot(num_exs+1,1,r+2)
    _, avg_rewards, _, grad_norms, dist_vars = results[rep]
    num_epochs = len(grad_norms)
    if (grad_norms > 0.).all(): num_epochs = grad_norms.size
    avg_rewards = avg_rewards[:num_epochs]
    dist_vars = dist_vars[:num_epochs]
    pt.plot(avg_rewards, c=fg, label="$\mathrm{\mathbb{E}}[R_0]$")
    pt.plot(normalize(dist_vars), c=bg, label="$D$/max($D$)")
    pt.ylim([0, 1.9])
    pt.legend(loc="upper left",ncol=2)
    # pt.ylabel("E[R], D/|D|")
    pt.ylabel("Trial %d" % rep)
    if r == num_exs-1: pt.xlabel("Epoch")
pt.tight_layout()
pt.savefig('reverse_dvc_examples.eps')
pt.show()


