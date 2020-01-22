import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

def normalize(x): return (x - x.min()) / (x.max() - x.min())

# ### Recall

# # Load results
# num_reps = 21
# results = {}
# for rep in range(num_reps):
#     save_file = "results/recall/new/run_5000_%d.pkl" % rep
#     with open(save_file,"rb") as f: results[rep] = pk.load(f)

# # Plot results
# pt.figure(figsize=(4.25,2))
# bg = (.75,.75,.75) # background color
# min_num_epochs = np.inf
# all_rewards = []
# for rep in results.keys():
#     avg_rewards, grad_norms = results[rep][1], results[rep][3]
#     num_epochs = len(grad_norms)
#     if (grad_norms > 0.).all(): num_epochs = grad_norms.size
#     if (avg_rewards[num_epochs-10:num_epochs] > 0.95).all():
#         avg_rewards[num_epochs:] = avg_rewards[num_epochs-1]
#         num_epochs = grad_norms.size
#     min_num_epochs = min(min_num_epochs, num_epochs)
#     all_rewards.append(avg_rewards)
#     pt.plot(avg_rewards[:num_epochs], c=bg, zorder=0)
# fg = 'k' # foreground color
# pt.plot(
#     np.array(all_rewards)[:,:min_num_epochs].mean(axis=0),
#     c=fg, zorder=1, label="Average Reward over %d trials" % len(results))
# # pt.title("Learning curves")
# pt.legend(loc="lower right")
# pt.ylabel("Average Reward")
# pt.xlabel("Epoch")
# pt.tight_layout()
# # pt.savefig('recall_plastic_learning_curves.eps')
# pt.show()

# # Distribution variance
# pt.figure(figsize=(4.25,2.25))
# _, avg_rewards, _, grad_norms, dist_vars = results[num_reps-1]
# num_epochs = len(grad_norms)
# if (grad_norms > 0.).all(): num_epochs = grad_norms.size
# avg_rewards = avg_rewards[:num_epochs]
# dist_vars = dist_vars[:num_epochs]
# pt.plot(normalize(dist_vars), c=bg, label="D")
# pt.plot(normalize(avg_rewards), c=fg, label="E[R]")
# pt.legend(loc="center right")
# pt.title("Reward and Regularization terms")
# pt.ylabel("(X-min X)/(max X-min X)")
# pt.xlabel("Epoch")
# pt.tight_layout()
# # pt.savefig('recall_plastic_dist_var.eps')
# pt.show()


### Reverse

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

# # Plot results
# pt.figure(figsize=(4.25,3.5))
# min_num_epochs = np.inf
# pt.subplot(2,1,1)
# for r,rep in enumerate(results.keys()):
#     pt.plot(all_rewards[r], c=bg, zorder=0)
# pt.plot(np.array(all_rewards).mean(axis=0),
#     c=fg, zorder=1, label="Average of %d trials" % len(results))
# # pt.title("Learning curves")
# pt.legend(loc="upper left")
# pt.ylabel("Average Reward")
# pt.xlabel("Epoch")

# pt.subplot(2,1,2)
# for r,rep in enumerate(results.keys()):
#     pt.plot(all_dvcs[r], c=bg, zorder=0)
# pt.plot(np.array(all_dvcs).mean(axis=0),
#     c=fg, zorder=1, label="Average of %d trials" % len(results))
# # pt.title("Learning curves")
# pt.legend(loc="upper left")
# pt.ylabel("Dist. Variance")
# pt.xlabel("Epoch")
# pt.tight_layout()
# # pt.savefig('reverse_with_repeats_learning_curves.eps')
# # pt.show()

# Perf vs dist var
pt.figure(figsize=(4.25,2.25))
# pt.scatter(np.array(all_rewards).flatten(), np.array(all_dvcs).flatten())
# pt.scatter(np.array(all_rewards)[:,-1], np.array(all_dvcs)[:,-1])
x = (np.array(all_rewards)[:,1:] - np.array(all_rewards)[:,:-1]).flatten()
y = (np.array(all_dvcs)[:,1:] - np.array(all_dvcs)[:,:-1]).flatten()
pt.scatter(x, y, c='k',marker='+')
# pt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), 'k--')
xlo, xhi = 1.1*x.min(), 1.1*x.max()
ylo, yhi = 1.1*y.min(), 1.1*y.max()
pt.plot([xlo, xhi], [0, 0], 'k--')
pt.plot([0, 0], [ylo, yhi], 'k--')
pt.xlim([xlo,xhi])
pt.ylim([ylo,yhi])
pt.xlabel("$\Delta$ E[R]")
pt.ylabel("$\Delta$ D")
pt.tight_layout()
pt.savefig('reverse_dvc_deltas.eps')
pt.show()

# Distribution variance
pt.figure(figsize=(4.25,5.00))
pt.subplot(4,1,1)
for r,rep in enumerate(results.keys()):
    pt.plot(all_dvcs[r], c=bg, zorder=0)
pt.plot(np.array(all_dvcs).mean(axis=0),
    c=fg, zorder=1, label="Avg. of %d trials" % len(results))
# pt.title("Learning curves")
pt.legend(loc="upper left")
pt.ylabel("D")
pt.title("Distribution variance")
# pt.xlabel("Epoch")
# pt.figure()
# for rep in range(2, 4):
#     pt.subplot(2,1,rep-1)
# for rep in range(num_reps):
#     pt.subplot(10,3,rep+1)
num_exs = 3
for r,rep in enumerate(np.random.choice(num_reps, num_exs)):
    pt.subplot(num_exs+1,1,r+2)
    # if r == 0: pt.title("Representative examples")
    # if rep == 2: pt.title("Reward and Regularization")
    _, avg_rewards, _, grad_norms, dist_vars = results[rep]
    num_epochs = len(grad_norms)
    if (grad_norms > 0.).all(): num_epochs = grad_norms.size
    avg_rewards = avg_rewards[:num_epochs]
    dist_vars = dist_vars[:num_epochs]
    # pt.plot(normalize(dist_vars), c=bg, label="D")
    # pt.plot(normalize(avg_rewards), c=fg, label="E[R]")
    pt.plot(avg_rewards, c=fg, label="E[R]")
    pt.plot(normalize(dist_vars), c=bg, label="D/max(D)")
    pt.legend(loc="center left",ncol=1)
    # pt.ylabel("E[R], D/|D|")
    pt.ylabel("Trial %d" % rep)
    if r == num_exs-1: pt.xlabel("Epoch")
pt.tight_layout()
pt.savefig('reverse_dvc_examples.eps')
# pt.savefig('reverse_with_repeats_dist_var.eps')
pt.show()
    

# ### Small Reverse

# # Load results
# num_reps = 5
# results = {}
# for rep in range(num_reps):
#     save_file = "results/big_reverse/run_2000_%d.pkl" % rep
#     with open(save_file,"rb") as f:
#         results[rep] = pk.load(f)

# # Plot results
# pt.figure(figsize=(4.25,2))
# bg = (.75,.75,.75) # background color
# min_num_epochs = np.inf
# all_rewards = []
# for rep in results.keys():
#     avg_rewards, grad_norms = results[rep][1:3]
#     num_epochs = len(grad_norms)
#     if (grad_norms > 0.).all(): num_epochs = grad_norms.size
#     min_num_epochs = min(min_num_epochs, num_epochs)
#     all_rewards.append(avg_rewards)
#     pt.plot(avg_rewards[:num_epochs], c=bg, zorder=0)
# fg = 'k' # foreground color
# pt.plot(
#     np.array(all_rewards)[:,:min_num_epochs].mean(axis=0),
#     c=fg, zorder=1, label="Average Reward over %d trials" % len(results))
# # pt.title("Learning curves")
# pt.ylabel("Average Reward")
# pt.xlabel("Epoch")
# pt.legend(loc="lower right")
# pt.tight_layout()
# pt.savefig('small_reverse_learning_curves.eps')
# pt.show()

# # Distribution variance
# pt.figure(figsize=(4.25,3.5))
# for rep in range(2, 4):
#     pt.subplot(2,1,rep-1)
#     # if rep == 2: pt.title("Reward and Regularization")
#     _, avg_rewards, grad_norms, dist_vars = results[rep]
#     num_epochs = len(grad_norms)
#     if (grad_norms > 0.).all(): num_epochs = grad_norms.size
#     avg_rewards = avg_rewards[:num_epochs]
#     dist_vars = dist_vars[:num_epochs]
#     pt.plot(normalize(dist_vars), c=bg, label="D")
#     pt.plot(normalize(avg_rewards), c=fg, label="E[R]")
#     pt.legend(loc="lower right" if rep == 2 else "center right")
#     pt.ylabel("Normalized scale")
#     if rep == 3: pt.xlabel("Epoch")
# pt.tight_layout()
# pt.savefig('small_reverse_dist_var.eps')
# pt.show()



