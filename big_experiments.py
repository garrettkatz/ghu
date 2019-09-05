import pickle as pk
import numpy as np
import matplotlib.pyplot as pt

def normalize(x): return (x - x.min()) / (x.max() - x.min())

# ### Recall

# # Load results
# num_reps = 6
# results = {}
# for rep in range(num_reps):
#     save_file = "results/recall/recallgk_%d.pkl" % rep
#     with open(save_file,"rb") as f:
#         results[rep] = pk.load(f)

# # Plot results
# pt.figure(figsize=(4.25,2))
# bg = (.75,.75,.75) # background color
# min_num_epochs = np.inf
# all_rewards = []
# for rep in results.keys():
#     avg_rewards, grad_norms = results[rep][1:3]
#     num_epochs = (grad_norms == 0.).argmax()
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
# pt.savefig('recallgk_learning_curves.eps')
# pt.show()

# # Distribution variance
# pt.figure(figsize=(4.25,2.25))
# _, avg_rewards, grad_norms, dist_vars = results[num_reps-1]
# num_epochs = (grad_norms == 0.).argmax()
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
# pt.savefig('recallgk_dist_var.eps')
# pt.show()


# ### Reverse

# # Load results
# num_reps = 5
# results = {}
# for rep in range(num_reps):
#     save_file = "results/reverse/reversegk_%d.pkl" % rep
#     with open(save_file,"rb") as f:
#         results[rep] = pk.load(f)

# # Plot results
# pt.figure(figsize=(4.25,2))
# bg = (.75,.75,.75) # background color
# min_num_epochs = np.inf
# all_rewards = []
# for rep in results.keys():
#     avg_rewards, grad_norms = results[rep][1:3]
#     num_epochs = (grad_norms == 0.).argmax()
#     if (grad_norms > 0.).all(): num_epochs = grad_norms.size
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
# pt.savefig('reversegk_learning_curves.eps')
# pt.show()

# # Distribution variance
# pt.figure(figsize=(4.25,3.5))
# for rep in range(2, 4):
#     pt.subplot(2,1,rep-1)
#     # if rep == 2: pt.title("Reward and Regularization")
#     _, avg_rewards, grad_norms, dist_vars = results[rep]
#     num_epochs = (grad_norms == 0.).argmax()
#     if (grad_norms > 0.).all(): num_epochs = grad_norms.size
#     avg_rewards = avg_rewards[:num_epochs]
#     dist_vars = dist_vars[:num_epochs]
#     pt.plot(normalize(dist_vars), c=bg, label="D")
#     pt.plot(normalize(avg_rewards), c=fg, label="E[R]")
#     pt.legend(loc="lower right")
#     pt.ylabel("Normalized scale")
#     if rep == 3: pt.xlabel("Epoch")
# pt.tight_layout()
# pt.savefig('reversegk_dist_var.eps')
# pt.show()

### Small Reverse

# Load results
num_reps = 5
results = {}
for rep in range(num_reps):
    save_file = "results/big_reverse/run_2000_%d.pkl" % rep
    with open(save_file,"rb") as f:
        results[rep] = pk.load(f)

# Plot results
pt.figure(figsize=(4.25,2))
bg = (.75,.75,.75) # background color
min_num_epochs = np.inf
all_rewards = []
for rep in results.keys():
    avg_rewards, grad_norms = results[rep][1:3]
    num_epochs = (grad_norms == 0.).argmax()
    if (grad_norms > 0.).all(): num_epochs = grad_norms.size
    min_num_epochs = min(min_num_epochs, num_epochs)
    all_rewards.append(avg_rewards)
    pt.plot(avg_rewards[:num_epochs], c=bg, zorder=0)
fg = 'k' # foreground color
pt.plot(
    np.array(all_rewards)[:,:min_num_epochs].mean(axis=0),
    c=fg, zorder=1, label="Average Reward over %d trials" % len(results))
# pt.title("Learning curves")
pt.ylabel("Average Reward")
pt.xlabel("Epoch")
pt.legend(loc="lower right")
pt.tight_layout()
pt.savefig('small_reverse_learning_curves.eps')
pt.show()

# Distribution variance
pt.figure(figsize=(4.25,3.5))
for rep in range(2, 4):
    pt.subplot(2,1,rep-1)
    # if rep == 2: pt.title("Reward and Regularization")
    _, avg_rewards, grad_norms, dist_vars = results[rep]
    num_epochs = (grad_norms == 0.).argmax()
    if (grad_norms > 0.).all(): num_epochs = grad_norms.size
    avg_rewards = avg_rewards[:num_epochs]
    dist_vars = dist_vars[:num_epochs]
    pt.plot(normalize(dist_vars), c=bg, label="D")
    pt.plot(normalize(avg_rewards), c=fg, label="E[R]")
    pt.legend(loc="lower right" if rep == 2 else "center right")
    pt.ylabel("Normalized scale")
    if rep == 3: pt.xlabel("Epoch")
pt.tight_layout()
pt.savefig('small_reverse_dist_var.eps')
pt.show()



