import json
import numpy as np 
import matplotlib.pyplot as pt

pt.figure(figsize=(4.25,3.))
bg = (.9,.9,.9)

# filenames = ["filteravgrwd.json", "rfilteravgrwd.json"]
filenames = ["filteravgrwd.json", "filteravggen.json"]

full = {}
for sp,filetoload in enumerate(filenames):

    with open("results/" + filetoload, "r") as file:
        result = json.load(file)
    full[sp] = []
    for k,val in result.items():
        avgr = np.array(val)
        full[sp].append(val)
    full[sp]=np.array(full[sp])

num_reps = len(full[1])
fg = tuple([1/10.]*3)
pt.subplot(2,1,1)
pt.plot(full[1].T, c=bg, zorder=0)
pt.plot(full[1].mean(axis=0), c=fg, zorder=1, label="Avg. over %d trials" % num_reps)
pt.legend(loc="lower right")
pt.ylabel("Test Rewards")

pt.subplot(2,1,2)
pt.plot((full[0] - full[1]).T, c=bg, zorder=0)
pt.plot((full[0]-full[1]).mean(axis=0), c=fg, zorder=1) #, label="Avg. over %d trials" % num_reps)
# pt.legend(loc="upper right")
pt.ylabel("Train - Test")
pt.xlabel("Epoch")

pt.tight_layout()
pt.savefig("filter_curves.eps")
pt.show()

