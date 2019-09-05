import json
import numpy as np 
import matplotlib.pyplot as pt

pt.figure(figsize=(4.25,3.5))
bg = (.9,.9,.9)

filenames = ["filteravgrwd.json", "rfilteravgrwd.json"]

for sp,filetoload in enumerate(filenames):

    with open("data/" + filetoload, "r") as file:
        result = json.load(file)
    full = []

    pt.subplot(2,1,sp+1)
    for k,val in result.items():
        avgr = np.array(val)
        full.append(val)
        pt.plot(avgr, c=bg, zorder=0)
	
    fg = tuple([1/10.]*3)
    full=np.array(full)
    pt.plot(full.mean(axis=0), c=fg, zorder=1, label="Average reward over 30 trials")
    pt.legend(loc="lower right")
    pt.title("No repeats" if sp == 0 else "With repeats")
    pt.ylabel("Average Reward")

pt.xlabel("Epoch")
pt.tight_layout()
pt.savefig("both_filter_curves.eps")
pt.show()

