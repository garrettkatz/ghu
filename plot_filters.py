import json
import numpy as np 
import matplotlib.pyplot as pt

pt.figure(figsize=(4.25,5.))
bg = (.9,.9,.9)
fg = (.1,.1,.1)

variant = ["max","rfilter","filter"]
titles = ["Max", "Cut-off", "Cut-off (no repeats)"]

for sp, vari in enumerate(variant):

    # filetoload = vari+"avggen.json"
    filetoload = vari+"avgrwd.json"
    
    full = {}
    with open("data/" + filetoload, "r") as file:
        result = json.load(file)
    full[sp] = []
    for k,val in result.items():
        avgr = np.array(val)
        full[sp].append(val)
    full[sp]=np.array(full[sp])
    
    num_reps = len(full[sp])

    pt.subplot(len(variant),1,sp+1)
    if sp == 0: pt.title("Testing set rewards")
    pt.plot(full[sp].T, c=bg, zorder=0)
    pt.plot(full[sp].mean(axis=0), c=fg, zorder=1, label="Avg. over %d trials" % num_reps)
    pt.legend(loc="lower right")
    pt.ylabel(titles[sp])
    
pt.tight_layout()
pt.savefig("filter_curves.eps")
# pt.show()
    
