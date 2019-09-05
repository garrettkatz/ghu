import json
import numpy as np 
import matplotlib.pyplot as pt

def makeplot(filetoload,filesave,label,task):
    with open(filetoload, "r") as file:
        result = json.load(file)

    pt.figure(figsize=(4.25,3))
    bg = (.9,.9,.9)
    full = []

    for k,val in result.items():
        avgr = np.array(val)
        full.append(val)
        pt.plot(avgr, c=bg, zorder=0)
	
    fg = tuple([1/10.]*3)
    full=np.array(full)
    pt.plot(full.mean(axis=0), c=fg, zorder=1, label=(label))

    pt.title("Learning curve for "+task)
    pt.ylabel("Average Reward")
    pt.xlabel("Epoch")
    pt.legend(loc="lower right")
    pt.tight_layout()
    pt.savefig(filesave)
    pt.show()

makeplot("filteravgrwd.json","filterplot.png","Average Reward over 30 trials","Filter")
makeplot("rfilteravgrwd.json","rfilterplot.png","Average Reward over 30 trials","Filter with repeat")
makeplot("echov2avgrwd.json","echov2plot.png","Average Reward over 30 trials","Multi - Echo")
makeplot("maxavgrwd.json","maxplot.png","Average Reward over 30 trials","Max")
makeplot("reverseavgrwd.json","reverseplot.png","Average Reward over 20 trials","Reverse without repeat")
makeplot("recallavgrwd.json","recallplot.png","Average Reward over 20 trials","Recall no plastic")