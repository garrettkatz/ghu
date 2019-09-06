import json
import numpy as np 
import matplotlib.pyplot as pt

def makeplot(filetoload,filesave,label,task,height=3):
    with open("data/" + filetoload, "r") as file:
        result = json.load(file)

    pt.figure(figsize=(4.25,height))
    bg = (.9,.9,.9)
    full = []

    for k,val in result.items():
        avgr = np.array(val)
        full.append(val)
        pt.plot(avgr, c=bg, zorder=0)
	
    fg = tuple([1/10.]*3)
    full=np.array(full)
    pt.plot(full.mean(axis=0), c=fg, zorder=1, label=(label))

    # pt.title("Learning curve for "+task)
    pt.ylabel("Average Reward")
    pt.xlabel("Epoch")
    pt.legend(loc="lower right")
    pt.tight_layout()
    pt.savefig(filesave)
    pt.show()



makeplot("filteravgrwd.json","filterplot.eps","Average Reward over 30 trials","filter (no repeats)",2)
makeplot("rfilteravgrwd.json","rfilterplot.eps","Average Reward over 30 trials","filter",2)
makeplot("echov2avgrwd.json","echov2plot.eps","Average Reward over 30 trials","repeated echo",2)
makeplot("maxavgrwd.json","maxplot.eps","Average Reward over 30 trials","max",2)
makeplot("reverseavgrwd.json","reverseplot.eps","Average Reward over 20 trials","reverse (no repeats)",2.5)
makeplot("recallavgrwd.json","recallplot.eps","Average Reward over 20 trials","static key-value mapping",2)
