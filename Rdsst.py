import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import *
from controller import Controller
from lvd import lvd
from reinforce import reinforce
import json 

def trials(i, avgrew, gradnorm):
    print("***************************** Trial ",str(i+1),"*******************************")
   
    letters = list('abcd')
    digits = list(map(str,range(len(letters))))
    #alpha = ["a","b","c"]
    #layer_sizes = {"rinp": 512, "rout":512, "rtemp":512}
    hidden_size = 32
    plastic = []
    #plastic = ["rtemp>rinp"]

    num_episodes = 500

    symbols = ["up","left","down","right","_","+","&"] + letters + digits
    length = getsize(max(len(symbols),32))
    layer_sizes = {"rinp": length, "rout":length, "rt1":length, "rt2":length}
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)

    
    codec = Codec(layer_sizes, symbols, rho=.999, requires_grad=False,ortho=True)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic, batch_size = num_episodes)
    ghu.associate(associations)
    
    # Initialize layers
    separator = "&"
    for k in layer_sizes.keys():
        # ghu_init.v[0][k] = codec.encode(k, separator) # !! no good anymore
        # !! Now we have to repeat the separator for each episode in the batch
        # !! v[t][k][e,:] is time t, layer k activity for episode e
        ghu.v[0][k] = tr.repeat_interleave(
            codec.encode(k, separator).view(1,-1),
            num_episodes, dim=0)

    def training_example():
        
        with open("datadsst.json", "r") as file:
	        result1 = json.load(file)
        choice = np.random.randint(1,500)
        diff = 160 - len(result1[str(choice)][0])
        if diff!=0:
        	#print("INSIDE")
        	inp = result1[str(choice)][0]
        	tar = result1[str(choice)][1]
        	for k in range(diff):
        		inp.append("&")
        		tar.append("&")
        	inputs = np.array(inp)
        	targets = np.array(tar)
        else:
            inp = result1[str(choice)][0]
            tar = result1[str(choice)][1]
        newinp = []
        for i in range(len(inp)):
        	newinp.append(inp[i])
        	newinp.append(tar[i])
        inputs = np.array(newinp) 
        targets = np.array(tar)      
        return inputs, targets

    # reward calculation based on individual steps
    def reward(ghu, targets, outputs):
        outputs_ = [o for o in outputs if o!="&"]
        targets = [t for t in targets if t!="&"]
        zeros = [o for o in outputs if o=="&"]
        totzeros = len(zeros)
        r = np.zeros(len(outputs))
        if len(outputs_)==0:
            r[-1] -= (len(outputs)+100)
        else:
            _,d = lvd(outputs_,targets) 
            for i in range(1,d.shape[0]):
                r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
            r[-1] -= 0.1*totzeros
        return r
    
    filename = "Rdsst"+str(i+1)+".png"
    # Optimization settings
    avg_rewards, grad_norms = reinforce(
        ghu,
        num_epochs = 10000,
        episode_duration = 320,
        training_example = training_example,
        reward = reward,
        task = "dsst",
        learning_rate = .005,
        verbose=1)

    gradnorm[i+1]=grad_norms.tolist()
    avgrew[i+1]=avg_rewards.tolist()

    pt.subplot(2,1,1)
    pt.plot(avg_rewards)
    pt.title("Learning curve of recall")
    pt.ylabel("Avg Reward")
    pt.subplot(2,1,2)
    pt.plot(grad_norms)
    pt.xlabel("Epoch")
    pt.ylabel("||Grad||")
    pt.savefig(filename)

    


allgradnorms = {}
allavgrewards = {}  


for i in range(1):
    trials(i,allavgrewards, allgradnorms)

with open("Rdsstavgrwd.json","w") as fp:
    json.dump(allavgrewards, fp)

with open("Rdsstgradnorm.json","w") as fp:
    json.dump(allgradnorms, fp)
