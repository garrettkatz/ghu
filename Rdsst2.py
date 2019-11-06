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

    symbols = ["_","+","&"] + letters + digits
    length = getsize(max(len(symbols),32))
    layer_sizes = {"rinp": length, "rout":length, "rt1":length, "rt2":length}
    pathways, associations = default_initializer( # all to all
        list(layer_sizes.keys()), symbols)

    #associations= associations + [("rt2<rt1","0","up"),("rt1<rt2","1","left"),("rt2<rout","2","down"),("rt1<rout","3","right")]
    
    # for i in range(len(associations)):
    #     associations[i]=list(associations[i])
    #     if associations[i][0]=="rt2<rt1" and associations[i][1]=="0":
    #         associations[i][2]="left"
    #     if associations[i][0]=="rt1<rt2" and associations[i][1]=="1":
    #         associations[i][2]="up"
    #     if associations[i][0]=="rt2<rout" and associations[i][1]=="2":
    #         associations[i][2]="down"
    #     if associations[i][0]=="rt1<rout" and associations[i][1]=="3":
    #         associations[i][2]="right"
    #     associations[i] = tuple(associations[i])
    #print("NEW associations", associations)
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

    def make_dsst_grid(letters, rows, cols):

        grid = [["+" for c in range(cols+2)] for r in range(2*rows+2)]
        pair = {}
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c < len(letters):
                    grid[2*r+1][c+1] = letters[c]
                    grid[2*r+2][c+1] = str(c)
                    pair.update({letters[c]:str(c)})
                else:
                    grid[2*r+1][c+1] = letters[np.random.randint(len(letters))]
                    grid[2*r+2][c+1] = "_"

        result = [["+" for c in range(cols+2)] for r in range(2*rows+2)]
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c < len(letters):
                    result[2*r+1][c+1] = letters[c]
                    result[2*r+2][c+1] = str(c)
                else:

                    result[2*r+1][c+1] = grid[2*r+1][c+1]
                    result[2*r+2][c+1] = pair[grid[2*r+1][c+1]]

        #print(newgrid)
        # print(grid)
        # print(pair)
        # print(result)
        return grid, result




    def training_example():
        letters = list('abcd')
        #make_dsst_grid(letters, 2, len(letters))
        grid, ansgrid = make_dsst_grid(letters, 2, len(letters))
        inp, tar = [],[]
        for item in grid:
            for i in item:
                inp.append(i)
        for item in ansgrid:
            for i in item:
                tar.append(i)
        inputs = np.array(inp) #np.array(act+inp)   
        targets = np.array(tar) #np.array(["&","&","&","&"]+tar)
        #print(len(inputs),len(targets))  
        return inputs, targets

    # reward calculation based on individual steps
    def reward(ghu, targets, outputs):
        # outputs_ = [o for o in outputs if o!="&"]
        # targets = [t for t in targets if t!="&"]
        # zeros = [o for o in outputs if o=="&"]
        # totzeros = len(zeros)
        # r = np.zeros(len(outputs))
        # if len(outputs_)==0:
        #     r[-1] -= (len(outputs)+100)
        # else:
        #     _,d = lvd(outputs_,targets) 
        #     for i in range(1,d.shape[0]):
        #         r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        #     r[-1] -= 0.1*totzeros
        # return r
        fix = [o for o in outputs if o!="_"]
        blanks = [o for o in outputs if o=="_"]
        r = np.zeros(len(outputs))
        _,d = lvd(outputs,targets) 
        for i in range(1,d.shape[0]):
            r[-1] += 1. if (i < d.shape[1] and d[i,i] == d[i-1,i-1]) else -1.
        if len(set(fix))==1:
        	r[-1]-=200
        r[-1] -= 10*len(blanks) 
        return r

    
    filename = "Rdsst2"+str(i+1)+".png"
    # Optimization settings
    avg_rewards, grad_norms = reinforce(
        ghu,
        num_epochs = 8000,
        episode_duration = 36,
        training_example = training_example,
        reward = reward,
        task = "dsst",
        learning_rate = .001,
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

with open("Rdsst2avgrwd.json","w") as fp:
    json.dump(allavgrewards, fp)

with open("Rdsst2gradnorm.json","w") as fp:
    json.dump(allgradnorms, fp)

