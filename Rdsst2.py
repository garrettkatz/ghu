import numpy as np
import torch as tr
import matplotlib.pyplot as pt
from ghu import *
from codec import *
from controller import Controller
from lvd import lvd
from reinforce import reinforce
import json 

def trials(i, avgrew, gradnorm, save_file):
    print("***************************** Trial ",str(i+1),"*******************************")
   
    letters = list('abcd')
    digits = list(map(str,range(len(letters))))
    plastic = ["rinp<rtmp"]
    remove_pathways = ["rinp<rout", "rout<rtmp"]
    num_episodes = 15000
    #num_episodes = 500
    hidden_size=32

    num_symbols = 3
    symbols = "abcd"[:num_symbols]  + "0_"+ "1234"[:num_symbols]
    length = getsize(max(len(symbols),32))
    layer_sizes = {"rinp": length, "rout":length, "rt1":length, "rt2":length, "m":length}
    pathways, associations = turing_initializer2( # all to all
        list(layer_sizes.keys()), symbols)

    codec = Codec(layer_sizes, symbols, rho=.999, requires_grad=False,ortho=True)
    controller = Controller(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = GatedHebbianUnit(layer_sizes, pathways, controller, codec, plastic=plastic, batch_size = num_episodes)
    ghu.associate(associations)
    
    # Initialize layers
    separator = "0"
    ghu.fill_layers(separator)

    # def make_dsst_grid(letters, rows, cols):

    #     grid = [["+" for c in range(cols+2)] for r in range(2*rows+2)]
    #     pair = {}
    #     for r in range(rows):
    #         for c in range(cols):
    #             if r == 0 and c < len(letters):
    #                 grid[2*r+1][c+1] = letters[c]
    #                 grid[2*r+2][c+1] = str(c)
    #                 pair.update({letters[c]:str(c)})
    #             else:
    #                 grid[2*r+1][c+1] = letters[np.random.randint(len(letters))]
    #                 grid[2*r+2][c+1] = "_"

    #     result = [["+" for c in range(cols+2)] for r in range(2*rows+2)]
    #     for r in range(rows):
    #         for c in range(cols):
    #             if r == 0 and c < len(letters):
    #                 result[2*r+1][c+1] = letters[c]
    #                 result[2*r+2][c+1] = str(c)
    #             else:

    #                 result[2*r+1][c+1] = grid[2*r+1][c+1]
    #                 result[2*r+2][c+1] = pair[grid[2*r+1][c+1]]
    #     print(grid)
    #     print(result)

    #     #print(newgrid)
    #     # print(grid)
    #     # print(pair)
    #     # print(result)
    #     return grid, result




    # def training_example():
    #     letters = list('abcd')
    #     #make_dsst_grid(letters, 2, len(letters))
    #     grid, ansgrid = make_dsst_grid(letters, 2, len(letters))
    #     inp, tar = [],[]
    #     for item in grid:
    #         for i in item:
    #             inp.append(i)
    #     for item in ansgrid:
    #         for i in item:
    #             tar.append(i)
    #     inputs = np.array(inp) #np.array(act+inp)   
    #     targets = np.array(tar) #np.array(["&","&","&","&"]+tar)
    #     #print(len(inputs),len(targets))  
    #     return inputs, targets

    def train():
        keys = list("abc"[:num_symbols])
        vals = list("123"[:num_symbols])
        i1,i2,t2 = [],[],[]
        for i in range(len(keys)):
            i1.append(keys[i])
            i1.append(vals[i])
        for i in range(len(keys)):
            k = np.random.randint(num_symbols)
            i2.append(keys[k])
            t2.append(keys[k])
            i2.append("_")
            t2.append(vals[k])
        inputs = i1 +i2
        targets = i1+t2
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
        wrong = 0
        for i in [1,3,5]:
            if outputs[-i].isalpha():
                wrong +=1
        r[-1] -= wrong*2
        return r

    
    filename = "Rdsst2"+str(i+1)+".png"
    # Optimization settings
    avg_rewards, avg_general, grad_norms = reinforce(
        ghu,
        num_epochs = 8000,
        episode_duration = 12,
        training_example = train,
        testing_example = None,
        reward = reward,
        task = "dsst",
        learning_rate = .01,
        distribution_variance_coefficient = .05,
        # choices=correct_choices, # perfect rewards with this
        verbose = 1,
        save_file = save_file)

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
    trials(i,allavgrewards, allgradnorms, "Rdsst2.pkl")

with open("Rdsst2avgrwd.json","w") as fp:
    json.dump(allavgrewards, fp)

with open("Rdsst2gradnorm.json","w") as fp:
    json.dump(allgradnorms, fp)

