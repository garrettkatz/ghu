"""
Echo input (rinp) at output (rout)
"""
import numpy as np
import torch as tr
import matplotlib.pyplot as pt
import pickle as pk
from ghu import *
from codec import *
from controller import *
from supervised import supervise

def trials(i, avgrew, gradnorm):
    print("***************************** supervised multiecho*******************************")
    
    # GHU settings
    num_symbols =6
    
    hidden_size = 24
    plastic = []
    num_episodes=200

    symbols = [str(a) for a in range(num_symbols)]
    length = max(getsize(len(symbols)),32)
    layer_sizes = {"rinp": length, "rout":length}
    pathways, associations = default_initializer( # all to all
        layer_sizes.keys(), symbols)
    codec = Codec(layer_sizes, symbols, rho=.9, requires_grad = True, ortho = True)
    controller = SController(layer_sizes, pathways, hidden_size, plastic)

    # Sanity check
    ghu = SGatedHebbianUnit(
        layer_sizes, pathways, controller, codec, plastic=plastic, batch_size=num_episodes)
    ghu.associate(associations)

    # Initialize layers
    separator = "0"
    for k in layer_sizes.keys():
        # ghu_init.v[0][k] = codec.encode(k, separator) # !! no good anymore
        # !! Now we have to repeat the separator for each episode in the batch
        # !! v[t][k][e,:] is time t, layer k activity for episode e
        ghu.v[0][k] = tr.repeat_interleave(
            codec.encode(k, separator).view(1,-1),
            num_episodes, dim=0)

    # training example generation
    def training_example():
        # Randomly choose echo symbol (excluding 0 separator)
        inputs = np.random.choice(symbols[1:], size=1)
        targets = [inputs[0] for i in range(int(inputs[0]))]
        for _ in range(5-int(inputs[0])):
            targets.append("0")
        return inputs, targets
    
   
    def sloss(pred,y):
        # if tr.abs(tr.mean(pred-y))<1:
        loss = (tr.mean(tr.pow(pred-y, 2.0)))
        # else:
        #     loss = (tr.abs(tr.mean(pred-y))-0.5)
        return loss

    loss = supervise(ghu,
        num_epochs = 1200,
        training_example = training_example,
        task = "echov2",
        episode_len=5,
        loss_fun = sloss,
        learning_rate = .1,
        Optimizer = tr.optim.SGD,
        verbose = 1,
        save_file = "echov2.pkl")
    
    with open("echov2.pkl","rb") as f:
        config, loss = pk.load(f)

    print(config)
    print(loss[-10:])
    #print(grad_norms[-10:])
    
    
    pt.plot(loss)
    pt.title("Learning curve")
    pt.ylabel("Loss")

    pt.xlabel("Epoch")
    #pt.tight_layout()
    pt.show()


allgradnorms = {}
allavgrewards = {}  


for i in range(1):
    trials(i,allavgrewards, allgradnorms)

# with open("echov2avgrwd.json","w") as fp:
#     json.dump(allavgrewards, fp)

# with open("echov2gradnorm.json","w") as fp:
#     json.dump(allgradnorms, fp)


