import itertools
import numpy as np
import json 
import multiprocessing as mp 


from nvm.nvm import make_scaled_nvm

def run(li, initial, final,my_nvm, register_names):
    
    br = 200
    states = {}
    for num in range(len(initial)):
        #print("initial states are", initial[num])
        regtime = {}
        for layer in my_nvm.net.layers.keys():
            #print("Checking for layer", layer)
            time = []
            for i in li:
                my_nvm.load("myfirstprogram", initial_state = initial[num])
                for t in itertools.count():  
                    my_nvm.net.tick()
                    one, two  = my_nvm.net.activity[layer].shape[0], my_nvm.net.activity[layer].shape[1]
                    #print(np.random.randn(one, two))
                    my_nvm.net.activity[layer] += (i*np.random.randn(one, two))
                    #my_nvm.net.tick()
                    #print(my_nvm.at_exit())
                    if t>br:
                        #print("Exiting the long loop")
                        #print(t)
                        #print("we failed")
                        res = "Fail"
                        break
                    if my_nvm.at_exit(): 
                        #print(t)
                        reg = my_nvm.decode_state(layer_names=register_names)
                        if reg["r0"]==final[num]["r0"] and reg["r1"]==final[num]["r1"]:
                            res= "Success"
                           
                        else:
                            res = "Fail"
                        #print(initial[num])
                        #print(final[num])
                        #print(t,res)
                        break
                
                time.append((t,res))
                #print("With noise of std dev", i)
                #print("Final register states are",my_nvm.decode_state(layer_names=register_names))
            regtime[layer] = time
        states["state"+str(num+1)]=regtime
    return states

def trial(n):
    register_names = ["r0", "r1"]

    programs = {
    "myfirstprogram":"""

    ### computes logical-and of r0 and r1, overwriting r0 with result

            nop           # do nothing
            sub and       # call logical-and sub-routine
            exit          # halt execution

    and:    cmp r0 false  # compare first conjunct to false
            jie and.f     # jump, if equal to false, to and.f label
            cmp r1 false  # compare second conjunct to false
            jie and.f     # jump, if equal false, to and.f label
            mov r0 true   # both conjuncts true, set r0 to true
            ret           # return from sub-routine
    and.f:  mov r0 false  # a conjunct was false, set r0 to false
            ret           # return from sub-routine

    """}
    orthogonality = [True, False]
    Scale = [0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.5]
    #Scale = [1]
    sd = np.linspace(0, 1, num=10)
    #vals = [1,2,3,4]
    initial = [{"r0":"true","r1":"false"}, {"r0":"true","r1":"true"}, {"r0":"false","r1":"false"}, {"r0":"false","r1":"true"}]
    final = [{"r0":"false","r1":"false"}, {"r0":"true","r1":"true"}, {"r0":"false","r1":"false"}, {"r0":"false","r1":"true"}]
    limit = 3*n + 1
    Trials = {}
    for i in range(limit - 3,limit):
        print("At trial ",i)
        Resultfinal = {}
        for o in orthogonality:
            print("For orthogonality and trial", o, i)
            scaleres = {}
            for sc in Scale:
                print("For scaling factor ", sc, o, i)
                my_nvm = make_scaled_nvm(register_names = register_names, programs = programs, orthogonal=o, scale_factor=sc)
                my_nvm.assemble(programs)
                newres = run(sd, initial, final,my_nvm, register_names)
            
                json.dump(newres, open("mid.json", "w"))
                scaleres["Scale"+" "+str(sc)] = newres
            Resultfinal[str(o)] = scaleres
        filename = "Trial" + str(i) +".json"
        json.dump(Resultfinal, open(filename, "w"))
        Trials["Trial"+str(i)]=Resultfinal
    newfile = "Trialsresult"+str(n)+".json"
    json.dump(Trials, open(newfile, 'w'))

pool = mp.Pool(mp.cpu_count())

total = [i for i in range(1,mp.cpu_count())]
print(total)
pool.map(trial, [k for k in total])
pool.close()




