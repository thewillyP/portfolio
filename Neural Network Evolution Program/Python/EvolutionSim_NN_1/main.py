# import
from scipy.io import loadmat
from os import path
import os
import numpy as np
from class_lib import *
from neuralnetwork_lib import *
import random
import copy


class Runner:
    
    def __init__(self, popSize, mutation_rate, icycles, start, neuron_lim, limits_list, parse, worldnum):
        # directories
        if popSize%4 == 1:
            print("Population size must be a multiple of 4")
            quit()

        self.dir = path.dirname(__file__)
        self.gens = path.join(self.dir, "Generations")
        self.data_name = path.join(self.dir, "data/alph_dat.mat")
        self.worldpath = path.join(self.gens, "World"+str(worldnum))

        # Create World Folder
        try:
            os.mkdir(self.worldpath)
        except FileExistsError:
            pass
        except OSError:
            print("World Folder unable to be created")
            quit()


        self.s = start
        self.savePath = path.join(self.worldpath, "generation{}.npz")

        # loading training data
        self.training = loadmat(self.data_name)['training']
        self.dat = self.training['data'][0][0]
        self.label = self.training['textdata'][0][0]
        self.tensorlabel = self.training['tensortext'][0][0]
        # loading parameters
        self.size = popSize
        self.keep = round(self.size*.1)
        self.a = mutation_rate
        self.icyc = icycles
        self.lims = limits_list
        self.neurons = neuron_lim
        #load the fitness trainer
        self.workout_trainer = NN_Manager(self.dat, self.label, self.tensorlabel, parse)


    def load_pop(self):
        try:
            self.pop = np.load(self.worldpath+"/generation"+ str(self.s-1)+".npz", allow_pickle=True)
            self.pop = self.pop["Population"]
        except:
            self.pop = np.array([], dtype=Population)


    def randPop(self):
        self.pop = np.ndarray((1,self.size),dtype=Population)[0]
        for j in range(0,self.size):
            self.pop[j] = Population(copy.deepcopy(self.lims))


    def startTrain(self, i):
        # trains the scores(ranks from worst to best) the population
        score = np.vectorize(self.workout_trainer.train, otypes=[float])(self.pop)
        ind = np.argsort(score)
        self.score = np.sort(score, kind="mergesort")
        self.pop = np.take_along_axis(self.pop, ind, axis=0)
        self.pop = np.vectorize(self.assignScore, otypes=[Population])(self.pop, self.score)

        # Saves and announces scores
        print("Generation"+str(i)+" Score: ",self.score)
        np.savez(self.savePath.format(i), Population=self.pop, Score= self.score)


    def assignScore(self, pop, score):
        pop.score = score
        return pop


    def selection(self):
        # Culls off half the pool
        half = int(self.size/2)
        underPreformCull = np.random.permutation(half)[0:(half-self.keep)]
        betterPreformCull = (np.random.permutation(half) + half)[0:self.keep]
        cull = np.concatenate((underPreformCull,betterPreformCull))
        self.pop = np.delete(self.pop, cull)


    def cross_mut(self):
        # Repopulates survivors, mutate children
        mid = int(len(self.pop)/2)
        half1, half2 = np.split(np.random.permutation(len(self.pop)),[mid])
        half1 = np.take_along_axis(self.pop, half1, axis=0)
        half2 = np.take_along_axis(self.pop,half2, axis=0)

        children = np.vectorize(self.combine, otypes=[Population])(half1,half2)
        self.pop = np.concatenate((self.pop,np.concatenate(children)))


    def combine(self, pop1, pop2):
        return pop1.germinate(pop2, self.a)


    def startSimulation(self):
        for i in range(self.s, self.icyc + 1):
            runner.startTrain(i)
            runner.selection()
            runner.cross_mut()


#random uniform modification
def randUniform(lower, higher):

    pool = int(np.log10(round(higher / lower)))
    chance =np.zeros(pool)
    for i in range(1, pool + 1):
        lower = higher * (10 ** -i)
        chance[i-1] = np.random.uniform(lower, higher* (10 ** (-i+1)))


    return chance[np.random.randint(pool)]


# Defining the parameters of Neural network
neurons = 50

lambda_fun = randUniform
lambda_range = [.000001,.001]
lamK = Limit(lambda_fun, lambda_range)
lamB = Limit(lambda_fun, lambda_range)

iter_fun = np.random.randint
iter_range = [10,11]
it = Limit(iter_fun,iter_range)

layer_fun = np.random.randint
layer_range = [2,neurons]
lay = NN_Layers(layer_fun,layer_range)

limit_list = np.array([lamK, lamB, it, lay])

# Make and start runner
""" Population size, mutation rate, number of generational cycles, which generation to start, 
    number of neurons max, list of genes, train-test split, Which evolution world it takes place"""
runner = Runner(100, .3, 100, 45, neurons, limit_list, 16000, 4)
already_running = True
if already_running:
    runner.load_pop()
else:
    runner.randPop()

runner.startSimulation()