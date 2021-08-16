import numpy as np

class Population:
    
    def __init__(self, components, new=True):
        self._components = components
        self.score = None
        if new:
            self.randomCreate()


    def toString(self):
        return "L1: "+str(self._components[0].val) +", L2: "+str(self._components[1].val)+", Iter: " + str(self._components[2].val) \
               +", Layers: " + str(self._components[3].val) +", Arch: " +str(self._components[3].neuronArray)


    def mutateGene(self, i):
        self._components[i].pull()


    def mutateBrain(self, a):
        self._components[len(self._components)-1].randFill(new=False,mutateRate=a)


    def germinate(self, mate, a):
        child1 = Population(np.ndarray((1,len(self._components)), dtype=Trigger)[0], new=False)
        child2 = Population(np.ndarray((1,len(self._components)), dtype=Trigger)[0], new=False)
        for i in range(0, len(self._components)):
            child1._components[i] = self._components[i].exchange(mate._components[i], self.score)
            child2._components[i] = mate._components[i].exchange(self._components[i], mate.score)
            if (a - (1-a)**len(self._components)) > np.random.rand():
                child1.mutateGene(i)
                child2.mutateGene(i)
        child1.mutateBrain(a)
        child2.mutateBrain(a)

        return child1, child2


    def randomCreate(self):
        for i in range(0, len(self._components)):
            self._components[i].pull()


class Trigger:

    # Parent class that imbues "triggering function" attribute to children classes.

    def __init__(self, fun, parameters=None):
        self._fun = fun
        self._parameters = parameters
        self.val = 0


    def pull(self):
        if self._parameters:
            self.val = self._fun(*self._parameters)
        else:
            self.val = self._fun()

        fun = getattr(self, "randFill", None)
        if callable(fun):
            self.randFill()


    def exchange(self, other, scalar):
        if not (type(self).__name__ == "NN_Layers"):
            temp = Limit(self._fun, self._parameters)
            temp.val = scalar*self.val + (1-scalar)*other.val
            if self._fun.__name__ == "randint":
                temp.val = round(temp.val)
            temp.val = min(max(temp.val, self._parameters[0]), self._parameters[1])
            return temp

        else:
            return self.neuralTransfer(other,scalar)


class Limit(Trigger):

    def __init__(self, fun, rangevec):
        super().__init__(fun, parameters=rangevec)


class NN_Layers(Trigger):
    
    def __init__(self, fun, rangevec):
        super().__init__(fun, parameters=rangevec)
        self.neurons = rangevec[len(rangevec)-1]


    def randFill(self, new=True, mutateRate=1):
        # Mutation function. If new is true and a is set to 0, equivalent to making random
        if new:
            randperm = np.random.permutation(self.val)
            self.neuronArray = np.zeros((self.val,), dtype=int)
        else:
            randperm = np.arange(self.val)

        thresh = self.neurons - self.val +1
        for i in randperm:
            if (mutateRate - (1 - mutateRate) ** self.val) > np.random.rand():
                self.neuronArray[i] = self._fun(1,thresh)
                if (thresh != 1) and new:
                    thresh = thresh - self.neuronArray[i] + 1
        if not new:
            self.neuronArray = self.shave(self.neuronArray)


    def neuralTransfer(self, nn, scalar):

        temp = NN_Layers(self._fun, self._parameters)
        if np.random.rand() > .5:

            pivot1 = min(np.random.randint(1, self.val+1), self.neurons-1)
            pivot2 = min(np.random.randint(max(1, nn.val-self.neurons+pivot1), nn.val-self._parameters[0]+pivot1+1), nn.val-1)

            neuronArray1 = self.neuronArray[0:pivot1]
            neuronArray2 = nn.neuronArray[pivot2:nn.val]
            neuronArray = np.concatenate((neuronArray1, neuronArray2))

        else:
            #This adds together
            neuronArray1 = self.neuronArray
            neuronArray2 = nn.neuronArray

            fulcrum = min(self.val, nn.val)
            if self.val == fulcrum:
                smaller = scalar * neuronArray1
                larger = (1-scalar)*neuronArray2[0:fulcrum]
            else:
                smaller = np.concatenate((neuronArray2, np.zeros(self.val - nn.val)))
                smaller = (1-scalar) * smaller
                ind = np.concatenate((np.ones(fulcrum), np.zeros(self.val - nn.val))) == 1
                nnarr = neuronArray1.astype(float)
                larger = np.multiply(nnarr, float(scalar), out=nnarr, where=ind)

            neuronArray = smaller+larger
            neuronArray = np.maximum(np.round(neuronArray), np.ones(len(neuronArray)) * self._parameters[0])

        temp.neuronArray = self.shave(neuronArray).astype(int)
        temp.val = len(temp.neuronArray)
        return temp


    def shave(self, neuronArray):
        while sum(neuronArray) > self.neurons:
            ind = np.argmax(neuronArray)
            neuronArray[ind] -= 1
        return neuronArray
