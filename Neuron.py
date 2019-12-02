import MathAndStats as ms
import random


class Neuron:

    def __init__(self, num_inputs, is_logistic):
        self.weights = []
        self.is_logistic = is_logistic
        self.clss = ' '
        # the +1 is to have a bias node
        for i in range(num_inputs+1):
            self.weights.append(random.uniform(-0.3, 0.3))

    def setClass(self, clss):
        self.clss = clss

    def resetWeights(self):
        for i in range(len(self.weights)):
            self.weights[i] = random.uniform(-0.3, 0.3)

    def getOutput(self, new_inputs):
        # calculate the weighted linear sum
        sum = ms.weightedSum(new_inputs, self.weights, len(new_inputs))
        # if a logistic unit, return the logistic(sum)
        if self.is_logistic:
            return ms.logistic(sum)
        else:
            return sum