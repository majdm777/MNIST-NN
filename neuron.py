import numpy as np

class Neuron:

    def __init__(self,input_size):
        self.weight = np.random.randn(input_size)
        self.bias = 0


    def forward(self,inputs):
        return np.dot(self.weight,inputs) + self.bias
        