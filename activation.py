import numpy as np

class ReLU:

    def forward(self,input):
        self.inputs=input
        return np.maximum(0,input)
    
    def backward(self, gradient):
        return gradient * (self.inputs > 0)


class Softmax:

    def forward(self, inputs):
        self.inputs = inputs
        inputs = inputs - np.max(inputs)
        exp_values = np.exp(inputs)
        self.output = exp_values / np.sum(exp_values)
        return self.output
    