import numpy as np

class Layer:

    def __init__(self,input_size,neuron_count):
        self.weights = np.random.randn(neuron_count,input_size) * np.sqrt(2/input_size)
        self.biases = np.zeros(neuron_count)

    def forward(self,inputs):
        self.inputs=inputs
        return np.dot(self.weights,inputs) + self.biases

    def backward(self,gradient):
        self.dweights = np.outer(gradient,self.inputs)
        self.dbiases = gradient
        return np.dot(self.weights.T,gradient)

    def update(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.biases -= learning_rate * self.dbiases