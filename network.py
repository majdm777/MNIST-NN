import numpy as np
from layer import Layer
from activation import ReLU, Softmax

class Network :

    def __init__(self):
        self.layer1 = Layer(784,128)
        self.relu1 = ReLU()

        self.layer2 = Layer(128,10)
        self.softmax = Softmax()


    def predict(self,image):
        prediction = self.forward(image)
        return np.argmax(prediction)    
        
    def forward(self,inputs):
        x = self.layer1.forward(inputs)
        x = self.relu1.forward(x)
        x=self.layer2.forward(x)
        x = self.softmax.forward(x)
        return x
    
    def backward(self,gradient):
        x = self.layer2.backward(gradient)
        x = self.relu1.backward(x)
        x = self.layer1.backward(x)

        return x
    
    def update(self, learning_rate):

        self.layer1.update(learning_rate)
        self.layer2.update(learning_rate)