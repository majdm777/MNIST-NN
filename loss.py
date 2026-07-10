import numpy as np

class CrossEntropy:

    def forward(self, prediction, true_label):

        self.prediction = prediction
        self.true_label = true_label

        epsilon = 1e-7
        loss = -np.log(prediction[true_label] + epsilon)

        return loss


    def backward(self):

        gradient = self.prediction.copy()

        gradient[self.true_label] -= 1

        return gradient