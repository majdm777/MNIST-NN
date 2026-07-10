import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from network import Network
from loss import CrossEntropy

print("Collecting data...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target 

# normalization [0,255] --> [0,1]

def normalize(Data):
    return Data / 255



# DATA
print("Data collected!")
X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]

y_train = y_train.astype(int)
y_test = y_test.astype(int)

print(X_train.shape)
print(X_test.shape)

X_test= normalize(X_test)
X_train = normalize(X_train)

# Create network and loss
network = Network()
loss_function = CrossEntropy()


learning_rate = 0.001
epochs = 5

print("running traing phase...")
for epoch in range(epochs):

    total_loss = 0

    for image, label in zip(X_train, y_train):

        # Forward
        prediction = network.forward(image)

        # Loss
        loss = loss_function.forward(prediction, label)

        total_loss += loss

        # Backward
        gradient = loss_function.backward()

        network.backward(gradient)

        # Update
        network.update(learning_rate)


    average_loss = total_loss / len(X_train)

    print(
        "Epoch:",
        epoch + 1,
        "Loss:",
        average_loss
    )

    correct = 0

print("training phase finished! ")
print("running testing phase... ")
for image, label in zip(X_test[:1000], y_test[:1000]):

    prediction = network.forward(image)

    predicted_label = np.argmax(prediction)

    if predicted_label == label:
        correct += 1

print("testing phase finished! ")
accuracy = correct / 1000

print("Accuracy:", accuracy)


for i in range(10):
    prediction = network.predict(X_test[i])

    print(
        "Predicted:",
        prediction,
        "Actual:",
        y_test[i]
    )


os.makedirs("Model", exist_ok=True)
print("saving data ...")
np.save("Model/layer1_weights.npy", network.layer1.weights)
np.save("Model/layer2_weights.npy", network.layer2.weights)
print("Done!")