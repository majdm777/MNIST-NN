# MNIST Neural Network From Scratch

A handwritten digit recognition system built from scratch using **Python and NumPy**.
This project implements a simple neural network without using deep learning frameworks like TensorFlow or PyTorch.

The model is trained on the **MNIST handwritten digits dataset** and can recognize handwritten digits drawn by the user through a custom GUI.

---

## Project Overview

The goal of this project is to understand the internal workings of neural networks by implementing the main components manually:

* Forward propagation
* Activation functions
* Softmax classification
* Cross-entropy loss
* Backpropagation
* Gradient descent optimization
* Model saving/loading
* Handwritten digit prediction

---

## Features

* ✅ Neural network implemented from scratch using NumPy
* ✅ Dense (fully connected) layers
* ✅ ReLU activation function
* ✅ Softmax output layer
* ✅ Cross-entropy loss
* ✅ Backpropagation implementation
* ✅ Gradient descent training
* ✅ MNIST digit classification
* ✅ Model weight saving/loading
* ✅ Drawing interface for testing custom handwriting

---

## Neural Network Architecture

The implemented network:

```
Input Image
(784 pixels)
      |
      v
Dense Layer
784 → 128 neurons
      |
      v
ReLU Activation
      |
      v
Dense Layer
128 → 10 neurons
      |
      v
Softmax
      |
      v
Digit Probabilities
(0 - 9)
```

---

## Dataset

The project uses the **MNIST dataset**:

* 70,000 grayscale images
* Image size: 28 × 28 pixels
* 60,000 training images
* 10,000 testing images
* 10 classes (digits 0-9)

Each image is converted into:

```
28 × 28 = 784 input values
```

Pixel values are normalized:

```
Original:
0 - 255

After normalization:
0 - 1
```

---

## Project Structure

```
MNIST-NN/

│
├── main.py                 # Train the neural network
├── draw.py                 # GUI for handwritten digit prediction
│
├── network.py              # Neural network architecture
├── layer.py                # Dense layer implementation
├── activation.py           # ReLU and Softmax functions
├── loss.py                 # Cross entropy loss
│
├── model/
│   ├── layer1_weights.npy  # Saved weights
│   └── layer2_weights.npy
│
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install numpy matplotlib pillow scikit-learn pandas
```

---

## Training the Model

Run:

```bash
python main.py
```

The training process:

1. Loads MNIST dataset
2. Normalizes images
3. Creates the neural network
4. Performs forward propagation
5. Calculates loss
6. Performs backpropagation
7. Updates weights
8. Saves trained weights

After training, the model weights are stored in:

```
model/
```

---

## Training Results

Example training results:

```
Epoch: 1 Loss: 0.414
Epoch: 2 Loss: 0.233
Epoch: 3 Loss: 0.187
Epoch: 4 Loss: 0.157
Epoch: 5 Loss: 0.136
```

Testing accuracy:

```
Accuracy: ~95%
```

---

## Testing Your Own Handwriting

After training, run:

```bash
python draw.py
```

A drawing window will appear.

Steps:

1. Draw a digit using the mouse
2. Press the **Predict** button
3. The program:

   * Crops empty space
   * Resizes the digit
   * Centers it
   * Converts it to MNIST format
   * Runs prediction

Example:

```
Prediction: 7
```

---

## How Backpropagation Works

The network learns by calculating how much each weight contributes to the error.

The process:

```
Prediction
    |
    v
Loss Calculation
    |
    v
Backward Propagation
    |
    v
Gradient Calculation
    |
    v
Weight Update
```

Weights are updated using:

```
new_weight =
old_weight - learning_rate * gradient
```

---

## Technologies Used

* Python
* NumPy
* Matplotlib
* Pillow
* Tkinter
* Scikit-learn

---

## Why This Project?

This project was built to understand neural networks at a lower level instead of relying on existing frameworks.

By implementing the main components manually, it provides a deeper understanding of:

* How neurons work
* How weights are updated
* How errors propagate backward
* How models learn from data

---

## Future Improvements

Possible improvements:

* Mini-batch gradient descent
* Better optimizers (Momentum, Adam)
* More hidden layers
* Dropout regularization
* Convolutional Neural Network (CNN)
* Data augmentation for handwritten input
* Export model as a standalone application

---

## License

This project is open-source and available for learning purposes.
