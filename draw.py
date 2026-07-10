import tkinter as tk
import numpy as np
import os
from PIL import Image

from network import Network


# =========================
# Load trained model
# =========================

layer1_path = "Model/layer1_weights.npy"
layer2_path = "Model/layer2_weights.npy"


if not os.path.exists(layer1_path) or not os.path.exists(layer2_path):
    print("Model not found!")
    print("Please run main.py first to train the network.")
    exit()


network = Network() 

network.layer1.weights = np.load(layer1_path)
network.layer2.weights = np.load(layer2_path)

print("Model loaded successfully!")


# =========================
# Drawing settings
# =========================

canvas_size = 280

# Store drawing as grayscale image
# 0 = black, 255 = white
image = np.zeros((canvas_size, canvas_size), dtype=np.uint8)


# =========================
# Drawing function
# =========================

def paint(event):

    x = event.x
    y = event.y

    radius = 10

    # Draw on tkinter canvas
    canvas.create_oval(
        x-radius,
        y-radius,
        x+radius,
        y+radius,
        fill="white",
        outline="white"
    )

    # Draw on numpy image
    x1 = max(0, x-radius)
    x2 = min(canvas_size, x+radius)

    y1 = max(0, y-radius)
    y2 = min(canvas_size, y+radius)

    image[y1:y2, x1:x2] = 255



# =========================
# Clear drawing
# =========================

def clear():

    canvas.delete("all")
    image[:] = 0



# =========================
# Prediction
# =========================

def predict_digit():

    # Convert numpy array to PIL image
    img = Image.fromarray(image)

    # Resize 280x280 -> 28x28
    img = img.resize((28, 28))

    # Convert back to numpy
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Flatten 28x28 -> 784
    img = img.reshape(784)

    prediction = network.predict(img)

    result_label.config(
        text=f"Prediction: {prediction}"
    )

    print("Prediction:", prediction)



# =========================
# GUI
# =========================

root = tk.Tk()

root.title("MNIST Digit Recognizer")


canvas = tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black"
)

canvas.pack()


canvas.bind(
    "<B1-Motion>",
    paint
)


predict_button = tk.Button(
    root,
    text="Predict",
    command=predict_digit
)

predict_button.pack()


clear_button = tk.Button(
    root,
    text="Clear",
    command=clear
)

clear_button.pack()


result_label = tk.Label(
    root,
    text="Prediction: ",
    font=("Arial", 16)
)

result_label.pack()


root.mainloop()