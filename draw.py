import tkinter as tk
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

from network import Network


# =========================
# Load trained model
# =========================

layer1_path = "model/layer1_weights.npy"
layer2_path = "model/layer2_weights.npy"


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

# Store drawing
image = np.zeros((canvas_size, canvas_size), dtype=np.uint8)


# =========================
# Drawing
# =========================

def paint(event):

    x = event.x
    y = event.y

    radius = 10

    canvas.create_oval(
        x-radius,
        y-radius,
        x+radius,
        y+radius,
        fill="white",
        outline="white"
    )

    x1 = max(0, x-radius)
    x2 = min(canvas_size, x+radius)

    y1 = max(0, y-radius)
    y2 = min(canvas_size, y+radius)

    image[y1:y2, x1:x2] = 255



# =========================
# Clear canvas
# =========================

def clear():

    canvas.delete("all")
    image[:] = 0

    result_label.config(
        text="Prediction:"
    )


# =========================
# Image preprocessing
# =========================

def crop_digit(img):

    coords = np.argwhere(img > 0)

    # Empty image
    if len(coords) == 0:
        return img


    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)


    cropped = img[
        y_min:y_max+1,
        x_min:x_max+1
    ]

    return cropped



def resize_digit(img):

    img = Image.fromarray(img)

    # Resize while keeping ratio
    img.thumbnail((20,20))


    # Create 28x28 black image
    new_img = Image.new(
        "L",
        (28,28),
        0
    )


    # Center digit
    x = (28 - img.width)//2
    y = (28 - img.height)//2


    new_img.paste(
        img,
        (x,y)
    )


    return np.array(new_img)



# =========================
# Prediction
# =========================

def predict_digit():

    img = image.copy()


    # Remove empty borders
    img = crop_digit(img)


    # Resize and center
    img = resize_digit(img)


    # Show what the AI sees
    plt.imshow(
        img,
        cmap="gray"
    )

    plt.title("Processed MNIST Image")

    plt.show()


    # Normalize
    img = img / 255.0


    # Flatten
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

root.title(
    "MNIST Digit Recognizer"
)


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
    text="Prediction:",
    font=("Arial",16)
)

result_label.pack()



root.mainloop()