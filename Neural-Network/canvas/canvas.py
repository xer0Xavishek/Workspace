import tkinter as tk
from tkinter import Canvas, Button
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from PIL import ImageGrab

# Function to preprocess the input image
def preprocess_image(image):
    image = image.resize((28, 28))
    image = np.array(image)
    # Check if the image is grayscale
    if len(image.shape) > 2:
        image = image.mean(axis=2)  # Convert to grayscale
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)  # Add channel dimension
    image = image.astype('float32')
    image /= 255
    return image

# Load the pre-trained model
model = tf.keras.models.load_model('handwriting_recognition_model.h5')

# Function to predict the digit
def predict_digit():
    # Get the drawing from the canvas and preprocess it
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()
    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()
    image = ImageGrab.grab((x, y, x1, y1)).convert('L')
    image = preprocess_image(image)

    # Predict the digit
    prediction = model.predict(image)
    digit = np.argmax(prediction)
    label.config(text=f'Predicted Digit: {digit}')

# Function to handle mouse drag event
def paint(event):
    x1, y1 = (event.x - 5), (event.y - 5)
    x2, y2 = (event.x + 5), (event.y + 5)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=10)

# Function to clear the canvas
def clear_canvas():
    canvas.delete("all")
    label.config(text='')

# GUI setup
root = tk.Tk()
root.title('Handwriting Recognition')

canvas = Canvas(root, width=200, height=200, bg='white')
canvas.pack()

canvas.bind("<B1-Motion>", paint)

label = tk.Label(root, text='')
label.pack()

button_predict = Button(root, text="Predict", command=predict_digit)
button_predict.pack()

button_clear = Button(root, text="Clear", command=clear_canvas)
button_clear.pack()

root.mainloop()