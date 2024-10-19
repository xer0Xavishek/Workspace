import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load and preprocess the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build the neural network model
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model on the test dataset
test_loss, test_acc = model.evaluate(x_test, y_test)
print("Test accuracy:", test_acc)

# Function to predict numbers from user input including PNG files
def predict_number():
    while True:
        try:
            image_path = input("Enter the path of the image containing a handwritten digit: ")
            img = Image.open(image_path)
            img = img.convert('L')  # Convert to grayscale
            img = img.resize((28, 28))  # Resize image to 28x28
            img = np.array(img)  # Convert to numpy array
            img = 1 - img / 255.0  # Invert colors and normalize
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            prediction = model.predict(img)
            predicted_number = np.argmax(prediction)
            print("Predicted number:", predicted_number)
            plt.imshow(img[0], cmap='gray')
            plt.show()
        except Exception as e:
            print("Error:", e)

# Call the function to predict numbers
predict_number()
