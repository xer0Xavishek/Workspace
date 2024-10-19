import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps
import numpy as np

# Load the pretrained model
model = torch.load('mnist_cnn_model.pth', map_location=torch.device('cpu'))
model.eval()

# Set up the camera
cap = cv2.VideoCapture(0)

# Define a function to preprocess the image
def preprocess_image(frame):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Resize and pad the image to make it 28x28
    padded = np.pad(gray, ((12, 12), (12, 12)), mode='constant', constant_values=255)
    # Convert to PIL Image
    pil_img = Image.fromarray(padded)
    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    img = transform(pil_img)
    return img

# Define a function to predict the digit
def predict_digit(frame):
    with torch.no_grad():
        # Preprocess the image
        img = preprocess_image(frame)
        # Add batch dimension
        img = img.unsqueeze(0)
        # Make prediction
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame
    cv2.imshow('Frame', frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Preprocess the frame and predict the digit
    digit = predict_digit(frame)
    print("Predicted Digit:", digit)

# Release the capture
cap.release()
cv2.destroyAllWindows()
