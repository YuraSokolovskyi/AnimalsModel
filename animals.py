import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model
from loadModel import *

# Load the Keras model for image classification
model = load_model('model')
file = ""


# Define a function to preprocess the image for classification
def preprocess_image(image):
    # Resize the image to the input size of the model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image_array = np.array(image)
    # Normalize the pixel values to be between 0 and 1
    image_array = image_array / 255.0
    # Expand the dimensions of the image array to match the input shape of the model
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def browse_file():
    global file
    # Open a file dialog and get the selected file
    file_path = filedialog.askopenfilename()
    print(file_path)
    file = file_path
    # Load the selected image and display it in the window
    image = Image.open(file_path)
    image = image.resize((400, 400))
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo


def run_code():
    global model, file
    result = modelPrediction(file, model)
    label_text = result
    # Update the label text with the result
    result_label.config(text=label_text)


# Create a Tkinter window
root = tk.Tk()

# Create a button to browse for an image file
browse_button = tk.Button(root, text="Browse", command=browse_file)

# Create a button to run the code
run_button = tk.Button(root, text="Run Code", command=run_code)

# Create a label to display the selected image
image_label = tk.Label(root)

# Create a label to display the result of the image classification
result_label = tk.Label(root, text="")

# Pack the buttons and labels into the window
browse_button.pack()
run_button.pack()
image_label.pack()
result_label.pack()

# Start the Tkinter event loop
root.mainloop()