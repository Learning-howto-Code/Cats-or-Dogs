import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained Keras model
model = load_model('/Users/jakehopkins/Desktop/Gpm_cnn/best_model.keras')  # Update the path to your .h5 model

# Define the directory containing images
image_directory = '/Users/jakehopkins/Desktop/clean:dirty/Validation/Clean 5 gpm'

# List all files in the directory
filepaths = [
    os.path.join(image_directory, fname)
    for fname in os.listdir(image_directory)
    if fname.lower().endswith(('.png', '.jpg', '.jpeg'))  # Ensure only image files are processed
]

# Preprocess images
images = [
    img_to_array(load_img(filepath, target_size=(150, 150))) / 255.0
    for filepath in filepaths
]

# Convert to numpy array and ensure the input shape matches the model
input_data = np.array(images, dtype=np.float32)  # Shape: (batch_size, height, width, channels)

# Make predictions
predictions = model.predict(input_data)
binary_predictions = (predictions > 0.5).astype(int)

# Print predictions for each file
for filepath, pred, binary_pred in zip(filepaths, predictions, binary_predictions):
    print(f"File: {filepath}")
    print(f"Prediction (raw): {pred}")
    print(f"Prediction (binary): {binary_pred}")
    print("-" * 40)
