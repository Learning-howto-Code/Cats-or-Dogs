import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load the trained Keras model
model = load_model('/Users/jakehopkins/Downloads/best_model_deep.keras')  # Update with your .h5 model path

# Specify the path to the single image
image_path = '/Users/jakehopkins/Downloads/Cats or Dogs/validation/Dog/dog1.jpg'  # Replace with your image file

# Load and preprocess the image
image = load_img(image_path, target_size=(150, 150))  # Resize the image to the expected input size
image_array = img_to_array(image) / 255.0  # Convert to array and normalize to [0, 1]
input_data = np.expand_dims(image_array, axis=0)  # Add batch dimension: shape becomes (1, 150, 150, 3)

# Make a prediction
prediction = model.predict(input_data)

# Print the prediction
print(f"Prediction for {image_path}: {prediction}")
