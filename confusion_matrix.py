import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the saved model
model_path = '/Users/jakehopkins/Desktop/Gpm_cnn/best_model.keras'
model = tf.keras.models.load_model(model_path)

# Define parameters
img_height, img_width = 150, 150
batch_size = 32

# Load the validation dataset
validation_dir = '/Users/jakehopkins/Desktop/clean:dirty/Validation'
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=False  # Ensure the order of images matches predictions
)

# Obtain true labels and predictions
val_labels = np.concatenate([y for x, y in validation_dataset], axis=0)  # True labels
predictions = model.predict(validation_dataset)  # Model predictions
predicted_labels = np.argmax(predictions, axis=1)  # Convert probabilities to class predictions
true_labels = np.argmax(val_labels, axis=1)  # Convert categorical labels to single label per sample

# Compute the confusion matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)

# Display the confusion matrix
plt.figure(figsize=(8, 8))
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=['Cat', 'Dog'])
disp.plot(cmap=plt.cm.Blues, values_format='d')
plt.title("Confusion Matrix for Cats vs. Dogs Model")
plt.show()
            