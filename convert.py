import tensorflow as tf

# Load the Keras model from your saved .keras file
model = tf.keras.models.load_model('/Users/jakehopkins/Desktop/Cats and Dogs/best_model.keras')

# Save the model in a format compatible with TFLiteConverter (SavedModel format)
model.save('saved_model')

# Now, convert the SavedModel to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
tflite_model = converter.convert()

# Save the converted TFLite model to a .tflite file
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
