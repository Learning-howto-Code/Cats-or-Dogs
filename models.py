import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os

# Define parameters
img_height, img_width = 150, 150
batch_size = 32
epochs = 50

# Define the data directories
train_dir = '/Users/jakehopkins/Downloads/Cats or Dogs/train'
validation_dir = '/Users/jakehopkins/Downloads/Cats or Dogs/validation'

# Function to create a tf.data.Dataset from directory
def create_dataset(directory, is_training=True):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='binary',
        shuffle=is_training
    )
    
    # Data augmentation for training set
    if is_training:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Create datasets
train_dataset = create_dataset(train_dir, is_training=True)
validation_dataset = create_dataset(validation_dir, is_training=False)

# Define the model
model = Sequential([
    Input(shape=(img_height, img_width, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Train the model
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    callbacks=[early_stopping, model_checkpoint]
)

# Save the final model
model.save('final_model.keras')

print("Training completed. Model saved.")