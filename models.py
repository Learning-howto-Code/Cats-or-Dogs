import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os


# Hyperparamaters
epochs = 2
batch_size = 64
drop = 0.4



train_dir = "/Users/jakehopkins/Downloads/Cats or Dogs/train"
val_dir = "/Users/jakehopkins/Downloads/Cats or Dogs/validation"

def create_generators(train_dir, val_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=0.1,  # Corrected typo: roation_range -> rotation_range
        width_shift_range=0.1,  # Random horizontal shifts
        height_shift_range=0.1,  # Random vertical shifts
        shear_range=0.1,  # Shearing transformations
        zoom_range=0.1,  # Zoom
        horizontal_flip=True,  # Randomly flip images
        fill_mode='nearest'  # Fill in newly created pixels
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,   # Path to train data
        target_size=(150, 150),
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,  # Path to validation data
        target_size=(150, 150),
        batch_size=32,
        color_mode='rgb',
        class_mode='categorical'
    )

    return train_generator, validation_generator


def count_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len([file for file in files if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))])
    return count

train_count = count_images(train_dir)
val_count = count_images(val_dir)
print(f"Number of training images: {train_count}")
print(f"Number of validation images: {val_count}")

def Cats_Dogs(nbr_classes=2):  # Make sure number of classes is correct
    my_input = Input(shape=(150, 150, 3))  # Input layer for images 150x150 with 3 color channels
    
    x = Conv2D(32, (3, 3), activation='relu')(my_input)  # First Conv2D layer
    x = MaxPooling2D()(x)  # Max Pooling layer
    x = BatchNormalization()(x)  # Batch Normalization layer
    x = Dropout(0.3)(x)

    # Convolutional Block 2
    x = Conv2D(64, (3, 3), activation='relu')(x)  # Second Conv2D layer
    x = MaxPooling2D()(x)  # Max Pooling layer
    x = BatchNormalization()(x)  # Batch Normalization layer
    x = Dropout(drop)(x)

    # Convolutional Block 3
    x = Conv2D(128, (5, 5), activation='relu')(x)  # Third Conv2D layer
    x = MaxPooling2D()(x)  # Max Pooling layer
    x = BatchNormalization()(x)  # Batch Normalization layer
    x = Dropout(drop)(x)

    x = Conv2D(128, (5, 5), activation='relu')(x)  # Fourth Conv2D layer
    x = MaxPooling2D()(x)  # Max Pooling layer
    x = BatchNormalization()(x)  # Batch Normalization layer
    x = Dropout(drop)(x)

    x = Conv2D(256, (7, 7), activation='relu', padding='same')(x)  # Updated Conv2D layer
    x = MaxPooling2D()(x)  # Max Pooling layer
    x = BatchNormalization()(x)  # Batch Normalization layer
    x = Dropout(drop)(x)

    # Global Average Pooling Layer
    x = GlobalAveragePooling2D()(x)  # Global Average Pooling layer

    # Fully Connected Layers
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(32, activation='relu')(x)  # Dense layer with 64 neurons
    x = Dense(nbr_classes, activation='softmax')(x)  # Output layer for classification


    return Model(inputs=my_input, outputs=x)  # Create and return the model



# Call the function to create the generators
train_generator, validation_generator = create_generators(train_dir, val_dir)


if __name__ == '__main__':
    # Ensure your directories are set
    train_dir = "/Users/jakehopkins/Downloads/Cats or Dogs/train"
    val_dir = "/Users/jakehopkins/Downloads/Cats or Dogs/validation"

    # Count images
    train_count = count_images(train_dir)
    val_count = count_images(val_dir)
    print(f"Number of training images: {train_count}")
    print(f"Number of validation images: {val_count}")

    # Create train and validation generators
    train_generator, validation_generator = create_generators(train_dir, val_dir)

    # Calculate steps
    batch_size = 32 #val=80,train=74 at 64
    steps_per_epoch = train_count // batch_size  # Number of batches in training
    validation_steps = val_count // batch_size  # Number of batches in validation

    # Instantiate and compile the model
    model = Cats_Dogs(2)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )
