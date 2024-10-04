import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Flatten
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

def Cats_Dogs(nbr_classes=2):  # Make sure number of classes is correct
    my_input = Input(shape=(150, 150, 3))  # Input layer for images 150x150 with 3 color channels
    x = Conv2D(32, (3, 3), activation='relu')(my_input)  # First Conv2D layer
    x = MaxPooling2D()(x)  # Max Pooling layer
    x = BatchNormalization()(x)  # Batch Normalization layer

    x = GlobalAveragePooling2D()(x)  # Global Average Pooling layer
    x = Dense(64, activation='relu')(x)  # Dense layer with 64 neurons
    x = Dense(nbr_classes, activation='softmax')(x)  # Output layer for classification

    return Model(inputs=my_input, outputs=x)  # Create and return the model

if __name__ == '__main__':
    model = Cats_Dogs(2)  # Instantiate the model with the correct number of classes (2 for Cats and Dogs)
    model.summary()  # Print model summary
