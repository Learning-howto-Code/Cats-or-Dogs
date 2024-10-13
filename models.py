import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.layers import Conv2D, Input, Dense, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, Flatten, Dropout
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import math

# Hyperparameters
epochs = 2
batch_size = 64
drop = 0.4

train_dir = "/Users/jakehopkins/Downloads/Cats or Dogs/train"
val_dir = "/Users/jakehopkins/Downloads/Cats or Dogs/validation"

def create_generators(train_dir, val_dir):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=batch_size,
        color_mode='rgb',
        class_mode='categorical'
    )

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(150, 150),
        batch_size=batch_size,
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

def Cats_Dogs(nbr_classes=2):
    my_input = Input(shape=(150, 150, 3))
    
    x = Conv2D(32, (3, 3), activation='relu')(my_input)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)

    return Model(inputs=my_input, outputs=x)

# Create the generators
train_generator, validation_generator = create_generators(train_dir, val_dir)

# Calculate steps using ceiling division
steps_per_epoch = math.ceil(train_count / batch_size)
validation_steps = math.ceil(val_count / batch_size)

# Build the model
model = Cats_Dogs(2)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for saving the model and early stopping
path_to_save_model = '/Users/jakehopkins/Downloads/best_model.keras'
ckpt_saver = ModelCheckpoint(
    path_to_save_model,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=[ckpt_saver, early_stop]
)

# Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()
