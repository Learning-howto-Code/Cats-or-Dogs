import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, Input, BatchNormalization, Add, Activation, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/gdrive')

# Define parameters
img_height, img_width = 150, 150  # Increased image size
batch_size = 32  # Reduced batch size due to larger model
epochs = 100  # Increased epochs
drop = .2  # Increased dropout
learning_rate = .0025

# Define the data directories
train_dir = '/content/Cats or Dogs/train'
validation_dir = '/content/Cats or Dogs/validation'

# Function to count images in a directory (unchanged)
def count_images(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        count += len([file for file in files if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif'))])
    return count

# Print number of training and validation images
train_count = count_images(train_dir)
val_count = count_images(validation_dir)
print(f"Number of training images: {train_count}")
print(f"Number of validation images: {val_count}")

# Function to create a tf.data.Dataset from directory
def create_dataset(directory, is_training=True):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        label_mode='categorical',
        shuffle=is_training
    )

    # Data augmentation for training set
    if is_training:
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])
        dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y),
                              num_parallel_calls=tf.data.AUTOTUNE)

    return dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Create datasets
train_dataset = create_dataset(train_dir, is_training=True)
validation_dataset = create_dataset(validation_dir, is_training=False)

# Define a residual block
def residual_block(x, filters, kernel_size=3):
    y = Conv2D(filters, kernel_size, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(filters, kernel_size, padding='same')(y)
    y = BatchNormalization()(y)
    out = Add()([x, y])
    out = Activation('relu')(out)
    return out

# Define the model
def Cats_Dogs_Deep(nbr_classes=2): # increae dnese layers
    my_input = Input(shape=(150, 150, 3))

    x = Conv2D(32, (7, 7), activation='relu')(my_input)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Conv2D(256, (3, 3), activation='relu')(x) # the 256 is the amount of pattens it's looking for, 256 is probalby to high
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)
    return Model(inputs=my_input, outputs=x)

    return Model(inputs, outputs)


def Cats_Dogs_Deep2(nbr_classes=2): 
    my_input = Input(shape=(150, 150, 3))

    x = Conv2D(64, (7, 7), activation='relu')(my_input)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv2D(32, (5, 5), activation='relu')(x)
    x = MaxPooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(drop)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)
    x = Dense(128, activation='relu')(x)
    x = Dense(nbr_classes, activation='softmax')(x)
    return Model(inputs=my_input, outputs=x)

    return Model(inputs, outputs)
# Build the model
model = Cats_Dogs_Deep2(2)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# Callbacks for saving the model and early stopping
path_to_save_model = '/content/gdrive/MyDrive/Models'
if not os.path.exists(path_to_save_model):
    os.makedirs(path_to_save_model)

ckpt_saver = ModelCheckpoint(
    os.path.join(path_to_save_model, 'best_model_deep.keras'),
    monitor='val_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)

# Train the model
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset,
    callbacks=[ckpt_saver, early_stop]
)

# Save the final model
model.save(os.path.join(path_to_save_model, 'final_model_deep.keras'))

print("Training completed. Model saved.")

# Uncomment the following lines if you want to plot the training history
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.show()