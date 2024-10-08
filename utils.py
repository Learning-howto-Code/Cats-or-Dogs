import os
import shutil
import random
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt



def split_and_move_images_and_resize(src_path, dest_train, dest_val, val_ratio=0.1, img_size=(150, 150)):
    
    def clear_directories():
        """
        Clears specific directories before moving and resizing images.
        """
        directories_to_clear = [
            "/Users/jakehopkins/Downloads/Cats or Dogs/validation/Dog",
            "/Users/jakehopkins/Downloads/Cats or Dogs/validation/Cat",
            "/Users/jakehopkins/Downloads/Cats or Dogs/train/Dog",
            "/Users/jakehopkins/Downloads/Cats or Dogs/train/Cat"
        ]

        for directory in directories_to_clear:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    file_path = os.path.join(directory, filename)
                    try:
                        if os.path.isfile(file_path) or os.path.islink(file_path):
                            os.unlink(file_path)
                        elif os.path.isdir(file_path):
                            shutil.rmtree(file_path)
                        print(f"Cleared {file_path}")
                    except Exception as e:
                        print(f"Failed to delete {file_path}. Reason: {e}")
    
    """
    Splits images into training and validation sets, resizes them, and moves them to their respective folders.
    Clears the destination folders before running.
    
    Args:
    - src_path (str): Path to the source images folder (where both 'Cat' and 'Dog' folders exist).
    - dest_train (str): Path to the destination train folder.
    - dest_val (str): Path to the destination validation folder.
    - val_ratio (float): Percentage of images to move to validation set (default is 0.1 or 10%).
    - img_size (tuple): Target image size as (width, height), e.g., (150, 150).
    """

    # Clear destination folders before moving files
    clear_directories()

    # Categories (e.g., 'Cat', 'Dog') based on folder names in the source directory
    categories = ['Cat', 'Dog']
    
    # Supported image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')  # Add other formats as needed

    for category in categories:
        src_category_folder = os.path.join(src_path, category)  # e.g., PetImages/Cat or PetImages/Dog
        dest_train_category_folder = os.path.join(dest_train, category)  # e.g., train/Cat or train/Dog
        dest_val_category_folder = os.path.join(dest_val, category)  # e.g., validation/Cat or validation/Dog

        # Create destination directories if they don't exist
        os.makedirs(dest_train_category_folder, exist_ok=True)
        os.makedirs(dest_val_category_folder, exist_ok=True)

        # List all valid image files in the source category folder (filter by extensions)
        all_files = [f for f in os.listdir(src_category_folder) if f.lower().endswith(valid_extensions) and os.path.isfile(os.path.join(src_category_folder, f))]

        # Check if there are files in the source folder
        if not all_files:
            print(f"No image files found in {src_category_folder} for category {category}")
            continue  # Skip this category if no files are found

        # Shuffle files to randomize selection
        random.shuffle(all_files)

        # Calculate number of validation files
        val_size = int(len(all_files) * val_ratio)

        # Split files into validation and training sets
        val_files = all_files[:val_size]
        train_files = all_files[val_size:]

        # Function to resize using Pillow and NumPy and move images
        def resize_and_move_numpy(file, dest_folder):
            src_file_path = os.path.join(src_category_folder, file)
            dest_file_path = os.path.join(dest_folder, file)

            try:
                # Open image using Pillow (PIL)
                img = Image.open(src_file_path)

                # Resize the image using the new LANCZOS filter
                resized_img = img.resize(img_size, Image.Resampling.LANCZOS)

                # Convert resized image to NumPy array (optional if you want to process it as a NumPy array)
                img_array = np.array(resized_img)

                # Convert back to Pillow Image for saving (if further NumPy processing was done)
                resized_pil_img = Image.fromarray(img_array)

                # Save the resized image to the destination folder
                resized_pil_img.save(dest_file_path)
                print(f"Moved and resized {file} to {dest_folder}")

            except Exception as e:
                print(f"Failed to process {file}. Error: {e}")

        # Move and resize validation files
        for file in val_files:
            resize_and_move_numpy(file, dest_val_category_folder)

        # Move and resize training files
        for file in train_files:
            resize_and_move_numpy(file, dest_train_category_folder)

        print(f"Moved {len(val_files)} files to validation and {len(train_files)} files to training for {category}.")


# Sets up Data Generators
def create_generators(train_dir, val_dir):

    train_datagen = ImageDataGenerator(
    rescale=1./255,
    roation_range=20,
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Shearing transformations
    zoom_range=0.2,  # Zoom
    horizontal_flip=True,  # Randomly flip images
    fill_mode='nearest'  # Fill in newly created pixel                 
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