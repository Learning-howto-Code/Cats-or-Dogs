from PIL import Image
import numpy as np
import os

# Paths to the dataset and output directory
input_dir = "/Users/jakehopkins/Downloads/Cats or Dogs/validation/Cat" # then change to cats
output_dir = "/Users/jakehopkins/Downloads/Cats or Dogs/val_edges/Cat" # then chagne to cats
os.makedirs(output_dir, exist_ok=True)

# Sobel kernels for x and y gradients
sobel_x = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]])

sobel_y = np.array([[-1, -2, -1],
                    [0,  0,  0],
                    [1,  2,  1]])

def apply_sobel_edge_detection(image_path, output_path):
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
    image_np = np.array(image)

    # Initialize gradients in x and y directions
    gx = np.zeros_like(image_np, dtype=float)
    gy = np.zeros_like(image_np, dtype=float)

    # Apply Sobel filters by convolution
    for i in range(1, image_np.shape[0] - 1):
        for j in range(1, image_np.shape[1] - 1):
            gx[i, j] = np.sum(sobel_x * image_np[i-1:i+2, j-1:j+2])
            gy[i, j] = np.sum(sobel_y * image_np[i-1:i+2, j-1:j+2])

    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(gx**2 + gy**2)

    # Scale to 0-255 and convert to uint8
    gradient_magnitude = np.uint8(255 * gradient_magnitude / np.max(gradient_magnitude))

    # Save result
    edge_image = Image.fromarray(gradient_magnitude)
    edge_image.save(output_path)

# Process all images in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        apply_sobel_edge_detection(input_path, output_path)

print("Edge detection applied to all images using Sobel operator.")



