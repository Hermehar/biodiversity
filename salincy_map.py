import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_gbvs_saliency(image):
    # GBVS (Graph-Based Visual Saliency) algorithm implementation
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the gradient magnitude using the Sobel operator
    gradient_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

    # Compute the dissimilarity measure
    dissimilarity = np.abs(np.log(gradient_magnitude / np.max(gradient_magnitude) + 1e-6))
      # Apply Gaussian smoothing to the dissimilarity map
    dissimilarity = cv2.GaussianBlur(dissimilarity, (5, 5), 0)

    # Compute the saliency map
    saliency_map = 1 - dissimilarity

    # Normalize the saliency map
    saliency_map = (saliency_map - np.min(saliency_map)) / (np.max(saliency_map) - np.min(saliency_map))

    return saliency_map

def generate_cos_saliency(image):
    # COS (Cluster-based Saliency Detection) algorithm implementation
    # Convert the image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Extract the L channel
    l_channel = lab_image[:, :, 0]

    # Apply K-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(l_channel.reshape((-1, 1)).astype(np.float32), 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
      # Reshape labels to match the original image
    labels = labels.reshape(l_channel.shape)

    # Check if the cluster centers have the expected size
    if centers.shape == (2, 1):
        # Compute the saliency map based on the distance to cluster centers
        diff = l_channel - centers[1]

        # Check if the arrays have the same size
        if diff.shape == l_channel.shape:
            # Convert zeros_like array to float32 to match the type of diff array
            zeros_like_array = np.zeros_like(l_channel, dtype=np.float32)
            saliency_map = cv2.magnitude(diff, zeros_like_array)

            # Normalize the saliency map
            min_value = np.min(saliency_map)
            max_value = np.max(saliency_map)

            if min_value != max_value:
                saliency_map = (saliency_map - min_value) / (max_value - min_value)
            else:
                # Handle the case where min_value and max_value are the same (e.g., constant image)
                saliency_map = np.zeros_like(saliency_map)

            return saliency_map
    # Handle the case where the cluster centers have unexpected size
    return None

def generate_binary_mask(saliency_map, threshold=0.5):
    # Apply threshold to create a binary mask
    binary_mask = (saliency_map > threshold).astype(np.uint8)

    return binary_mask

example_image = cv2.imread('download1.jpg')
gbvs_saliency_map = generate_gbvs_saliency(example_image)
cos_saliency_map = generate_cos_saliency(example_image)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(example_image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')

# Display the GBVS saliency map
plt.subplot(1, 3, 2)
plt.imshow(gbvs_saliency_map, cmap='viridis')
plt.title('GBVS Saliency Map')

# Display the COS saliency map
plt.subplot(1, 3, 3)
plt.imshow(cos_saliency_map, cmap='viridis')
plt.title('COS Saliency Map')
plt.tight_layout()
plt.show()
