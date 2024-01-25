import cv2
import numpy as np


def extract_sticky_paper_region(image_path):   
    original_image = cv2.imread(image_path)
    ycbcr_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2YCrCb)

    # Extract the Cb component
    cb_component = ycbcr_image[:, :, 1]

    # Apply Otsu's method for binary image
    _, binary_image = cv2.threshold(cb_component, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological fill operation
    kernel = np.ones((5, 5), np.uint8)
    binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    # Extract the Region of Interest (RoI)
    roi_image = cv2.bitwise_and(original_image, original_image, mask=binary_image)
    return roi_image

def divide_into_sub_blocks(image, block_size):
    height, width, _ = image.shape

    # Create a list to store sub-block images
    sub_blocks = []
    for y in range(0, height, block_size):
        for x in range(0, width, block_size):
            sub_block = image[y:y+block_size, x:x+block_size]
            sub_blocks.append(sub_block)
    return sub_blocks

def saliency_region_detection(sub_block):
    # Convert the sub-block to grayscale
    gray_sub_block = cv2.cvtColor(sub_block, cv2.COLOR_BGR2GRAY)

    # Calculate the spectral residual using the Fourier Transform
    spectrum = np.fft.fft2(gray_sub_block)
    log_spectrum = np.log(np.abs(spectrum) + 1e-5)
    spectral_residual = log_spectrum - cv2.boxFilter(log_spectrum, -1, (3, 3))

    # Inverse Fourier Transform to obtain the saliency map
    saliency_map = np.abs(np.fft.ifft2(np.exp(spectral_residual + 1j*np.angle(spectrum))))**2

    # Convert the saliency map to 8-bit unsigned integer format
    saliency_map = cv2.convertScaleAbs(saliency_map)

    # Binarization of the saliency map
    _, binary_saliency = cv2.threshold(saliency_map, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations for better segmentation
    kernel = np.ones((5, 5), np.uint8)
    binary_saliency = cv2.morphologyEx(binary_saliency, cv2.MORPH_CLOSE, kernel)
    return binary_saliency

def visualize_detected_objects(original_image, binary_saliency):
    # Find contours of the detected objects
    contours, _ = cv2.findContours(binary_saliency, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image
    result_image = original_image.copy()
    cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Detected Objects", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Provide the path to your image
    image_path = "download.jpg"

    # Extract Sticky Paper Region
    roi_image = extract_sticky_paper_region(image_path)

    # Divide into sub-blocks
    block_size = 64
    sub_blocks = divide_into_sub_blocks(roi_image, block_size)

    # Saliency region detection and visualization for each sub-block
    for sub_block in sub_blocks:
        binary_saliency = saliency_region_detection(sub_block)
        visualize_detected_objects(sub_block, binary_saliency)

if __name__ == "__main__":
    main()
