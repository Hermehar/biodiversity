import cv2
import numpy as np

# Path to the image you want to test
IMAGE_PATH = "/content/download1.jpg"

# Read the image
image = cv2.imread(IMAGE_PATH)

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply thresholding to create a binary image
_, binary_image = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours in the binary image
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours on the original image
result_image = image.copy()
cv2.drawContours(result_image, contours, -1, (0, 255, 0), 2)

# Display the result
cv2_imshow(result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
