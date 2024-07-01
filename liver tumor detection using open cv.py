#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # Apply thresholding using Otsu's method
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresholded

def segment_tumor(image_path):
    # Preprocess the image
    preprocessed_image = preprocess_image(image_path)

    # Find contours
    contours, _ = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image for drawing contours
    contour_image = np.zeros_like(preprocessed_image)

    # Draw contours on the image
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), thickness=cv2.FILLED)

    return contour_image, contours

def main():
    image_path = r'C:\Users\HP\OneDrive\Desktop\mini project\wind photos\images (2).jpg'  

    # Segment the tumor
    segmented_image, contours = segment_tumor(image_path)

    # Display the original and segmented images
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError(f"Image at path {image_path} could not be loaded.")

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Segmented Tumor")
    plt.imshow(segmented_image, cmap='gray')
    plt.axis('off')

    plt.show()

    # Print the number of detected contours
    print(f"Number of detected tumor-like regions: {len(contours)}")

if __name__ == "__main__":
    main()





