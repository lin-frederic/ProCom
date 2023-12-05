import numpy as np
import scipy.ndimage as sc
import cv2
from PIL import Image
# black background
def remove_bg (image , mask):
    image_masked = image.copy()
    image_masked[mask == False] = [0, 0, 0]  # Set the BGR color to [0, 0, 0] for black
    #image_masked = cv2.cvtColor(image_masked, cv2.COLOR_BGR2RGB)
    return image_masked

# crop
def crop (image, bbox, z):
    # z is an additional margin
    if isinstance(image, Image.Image):
        image = np.array(image)
    image = image[bbox[1]-z:bbox[1]+bbox[3]+z, bbox[0]-z:bbox[2]+bbox[0]+z]
    return Image.fromarray(image)

# crop with mask
def crop_mask (image, mask, z):
    bbox = [np.min(np.where(mask)[1]), np.min(np.where(mask)[0]), np.max(np.where(mask)[1]), np.max(np.where(mask)[0])]
    return crop(image, bbox, z)

def gaussian_noise(image, mask, sigma=1.5):
    # Load your image and mask
    original_image = image.copy() # Replace with your image data

    # Apply Gaussian noise to the entire image
    noisy_image = sc.gaussian_filter(original_image, sigma)

    # Retain original pixel values where the mask is True
    noisy_image[mask] = original_image[mask]
    return noisy_image

def highlighted_image(image , mask):
    # Highlight important elements with a red border
    image_with_highlight = image.copy()
    image_with_highlight[mask] = [0, 0, 255]  # Set color (BGR) for important elements
    return image_with_highlight

def gaussian_blur(image, mask, sigma):
    # Load your image and mask
    original_image = image.copy() # Replace with your image data

    # Apply Gaussian noise to the entire image
    blur_mask = np.logical_not(mask)
    noisy_image = cv2.GaussianBlur(original_image, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REPLICATE)

    # Retain original pixel values where the mask is True
    noisy_image[mask] = original_image[mask]
    return noisy_image

def highlighted_contour(image, mask):
    # Create a copy of the image
    highlighted_image = image.copy()

    # Find the boundary between True and False elements in the mask
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw a red outline around the boundary
    cv2.drawContours(highlighted_image, contours, -1, (0, 0, 255), thickness=2)  # BGR color (red)
    return highlighted_image
