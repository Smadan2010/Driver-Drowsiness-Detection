"""
Image preprocessing utilities for Driver Drowsiness Detection.
Matches the exact preprocessing used in the trained MobileNetV2 model.
"""

import numpy as np
import cv2
from PIL import Image
import io


def resize_image(image, target_size=224):
    """
    Resize image to target size using OpenCV.
    
    Args:
        image: PIL Image or numpy array
        target_size: Target size (224x224 for MobileNetV2)
    
    Returns:
        numpy array of shape (target_size, target_size, 3)
    """
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    elif isinstance(image, np.ndarray):
        # Assume RGB format, convert to BGR for cv2
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Resize using INTER_LINEAR (same as notebook)
    resized = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    
    # Convert back to RGB
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    
    return resized


def normalize_image(image):
    """
    Normalize pixel values to [0, 1] range.
    This is the exact preprocessing used by MobileNetV2 in the notebook.
    
    Args:
        image: numpy array with pixel values in [0, 255]
    
    Returns:
        normalized numpy array with values in [0, 1] as float32
    """
    return image.astype('float32') / 255.0


def preprocess_single_image(image_input):
    """
    Complete preprocessing pipeline for a single image.
    Matches the exact pipeline used in the training notebook.
    
    Args:
        image_input: PIL Image, numpy array, or file path
    
    Returns:
        preprocessed image as float32 array of shape (224, 224, 3)
    """
    # Load image if file path
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        # Ensure RGB
        if image_input.mode != 'RGB':
            image = image_input.convert('RGB')
        else:
            image = image_input
    else:
        # Assume numpy array
        image = Image.fromarray((image_input * 255).astype(np.uint8)) if image_input.max() <= 1 else Image.fromarray(image_input.astype(np.uint8))
        if image.mode != 'RGB':
            image = image.convert('RGB')
    
    # Convert to numpy for processing
    image_array = np.array(image)
    
    # Resize to 224x224
    resized = resize_image(image_array, target_size=224)
    
    # Normalize to [0, 1]
    normalized = normalize_image(resized)
    
    # Expand dimensions for batch (add batch dimension)
    # Model expects (batch_size, 224, 224, 3)
    batched = np.expand_dims(normalized, axis=0)
    
    return batched, normalized


def preprocess_for_display(image_input):
    """
    Preprocess image for display purposes (resize but keep as uint8 for visualization).
    
    Args:
        image_input: PIL Image, numpy array, or file path
    
    Returns:
        resized image as numpy array uint8 (for display)
    """
    # Load image if file path
    if isinstance(image_input, str):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        if image_input.mode != 'RGB':
            image = image_input.convert('RGB')
        else:
            image = image_input
    else:
        image = Image.fromarray(image_input.astype(np.uint8))
        if image.mode != 'RGB':
            image = image.convert('RGB')
    
    # Convert to numpy and resize
    image_array = np.array(image)
    resized = resize_image(image_array, target_size=224)
    
    return resized
