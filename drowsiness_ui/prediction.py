"""
Model loading and prediction module for Driver Drowsiness Detection.
Uses the actual trained MobileNetV2 model.
"""

import os
import numpy as np
import tensorflow as tf
from PIL import Image
import streamlit as st

from preprocessing import preprocess_single_image, preprocess_for_display
from mappings import (
    CLASS_NAMES, 
    get_fatigue_level,
    INDEX_TO_CLASS
)


@st.cache_resource
def load_model():
    """
    Load the trained MobileNetV2 model.
    Cached to avoid reloading on every Streamlit rerun.
    
    Returns:
        Loaded Keras model
    """
    model_path = 'mobilenetv2_finetuned.keras'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Current directory: {os.getcwd()}\n"
            f"Files in directory: {os.listdir('.')}"
        )
    
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {str(e)}")


def predict_drowsiness(image_input):
    """
    Run prediction on a single image using the trained MobileNetV2 model.
    
    Args:
        image_input: PIL Image, numpy array, or file path
    
    Returns:
        dict with keys:
            - 'predicted_class': str, class name
            - 'predicted_index': int, class index
            - 'confidence': float, confidence score (0-1)
            - 'all_probabilities': dict, probabilities for all classes
            - 'fatigue_level': int, 0/1/2
            - 'fatigue_name': str
            - 'fatigue_description': str
            - 'display_image': numpy array for display
            - 'success': bool
            - 'error': str or None
    """
    try:
        # Load model
        model = load_model()
        
        # Preprocess image for model
        preprocessed_batch, preprocessed_single = preprocess_single_image(image_input)
        
        # Get preprocess image for display
        display_image = preprocess_for_display(image_input)
        
        # Run prediction
        predictions = model.predict(preprocessed_batch, verbose=0)
        
        # Get predicted class
        predicted_index = np.argmax(predictions[0])
        predicted_class = CLASS_NAMES[predicted_index]
        confidence = float(predictions[0][predicted_index])
        
        # Get all probabilities
        all_probs = {CLASS_NAMES[i]: float(predictions[0][i]) for i in range(len(CLASS_NAMES))}
        
        # Get fatigue level
        fatigue_level, fatigue_name, fatigue_description = get_fatigue_level(predicted_index)
        
        return {
            'predicted_class': predicted_class,
            'predicted_index': predicted_index,
            'confidence': confidence,
            'all_probabilities': all_probs,
            'fatigue_level': fatigue_level,
            'fatigue_name': fatigue_name,
            'fatigue_description': fatigue_description,
            'display_image': display_image,
            'success': True,
            'error': None
        }
    
    except Exception as e:
        return {
            'success': False,
            'error': f"Prediction failed: {str(e)}",
            'predicted_class': None,
            'predicted_index': None,
            'confidence': None,
            'all_probabilities': {},
            'fatigue_level': None,
            'fatigue_name': None,
            'fatigue_description': None,
            'display_image': None
        }


def predict_batch(image_paths_or_arrays):
    """
    Run prediction on multiple images.
    
    Args:
        image_paths_or_arrays: list of image paths, PIL Images, or numpy arrays
    
    Returns:
        list of prediction results (same format as predict_drowsiness)
    """
    results = []
    for img_input in image_paths_or_arrays:
        result = predict_drowsiness(img_input)
        results.append(result)
    return results
