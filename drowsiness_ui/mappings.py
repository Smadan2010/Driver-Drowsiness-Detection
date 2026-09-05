"""
Class and fatigue level mappings for Driver Drowsiness Detection.
These match the exact mappings used in the trained model.
"""

# Class names - EXACT order from the notebook
CLASS_NAMES = ['Closed', 'Open', 'no_yawn', 'yawn']

# Class indices
CLASS_INDICES = {
    'Closed': 0,
    'Open': 1,
    'no_yawn': 2,
    'yawn': 3
}

# Reverse mapping
INDEX_TO_CLASS = {v: k for k, v in CLASS_INDICES.items()}


def get_fatigue_level(predicted_class_index):
    """
    Convert 4-class model prediction to 3-level fatigue classification.
    This is the exact decision fusion logic from the notebook.
    
    Args:
        predicted_class_index: Model output class index (0-3)
    
    Returns:
        tuple: (fatigue_level_int, fatigue_name_str, description)
        - fatigue_level_int: 0 (Alert), 1 (Mild Fatigue), 2 (Severe Fatigue)
        - fatigue_name_str: Human-readable fatigue level
        - description: Explanation
    """
    if predicted_class_index == 0:  # Closed
        return (
            2,
            "SEVERE FATIGUE",
            "Eyes are closed. The driver is drowsy and a break is needed immediately."
        )
    elif predicted_class_index == 1:  # Open
        return (
            0,
            "ALERT",
            "Eyes are open. The driver appears alert and focused."
        )
    elif predicted_class_index == 2:  # no_yawn
        return (
            0,
            "ALERT",
            "No yawning detected. The driver is in a normal state."
        )
    else:  # yawn (3)
        return (
            1,
            "MILD FATIGUE",
            "Yawning detected. The driver should consider taking a break soon."
        )


def get_fatigue_color(fatigue_level):
    """
    Get color code for fatigue level for visualization.
    
    Args:
        fatigue_level: 0 (Alert), 1 (Mild), 2 (Severe)
    
    Returns:
        Color code (hex)
    """
    colors = {
        0: "#51CF66",  # Green - Alert
        1: "#FFD43B",  # Yellow - Mild Fatigue
        2: "#FF6B6B"   # Red - Severe Fatigue
    }
    return colors.get(fatigue_level, "#999999")


def get_fatigue_emoji(fatigue_level):
    """
    Get emoji for fatigue level.
    
    Args:
        fatigue_level: 0 (Alert), 1 (Mild), 2 (Severe)
    
    Returns:
        Emoji string
    """
    emojis = {
        0: "✅",  # Alert - Safe
        1: "⚠️",  # Mild - Warning
        2: "🛑"   # Severe - Stop
    }
    return emojis.get(fatigue_level, "❓")


# Project metrics - ACTUAL VALUES from the notebook
PROJECT_METRICS = {
    'custom_cnn': {
        'accuracy': 0.6828,
        'precision': 0.6845,
        'recall': 0.6828,
        'f1_score': 0.6801,
        'training_time': 19.45,
        'test_set_size': 435
    },
    'mobilenetv2': {
        'accuracy': 0.9770,
        'precision': 0.9783,
        'recall': 0.9770,
        'f1_score': 0.9770,
        'training_time': 23.83,
        'test_set_size': 435
    }
}

# Dataset information - ACTUAL VALUES from the notebook
DATASET_INFO = {
    'total_images': 2900,
    'training_samples': 2030,
    'validation_samples': 435,
    'test_samples': 435,
    'image_size': 224,
    'classes': CLASS_NAMES,
    'num_classes': 4,
}

# Class distribution in dataset (ACTUAL from notebook)
CLASS_DISTRIBUTION = {
    'Closed': 726,      # 25.03%
    'Open': 726,        # 25.03%
    'no_yawn': 725,     # 25.00%
    'yawn': 723         # 24.93%
}

# MobileNetV2 per-class performance (from notebook classification report)
MOBILENETV2_PER_CLASS_METRICS = {
    'Closed': {'precision': 1.0000, 'recall': 1.0000, 'f1_score': 1.0000, 'support': 109},
    'Open': {'precision': 1.0000, 'recall': 1.0000, 'f1_score': 1.0000, 'support': 109},
    'no_yawn': {'precision': 0.9224, 'recall': 0.9907, 'f1_score': 0.9554, 'support': 108},
    'yawn': {'precision': 0.9901, 'recall': 0.9174, 'f1_score': 0.9524, 'support': 109}
}

# Custom CNN per-class performance (not extracted from notebook - omitted)
# Only verified overall metrics are used for Custom CNN
CUSTOM_CNN_PER_CLASS_METRICS = None
