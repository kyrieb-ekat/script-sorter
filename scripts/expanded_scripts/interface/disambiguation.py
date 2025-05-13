# User disambiguation interface
import os
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import tensorflow as tf

# Import from your project modules
from interface.classifier import classify_new_image
import config

def handle_user_disambiguation(
    model: tf.keras.Model,
    img_path: str,
    options: int = 2,
    features_db: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Present sample options to the user to help with difficult classifications.
    
    Args:
        model: The trained TensorFlow model for classification
        img_path: Path to the image file that needs disambiguation
        options: Number of top options to present to the user
        features_db: Optional dictionary of extracted features
        
    Returns:
        Dictionary containing original prediction, top alternative options,
        and paths to example images for each option
    """
    # Get a prediction first to know which classes are closest
    result = classify_new_image(model, img_path, features_db, confidence_threshold=1.0)  # Always return user input needed
    
    # Sort classes by confidence scores
    sorted_classes = sorted(
        result['confidence_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Select top N options to present to user
    top_options = sorted_classes[:options]
    
    # For each top option, find a representative example
    examples = {}
    for class_name, _ in top_options:
        class_idx = config.EXPECTED_CLASSES.index(class_name)
        
        # Find examples in training set
        class_dir = os.path.join(config.TRAIN_DIR, class_name)
        if os.path.exists(class_dir):
            examples_list = [f for f in os.listdir(class_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if examples_list:
                # Select a random example
                example_path = os.path.join(class_dir, random.choice(examples_list))
                examples[class_name] = example_path