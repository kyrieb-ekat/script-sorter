# Classification function for new images
import os
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Dict, Any, Union, Optional

# Import from your project modules
from visualization.grad_cam import visualize_prediction
import config

def classify_new_image(
    model: tf.keras.Model,
    img_path: str,
    features_db: Optional[Dict[str, Any]] = None,
    confidence_threshold: float = 0.7
) -> Dict[str, Any]:
    """Classify a new image and determine if user input is needed.
    
    Args:
        model: The trained TensorFlow model for classification
        img_path: Path to the image file to classify
        features_db: Optional dictionary of extracted features
        confidence_threshold: Threshold below which user input is requested
        
    Returns:
        Dictionary containing classification results, confidence scores,
        and a flag indicating if user disambiguation is needed
    """
    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img_array = tf.cast(img, tf.float32) / 255.0
    
    # Extract features if available
    features = None
    if features_db and img_path in features_db:
        feature_dict = features_db[img_path]
        features = [
            feature_dict['staffline_confidence'],
            feature_dict['layout_features']['text_ratio'],
            feature_dict['layout_features']['top_margin'],
            feature_dict['layout_features']['bottom_margin'],
            feature_dict['layout_features']['left_margin'],
            feature_dict['layout_features']['right_margin']
        ]
        features = np.array([features])
    
    # Make prediction
    if features is not None and hasattr(model, 'input_names') and len(model.input_names) > 1:
        pred = model.predict({'image_input': tf.expand_dims(img_array, 0), 
                             'feature_input': features})[0]
    else:
        pred = model.predict(tf.expand_dims(img_array, 0))[0]
    
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class]
    
    # Check if the confidence is below threshold
    needs_user_input = confidence < confidence_threshold
    
    # Generate visualization
    visualize_prediction(model, img_path)
    
    return {
    'predicted_class': int(predicted_class),
    'class_name': config.EXPECTED_CLASSES[predicted_class],
    'confidence': float(confidence),
    'confidence_scores': {config.EXPECTED_CLASSES[i]: float(pred[i]) 
                         for i in range(len(config.EXPECTED_CLASSES))},
    'needs_user_input': needs_user_input
}