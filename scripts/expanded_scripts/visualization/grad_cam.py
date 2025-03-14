# grad_cam.py - Class activation mapping for model visualization

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Tuple, List, Union, Optional
import config

def create_visualization_model(model: tf.keras.Model) -> tf.keras.Model:
    """Create a model that outputs class activation maps for visualization.
    
    This function creates a modified version of the input model that outputs
    both the original predictions and the activations from the last convolutional layer,
    which are needed for Grad-CAM visualization.
    
    Args:
        model: The trained TensorFlow model to visualize
        
    Returns:
        A modified model that outputs both predictions and activations
        
    Raises:
        ValueError: If no convolutional layer is found in the model
    """
    # Find the last convolutional layer
    last_conv_layer = None
    
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_conv_layer = layer.name
            break
    
    if not last_conv_layer:
        raise ValueError("Could not find a convolutional layer in the model")
    
    # Create a model that outputs both the predictions and the activations
    try:
        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.output, model.get_layer(last_conv_layer).output]
        )
        return grad_model
    except Exception as e:
        raise ValueError(f"Failed to create visualization model: {str(e)}")

def generate_class_activation_map(
    model: tf.keras.Model,
    img_array: Union[np.ndarray, tf.Tensor],
    class_idx: int
) -> np.ndarray:
    """Generate class activation map using Grad-CAM technique.
    
    This function creates a heatmap that highlights the regions of the input image
    that most influenced the model's prediction for the specified class.
    
    Args:
        model: The trained TensorFlow model
        img_array: The preprocessed input image (normalized to [0,1])
        class_idx: The index of the target class to visualize
        
    Returns:
        A numpy array containing the class activation map (values in range [0,1])
    """
    # Create a model that outputs activations
    grad_model = create_visualization_model(model)
    
    # Compute gradient of the class output with respect to the feature maps
    with tf.GradientTape() as tape:
        # Cast image to float32
        img_array = tf.cast(img_array, tf.float32)
        
        # Add batch dimension if not already present
        if len(img_array.shape) == 3:
            img_array = tf.expand_dims(img_array, 0)
        
        # Forward pass to get predictions and feature maps
        pred, conv_outputs = grad_model(img_array)
        
        # Get the score for the target class
        class_output = pred[:, class_idx]
    
    # Gradient of the class with respect to the output feature map
    grads = tape.gradient(class_output, conv_outputs)
    
    # Importance weights for each feature map
    # Global average pooling of the gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the feature maps with the importance
    weighted_conv_outputs = conv_outputs * pooled_grads[tf.newaxis, tf.newaxis, tf.newaxis, :]
    
    # Average over all feature maps
    cam = tf.reduce_sum(weighted_conv_outputs, axis=-1)
    
    # Normalize the CAM
    cam = tf.maximum(cam, 0) / (tf.reduce_max(cam) + tf.keras.backend.epsilon())
    
    # Resize to the image size
    cam = tf.image.resize(cam, config.IMAGE_SIZE)  # Using config.IMAGE_SIZE instead of IMAGE_SIZE
    
    # Convert to numpy
    cam = cam.numpy()[0]
    
    return cam

def visualize_prediction(
    model: tf.keras.Model,
    img_path: str,
    true_class: Optional[int] = None,
    save_path: Optional[str] = None
) -> Tuple[int, float]:
    """Visualize model prediction with class activation map.
    
    This function creates a visualization that shows:
    1. The original image
    2. The class activation map (heatmap)
    3. The heatmap overlaid on the original image
    
    Args:
        model: The trained TensorFlow model
        img_path: Path to the image file
        true_class: Optional true class index for comparison
        save_path: Optional path to save the visualization
        
    Returns:
        Tuple of (predicted_class, confidence)
    """
    # Create the directory for visualizations if it doesn't exist
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    elif not os.path.exists(config.VISUALIZATION_DIR):
        os.makedirs(config.VISUALIZATION_DIR, exist_ok=True)
    
    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, config.IMAGE_SIZE)
    img_display = img.numpy().astype('uint8')
    
    img = tf.cast(img, tf.float32) / 255.0
    
    # Make prediction
    pred = model.predict(tf.expand_dims(img, 0))[0]
    predicted_class = np.argmax(pred)
    confidence = pred[predicted_class]
    
    # Generate CAM for predicted class
    cam = generate_class_activation_map(model, img, predicted_class)
    
    # Create heatmap
    heatmap = np.uint8(255 * cam)
    
    # Apply heatmap to original image
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    # Create superimposed image
    superimposed_img = jet_heatmap * 0.4 + img_display / 255.0 * 0.6
    
    # Plot
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_display)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Class Activation Map')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(superimposed_img)
    
    # Add title with class information
    title = f'Predicted: {config.EXPECTED_CLASSES[predicted_class]} ({confidence:.2f})'
    if true_class is not None:
        title += f'\nTrue: {config.EXPECTED_CLASSES[true_class]}'
    plt.title(title)
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    if save_path is not None:
        plt.savefig(save_path)
    else:
        vis_filename = f'{os.path.basename(img_path)}_prediction.png'
        plt.savefig(os.path.join(config.VISUALIZATION_DIR, vis_filename))
    
    plt.close()
    
    # Create confidence bar chart for all classes
    plt.figure(figsize=(10, 4))
    class_names = config.EXPECTED_CLASSES
    plt.bar(class_names, pred)
    plt.xlabel('Class')
    plt.ylabel('Confidence')
    plt.title('Class Confidence Scores')
    plt.tight_layout()
    
    # Save confidence chart
    if save_path is not None:
        conf_path = save_path.replace('.png', '_confidence.png')
        plt.savefig(conf_path)
    else:
        conf_filename = f'{os.path.basename(img_path)}_confidence.png'
        plt.savefig(os.path.join(config.VISUALIZATION_DIR, conf_filename))
    
    plt.close()
    
    return predicted_class, confidence