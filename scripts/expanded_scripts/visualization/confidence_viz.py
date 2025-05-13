#confidence visualization
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import config

def visualize_prediction(model, img_path, true_class=None):
    """Visualize model prediction with class activation map."""
    # Load and preprocess image
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMAGE_SIZE)
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
    plt.title(f'Predicted: {expected_classes[predicted_class]} ({confidence:.2f})')
    if true_class is not None:
        plt.title(f'Predicted: {expected_classes[predicted_class]} ({confidence:.2f})\nTrue: {expected_classes[true_class]}')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    os.makedirs('visualizations', exist_ok=True)
    plt.savefig(f'visualizations/{os.path.basename(img_path)}_prediction.png')
    plt.close()
    
    # Create confidence bar chart for all classes
    plt.figure(figsize=(10, 4))
    plt.bar(expected_classes, pred)
    plt.xlabel('Class')
    plt.ylabel('Confidence')
    plt.title('Class Confidence Scores')
    plt.tight_layout()
    plt.savefig(f'visualizations/{os.path.basename(img_path)}_confidence.png')
    plt.close()
    
    return predicted_class, confidence
