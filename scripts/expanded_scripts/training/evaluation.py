#model evaluation functions
#model evaluation and performance analysis

# evaluation.py - Model evaluation functions for manuscript classification

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional, Union
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support

from training.dataset import create_dataset
from visualization.grad_cam import visualize_prediction
import config

def evaluate_model(
    model: tf.keras.Model,
    dataset_dir: str = config.TEST_DIR,
    features_db: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
    save_dir: Optional[str] = None
) -> Dict[str, Any]:
    """Evaluate model performance on a dataset.
    
    This function evaluates a trained model on a dataset (typically the test set)
    and calculates various performance metrics.
    
    Args:
        model: The trained TensorFlow model
        dataset_dir: Directory containing the evaluation dataset
        features_db: Optional dictionary of extracted features
        verbose: Whether to print detailed metrics
        save_dir: Optional directory to save evaluation results
        
    Returns:
        Dictionary containing evaluation metrics
    """
    # Create evaluation dataset
    eval_dataset, eval_size = create_dataset(dataset_dir, features_db, is_training=False)
    
    # Run basic evaluation with TensorFlow
    loss, accuracy = model.evaluate(eval_dataset, verbose=0)
    
    if verbose:
        print(f"\nBasic Evaluation Metrics:")
        print(f"  Loss: {loss:.4f}")
        print(f"  Accuracy: {accuracy:.4f}")
    
    # Collect predictions and true labels for detailed metrics
    true_labels = []
    predicted_labels = []
    confidence_scores = []
    
    for x, y in eval_dataset:
        predictions = model.predict(x, verbose=0)
        
        # Handle features if present
        if isinstance(x, tuple):
            batch_size = x[0].shape[0]
        else:
            batch_size = x.shape[0]
        
        for i in range(batch_size):
            pred = predictions[i]
            pred_label = np.argmax(pred)
            true_label = y[i].numpy()
            
            true_labels.append(true_label)
            predicted_labels.append(pred_label)
            confidence_scores.append(pred[pred_label])
    
    # Calculate metrics
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predicted_labels, average=None
    )
    
    # Calculate metrics per class
    metrics_per_class = {}
    for i, class_name in enumerate(config.EXPECTED_CLASSES):
        metrics_per_class[class_name] = {
            'precision': precision[i],
            'recall': recall[i],
            'f1_score': f1[i],
            'support': support[i]
        }
    
    # Calculate confusion statistics
    confusion_stats = analyze_confusion_matrix(conf_matrix)
    
    # Organize all metrics
    all_metrics = {
        'loss': loss,
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'class_metrics': metrics_per_class,
        'confusion_statistics': confusion_stats,
        'mean_confidence': np.mean(confidence_scores),
        'median_confidence': np.median(confidence_scores)
    }
    
    # Print detailed metrics if requested
    if verbose:
        print_detailed_metrics(all_metrics)
    
    # Save metrics if requested
    if save_dir:
        save_evaluation_results(all_metrics, save_dir)
    
    return all_metrics

def analyze_confusion_matrix(conf_matrix: np.ndarray) -> Dict[str, Any]:
    """Analyze the confusion matrix to extract useful statistics.
    
    Args:
        conf_matrix: Confusion matrix as a numpy array
        
    Returns:
        Dictionary of statistics derived from the confusion matrix
    """
    # Calculate class accuracies (diagonal elements divided by row sums)
    class_accuracies = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    
    # Calculate most common misclassifications
    misclassifications = []
    n_classes = conf_matrix.shape[0]
    
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and conf_matrix[i, j] > 0:
                misclassifications.append({
                    'true_class': config.EXPECTED_CLASSES[i],
                    'predicted_class': config.EXPECTED_CLASSES[j],
                    'count': int(conf_matrix[i, j]),
                    'percentage': float(conf_matrix[i, j] / np.sum(conf_matrix[i, :]) * 100)
                })
    
    # Sort misclassifications by count (descending)
    misclassifications.sort(key=lambda x: x['count'], reverse=True)
    
    # Calculate overall confusion statistics
    total_samples = np.sum(conf_matrix)
    correct_predictions = np.sum(np.diag(conf_matrix))
    incorrect_predictions = total_samples - correct_predictions
    
    # Package all statistics
    stats = {
        'class_accuracies': {
            config.EXPECTED_CLASSES[i]: float(class_accuracies[i]) 
            for i in range(n_classes)
        },
        'top_misclassifications': misclassifications[:5],  # Top 5 misclassifications
        'total_samples': int(total_samples),
        'correct_predictions': int(correct_predictions),
        'incorrect_predictions': int(incorrect_predictions),
        'accuracy': float(correct_predictions / total_samples)
    }
    
    return stats

def print_detailed_metrics(metrics: Dict[str, Any]) -> None:
    """Print detailed evaluation metrics in a readable format.
    
    Args:
        metrics: Dictionary of evaluation metrics
    """
    print("\n================ Detailed Evaluation Metrics ================")
    
    # Print overall metrics
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Mean Confidence: {metrics['mean_confidence']:.4f}")
    print(f"Median Confidence: {metrics['median_confidence']:.4f}")
    
    # Print confusion matrix
    print("\nConfusion Matrix:")
    conf_matrix = np.array(metrics['confusion_matrix'])
    print("True \\ Pred", end="  ")
    for class_name in config.EXPECTED_CLASSES:
        print(f"{class_name:^8}", end="")
    print()
    
    for i, class_name in enumerate(config.EXPECTED_CLASSES):
        print(f"{class_name:^10}", end="")
        for j in range(len(config.EXPECTED_CLASSES)):
            print(f"{conf_matrix[i, j]:^8}", end="")
        print()
    
    # Print per-class metrics
    print("\nPer-Class Metrics:")
    print(f"{'Class':^8} {'Precision':^10} {'Recall':^10} {'F1 Score':^10} {'Support':^10}")
    for class_name, metrics_dict in metrics['class_metrics'].items():
        print(f"{class_name:^8} {metrics_dict['precision']:^10.4f} "
              f"{metrics_dict['recall']:^10.4f} {metrics_dict['f1_score']:^10.4f} "
              f"{metrics_dict['support']:^10}")
    
    # Print top misclassifications
    print("\nTop Misclassifications:")
    for misclass in metrics['confusion_statistics']['top_misclassifications']:
        print(f"  {misclass['true_class']} → {misclass['predicted_class']}: "
              f"{misclass['count']} samples ({misclass['percentage']:.2f}%)")
    
    print("=============================================================")

def save_evaluation_results(
    metrics: Dict[str, Any],
    save_dir: str
) -> None:
    """Save evaluation results to files.
    
    This function saves the evaluation metrics, confusion matrix visualization,
    and other useful outputs to files for later reference.
    
    Args:
        metrics: Dictionary of evaluation metrics
        save_dir: Directory to save results
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save metrics as JSON
    import json
    metrics_path = os.path.join(save_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    conf_matrix = np.array(metrics['confusion_matrix'])
    
    # Create a more visually appealing confusion matrix
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add axis labels
    tick_marks = np.arange(len(config.EXPECTED_CLASSES))
    plt.xticks(tick_marks, config.EXPECTED_CLASSES, rotation=45)
    plt.yticks(tick_marks, config.EXPECTED_CLASSES)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    # Save the plot
    conf_matrix_path = os.path.join(save_dir, 'confusion_matrix.png')
    plt.savefig(conf_matrix_path)
    plt.close()
    
    # Plot and save confidence distribution
    plt.figure(figsize=(10, 6))
    
    # Compute confidence histograms per class
    class_accuracies = metrics['confusion_statistics']['class_accuracies']
    plt.bar(range(len(class_accuracies)), list(class_accuracies.values()))
    plt.xticks(range(len(class_accuracies)), list(class_accuracies.keys()))
    plt.ylim(0, 1.0)
    plt.ylabel('Accuracy')
    plt.xlabel('Class')
    plt.title('Accuracy by Class')
    
    # Save the plot
    class_accuracy_path = os.path.join(save_dir, 'class_accuracy.png')
    plt.savefig(class_accuracy_path)
    plt.close()
    
    print(f"Evaluation results saved to {save_dir}")

def visualize_misclassifications(
    model: tf.keras.Model,
    dataset_dir: str = config.TEST_DIR,
    features_db: Optional[Dict[str, Any]] = None,
    num_examples: int = 5,
    save_dir: Optional[str] = None
) -> None:
    """Visualize examples of misclassified images.
    
    This function helps understand where the model is making mistakes by
    visualizing misclassified examples with class activation maps.
    
    Args:
        model: The trained TensorFlow model
        dataset_dir: Directory containing the evaluation dataset
        features_db: Optional dictionary of extracted features
        num_examples: Number of misclassified examples to visualize
        save_dir: Optional directory to save visualizations
    """
    # Initialize lists to collect misclassified examples
    misclassified_paths = []
    misclassified_true_labels = []
    misclassified_pred_labels = []
    misclassified_confidences = []
    
    # Go through each class directory
    for class_idx, class_name in enumerate(config.EXPECTED_CLASSES):
        class_dir = os.path.join(dataset_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        # Process each image in the class directory
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
                
            img_path = os.path.join(class_dir, img_name)
            
            # Load and preprocess image
            img = tf.io.read_file(img_path)
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, config.IMAGE_SIZE)
            img = tf.cast(img, tf.float32) / 255.0
            
            # Extract features if available
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
                
                # Make prediction with features
                prediction = model.predict({
                    'image_input': tf.expand_dims(img, 0),
                    'feature_input': features
                }, verbose=0)[0]
            else:
                # Make prediction without features
                prediction = model.predict(tf.expand_dims(img, 0), verbose=0)[0]
            
            # Check if misclassified
            pred_class = np.argmax(prediction)
            confidence = prediction[pred_class]
            
            if pred_class != class_idx:
                misclassified_paths.append(img_path)
                misclassified_true_labels.append(class_idx)
                misclassified_pred_labels.append(pred_class)
                misclassified_confidences.append(confidence)
                
                # Break if we have enough examples
                if len(misclassified_paths) >= num_examples:
                    break
        
        # Break if we have enough examples
        if len(misclassified_paths) >= num_examples:
            break
    
    # Visualize the misclassified examples
    if not misclassified_paths:
        print("No misclassified examples found.")
        return
    
    print(f"\nVisualizing {len(misclassified_paths)} misclassified examples:")
    
    for i, (img_path, true_label, pred_label, confidence) in enumerate(
        zip(misclassified_paths, misclassified_true_labels, 
            misclassified_pred_labels, misclassified_confidences)):
        
        print(f"Example {i+1}:")
        print(f"  Path: {img_path}")
        print(f"  True class: {config.EXPECTED_CLASSES[true_label]}")
        print(f"  Predicted class: {config.EXPECTED_CLASSES[pred_label]}")
        print(f"  Confidence: {confidence:.4f}")
        
        # Generate visualization
        if save_dir:
            vis_path = os.path.join(save_dir, f"misclassified_{i+1}.png")
            visualize_prediction(model, img_path, true_label, vis_path)
        else:
            visualize_prediction(model, img_path, true_label)