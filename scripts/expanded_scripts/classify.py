# classify.py - Main classification script that calls interface modules

import argparse
import os
import sys
import pickle
import tensorflow as tf
from typing import Dict, Any, Optional
from pathlib import Path

from interface.classifier import classify_new_image
from interface.disambiguation import handle_user_disambiguation
from interface.feedback import save_user_feedback
import config

def main() -> int:
    """Execute the manuscript classification process.
    
    This function handles:
    1. Loading a trained model
    2. Classifying the provided image
    3. Displaying confidence scores
    4. Managing user disambiguation for low-confidence predictions
    5. Saving user feedback for model improvement
    
    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    parser = argparse.ArgumentParser(description='Classify manuscript images')
    parser.add_argument('--image', type=str, required=True,
                      help='Path to the image to classify')
    parser.add_argument('--model', type=str, default=None,
                      help='Path to the model file (defaults to latest in models directory)')
    parser.add_argument('--threshold', type=float, default=config.CONFIDENCE_THRESHOLD,
                      help='Confidence threshold for disambiguation')
    parser.add_argument('--features', type=str, default=config.FEATURES_DB,
                      help='Path to features database (defaults to configuration setting)')
    parser.add_argument('--save-visualization', action='store_true',
                      help='Save visualization of model decision process')
    args = parser.parse_args()
    
    try:
        # Verify image exists
        if not os.path.exists(args.image):
            raise ValueError(f"Image file not found: {args.image}")
            
        # Find latest model if not specified
        if args.model is None:
            model_path = find_latest_model()
            if model_path is None:
                raise ValueError("No model files found in models directory")
            args.model = model_path
        
        # Load model
        print(f"Loading model from {args.model}...")
        try:
            model = tf.keras.models.load_model(args.model)
        except Exception as e:
            raise ValueError(f"Failed to load model: {str(e)}")
        
        # Load features if available
        features_db = load_features_database(args.features)
        
        # Classify image
        print(f"Classifying image {args.image}...")
        result = classify_new_image(
            model, 
            args.image, 
            features_db, 
            args.threshold,
            save_visualization=args.save_visualization
        )
        
        # Print results with class descriptions
        display_classification_results(result)
        
        # Handle disambiguation if needed
        if result['needs_user_input']:
            handle_disambiguation_process(model, args.image, features_db)
        
        print("\nClassification complete!")
        return 0
        
    except Exception as e:
        print(f"Error during classification: {str(e)}", file=sys.stderr)
        return 1

def find_latest_model() -> Optional[str]:
    """Find the most recently created model file.
    
    Returns:
        Path to the latest model file or None if no models found
    """
    if not os.path.exists(config.MODELS_DIR):
        return None
        
    model_files = [
        os.path.join(config.MODELS_DIR, f) 
        for f in os.listdir(config.MODELS_DIR) 
        if f.endswith(('.keras', '.h5'))
    ]
    
    if not model_files:
        return None
        
    # Sort by modification time, most recent last
    model_files.sort(key=os.path.getmtime)
    return model_files[-1]

def load_features_database(features_path: str) -> Optional[Dict[str, Any]]:
    """Load the features database if available.
    
    Args:
        features_path: Path to the features database file
        
    Returns:
        Dictionary of extracted features or None if unavailable
    """
    features_db = None
    if os.path.exists(features_path):
        print(f"Loading features database from {features_path}...")
        try:
            with open(features_path, 'rb') as f:
                features_db = pickle.load(f)
            print(f"Loaded features for {len(features_db)} images")
        except Exception as e:
            print(f"Warning: Failed to load features database: {str(e)}")
            print("Continuing without extracted features.")
    else:
        print("Features database not found. Continuing without extracted features.")
    
    return features_db

def display_classification_results(result: Dict[str, Any]) -> None:
    """Display the classification results in a user-friendly format.
    
    Args:
        result: Classification result dictionary
    """
    class_name = result['class_name']
    confidence = result['confidence']
    
    print("\n======= Classification Results =======")
    print(f"Predicted class: {class_name}")
    if class_name in config.CLASS_DESCRIPTIONS:
        print(f"Description: {config.CLASS_DESCRIPTIONS[class_name]}")
    print(f"Confidence: {confidence:.2f}")
    
    print("\nConfidence scores for all classes:")
    # Sort by confidence score, highest first
    sorted_scores = sorted(
        result['confidence_scores'].items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    for class_name, score in sorted_scores:
        description = config.CLASS_DESCRIPTIONS.get(class_name, "")
        print(f"  Class {class_name}: {score:.4f}" + (f" - {description}" if description else ""))
    
    print("=====================================")

def handle_disambiguation_process(
    model: tf.keras.Model, 
    img_path: str,
    features_db: Optional[Dict[str, Any]] = None
) -> None:
    """Handle the disambiguation process for low-confidence predictions.
    
    Args:
        model: The trained model
        img_path: Path to the image file
        features_db: Optional dictionary of extracted features
    """
    print("\nConfidence below threshold. Disambiguation needed.")
    disambiguation = handle_user_disambiguation(model, img_path, features_db=features_db)
    
    print("\nPlease select from the following options:")
    for i, (class_name, score) in enumerate(disambiguation['top_options']):
        description = config.CLASS_DESCRIPTIONS.get(class_name, "")
        print(f"  {i+1}. Class {class_name} (confidence: {score:.4f})" + 
              (f" - {description}" if description else ""))
    
    # In a real application, you would get user input
    # For this script implementation, we'll request input from the console
    try:
        selected_index = int(input("\nEnter your selection (1-" + 
                                  str(len(disambiguation['top_options'])) + "): ")) - 1
        
        if 0 <= selected_index < len(disambiguation['top_options']):
            selected_class = disambiguation['top_options'][selected_index][0]
            
            # Save the feedback
            save_user_feedback(img_path, selected_class)
            print(f"\nSelected class: {selected_class}")
            if selected_class in config.CLASS_DESCRIPTIONS:
                print(f"Description: {config.CLASS_DESCRIPTIONS[selected_class]}")
        else:
            print("Invalid selection. Using top prediction.")
            selected_class = disambiguation['top_options'][0][0]
    except (ValueError, IndexError):
        print("Invalid input. Using top prediction.")
        selected_class = disambiguation['top_options'][0][0]

if __name__ == "__main__":
    sys.exit(main())