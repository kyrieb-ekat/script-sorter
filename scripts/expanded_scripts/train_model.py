# train_model.py - Main training script that calls training modules

import argparse
import os
import tensorflow as tf
import pickle
from typing import Dict, Any, Optional
from pathlib import Path

from training.dataset import verify_data_directories, create_dataset
from training.train import train_with_kfold
from models.dual_pathway import create_dual_pathway_model
import config
from utils.memory_monitor import get_available_memory, optimize_batch_size

def main() -> None:
    """Execute the main training pipeline.
    
    This function handles the complete training workflow:
    1. Parse command-line arguments
    2. Optimize batch size based on available memory
    3. Verify the dataset structure and integrity
    4. Load extracted features if available
    5. Train the model using K-fold cross-validation
    """
    parser = argparse.ArgumentParser(description='Train manuscript classification model')
    parser.add_argument('--architecture', type=str, default='resnet50',
                      choices=['vgg16', 'resnet50'], 
                      help='Base model architecture')
    parser.add_argument('--no_features', action='store_true',
                      help='Disable use of extracted features')
    parser.add_argument('--folds', type=int, default=config.NUM_FOLDS,
                      help='Number of folds for cross-validation')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                      help='Maximum number of training epochs')
    args = parser.parse_args()
    
    # Update batch size based on available memory
    config.BATCH_SIZE = optimize_batch_size(
        get_available_memory(), config.IMAGE_SIZE)
    print(f"Optimized batch size: {config.BATCH_SIZE}")
    
    # Verify dataset
    print("Verifying dataset requirements...")
    class_counts = verify_data_directories()
    
    # Load preprocessed features if available
    features_db = load_feature_database(args.no_features)
    
    # Train model with K-fold cross validation
    use_features = features_db is not None and not args.no_features
    histories = train_with_kfold(
        architecture=args.architecture,
        use_features=use_features, 
        features_db=features_db,
        num_folds=args.folds,
        epochs=args.epochs
    )
    
    print("Training complete!")

def load_feature_database(disable_features: bool = False) -> Optional[Dict[str, Any]]:
    """Load the extracted features database if available.
    
    Args:
        disable_features: If True, skip loading features even if available
        
    Returns:
        Dictionary of extracted features or None if unavailable/disabled
    """
    features_db = None
    
    if disable_features:
        print("Feature usage disabled via command line argument.")
        return None
        
    if os.path.exists(config.FEATURES_DB):
        print("Loading extracted features database...")
        try:
            with open(config.FEATURES_DB, 'rb') as f:
                features_db = pickle.load(f)
            print(f"Loaded features for {len(features_db)} images")
        except Exception as e:
            print(f"Error loading features database: {e}")
            print("Continuing without extracted features.")
    else:
        print("No feature database found. Using image classification only.")
        
    return features_db

if __name__ == "__main__":
    main()