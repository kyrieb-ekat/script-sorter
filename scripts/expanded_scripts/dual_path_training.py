#this script differs from train.py and dual_pathway.py in that this script
#explicitly is dedicated to the dual-pathway architecture handling

#!/usr/bin/env python3
# dual_path_training.py - Script for training the dual pathway manuscript classification model

import os
import argparse
import pickle
import datetime
import tensorflow as tf
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

# Import from your project modules
from models.dual_pathway import create_dual_pathway_model
from training.dataset import create_dataset, verify_data_directories
from utils.memory_monitor import MemoryMonitor, optimize_batch_size, get_available_memory
import config

def train_dual_pathway_model(
    architecture: str = 'resnet50',
    use_features: bool = True,
    features_db: Optional[Dict[str, Any]] = None,
    epochs: int = 20,
    learning_rate: float = 0.0001,
    custom_batch_size: Optional[int] = None
) -> tf.keras.Model:
    """Train the dual-pathway model on the full training dataset.
    
    Args:
        architecture: Base model architecture ('vgg16' or 'resnet50')
        use_features: Whether to use the extracted features pathway
        features_db: Dictionary of extracted features
        epochs: Maximum number of training epochs
        learning_rate: Learning rate for the optimizer
        custom_batch_size: Optional custom batch size (otherwise auto-calculated)
        
    Returns:
        The trained TensorFlow model
    """
    # Set up batch size
    if custom_batch_size:
        batch_size = custom_batch_size
    else:
        available_memory = get_available_memory()
        batch_size = optimize_batch_size(available_memory, config.IMAGE_SIZE)
        
    print(f"Using batch size: {batch_size}")
        
    # Save original batch size and temporarily set to our calculated value
    original_batch_size = config.BATCH_SIZE
    config.BATCH_SIZE = batch_size
    
    # Create datasets
    print("Creating training and validation datasets...")
    train_dataset, train_size = create_dataset(
        config.TRAIN_DIR, features_db, is_training=True)
    val_dataset, val_size = create_dataset(
        config.VAL_DIR, features_db, is_training=False)
    
    # Create model
    print(f"Creating dual-pathway model with {architecture} base architecture...")
    model = create_dual_pathway_model(
        architecture=architecture,
        use_features=use_features and features_db is not None,
        num_features=6  # Match the feature vector size in your dataset.py
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(config.MODELS_DIR, 'dual_pathway')
    os.makedirs(model_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, f'model_{timestamp}.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, f'dual_pathway_{timestamp}'),
            histogram_freq=1
        ),
        MemoryMonitor(print_interval=1)
    ]
    
    # Train model
    print(f"Training dual-pathway model for up to {epochs} epochs...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # Restore original batch size
    config.BATCH_SIZE = original_batch_size
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    test_dataset, test_size = create_dataset(
        config.TEST_DIR, features_db, is_training=False)
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save final model
    final_model_path = os.path.join(model_dir, f'final_model_{timestamp}.keras')
    model.save(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    return model

def load_feature_database(features_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Load the extracted features database.
    
    Args:
        features_path: Path to the features database file
                       (defaults to config.FEATURES_DB)
        
    Returns:
        Dictionary of extracted features or None if unavailable
    """
    if features_path is None:
        features_path = config.FEATURES_DB
        
    if not os.path.exists(features_path):
        print(f"Features database not found at {features_path}")
        return None
        
    try:
        print(f"Loading features database from {features_path}...")
        with open(features_path, 'rb') as f:
            features_db = pickle.load(f)
        print(f"Loaded features for {len(features_db)} images")
        return features_db
    except Exception as e:
        print(f"Error loading features database: {e}")
        return None

def main() -> None:
    """Execute dual-pathway model training.
    
    This script handles training a dual-pathway model that combines:
    1. A CNN-based image pathway
    2. A specialized neume detection pathway
    3. An optional extracted features pathway
    """
    parser = argparse.ArgumentParser(description='Train dual-pathway manuscript classification model')
    
    parser.add_argument('--architecture', type=str, default='resnet50',
                      choices=['vgg16', 'resnet50'], 
                      help='Base model architecture')
    parser.add_argument('--no_features', action='store_true',
                      help='Disable use of extracted features')
    parser.add_argument('--features_path', type=str, default=None,
                      help='Path to features database (defaults to config setting)')
    parser.add_argument('--batch_size', type=int, default=None,
                      help='Custom batch size (defaults to auto-calculated)')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Maximum number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                      help='Learning rate for the optimizer')
    
    args = parser.parse_args()
    
    # Verify dataset structure
    print("Verifying dataset requirements...")
    verify_data_directories()
    
    # Load features database
    features_db = None
    if not args.no_features:
        features_db = load_feature_database(args.features_path)
    
    # Train model
    model = train_dual_pathway_model(
        architecture=args.architecture,
        use_features=not args.no_features and features_db is not None,
        features_db=features_db,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        custom_batch_size=args.batch_size
    )
    
    print("Dual-pathway model training complete!")

if __name__ == "__main__":
    main()