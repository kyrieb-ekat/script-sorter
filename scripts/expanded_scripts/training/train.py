#training loop and k-fold implementation
#handles core training functionality

# train.py - Model training functions for manuscript classification

import os
import gc
import datetime
import numpy as np
import tensorflow as tf
import Keras
from typing import List, Dict, Any, Optional, Tuple, Union
import shutil
import pickle
from pathlib import Path
from sklearn.model_selection import KFold

from utils.memory_monitor import MemoryMonitor
from models.dual_pathway import create_dual_pathway_model
from models.base_model import create_model
import config

def train_with_kfold(
    architecture: str = 'resnet50',
    use_features: bool = False,
    features_db: Optional[Dict[str, Any]] = None,
    num_folds: int = config.NUM_FOLDS,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE
) -> List[Dict[str, List[float]]]:
    """Train model using K-fold cross validation.
    
    This function implements K-fold cross-validation to provide a robust
    evaluation of model performance. For each fold, it:
    1. Splits the dataset into training and validation sets
    2. Creates a new model instance
    3. Trains the model on the training set
    4. Evaluates the model on the validation set
    
    Args:
        architecture: Base model architecture ('vgg16' or 'resnet50')
        use_features: Whether to use extracted features pathway
        features_db: Dictionary of extracted features
        num_folds: Number of folds for cross-validation
        epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        List of training histories for each fold
    """
    # Memory cleanup before training
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Get all image paths and labels
    image_paths = []
    labels = []
    for class_idx, class_name in enumerate(config.EXPECTED_CLASSES):
        class_path = os.path.join(config.TRAIN_DIR, class_name)
        class_images = [os.path.join(class_path, fname) 
                       for fname in os.listdir(class_path)
                       if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Initialize K-fold
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
    fold_histories = []

    # K-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        print(f"\nTraining fold {fold + 1}/{num_folds}")
        
        # Prepare data for this fold
        train_paths, val_paths = image_paths[train_idx], image_paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        # Create temporary directories and datasets for this fold
        fold_train_dir, fold_val_dir = prepare_fold_data(
            train_paths, train_labels, val_paths, val_labels, fold)
        
        # Create datasets with or without features
        from training.dataset import create_dataset
        train_dataset, train_size = create_dataset(fold_train_dir, features_db, is_training=True)
        val_dataset, val_size = create_dataset(fold_val_dir, features_db, is_training=False)
        
        # Create model based on whether we're using features
        if use_features:
            model = create_dual_pathway_model(architecture, use_features=True)
        else:
            model = create_model(architecture)
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Set up callbacks
        callbacks = setup_training_callbacks(fold)
        
        # Train the model
        history = model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks
        )
        
        fold_histories.append(history.history)
        
        # Calculate and print metrics for this fold
        val_loss = min(history.history['val_loss'])
        val_acc = max(history.history['val_accuracy'])
        print(f"\nFold {fold + 1} results:")
        print(f"  Best validation loss: {val_loss:.4f}")
        print(f"  Best validation accuracy: {val_acc:.4f}")
        
        # Clean up fold data
        clean_fold_data(fold)
    
    # Calculate and print final cross-validation metrics
    print_cross_validation_results(fold_histories)
    
    return fold_histories

def prepare_fold_data(
    train_paths: np.ndarray, 
    train_labels: np.ndarray, 
    val_paths: np.ndarray, 
    val_labels: np.ndarray, 
    fold: int
) -> Tuple[str, str]:
    """Create temporary directories for current fold.
    
    For each fold of cross-validation, this function creates temporary directories
    for the training and validation datasets, copying the appropriate images to each.
    
    Args:
        train_paths: Paths to training images
        train_labels: Labels for training images
        val_paths: Paths to validation images
        val_labels: Labels for validation images
        fold: Current fold number
        
    Returns:
        Tuple of (train_dir_path, val_dir_path) for this fold
    """
    fold_dir = os.path.join('fold_data', f'fold_{fold}')
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
    
    for split in ['train', 'val']:
        for class_name in config.EXPECTED_CLASSES:
            os.makedirs(os.path.join(fold_dir, split, class_name), exist_ok=True)
    
    # Copy files to temporary locations
    for path, label in zip(train_paths, train_labels):
        dest = os.path.join(fold_dir, 'train', str(label), 
                           os.path.basename(path))
        shutil.copy2(path, dest)
    
    for path, label in zip(val_paths, val_labels):
        dest = os.path.join(fold_dir, 'val', str(label), 
                           os.path.basename(path))
        shutil.copy2(path, dest)
    
    return os.path.join(fold_dir, 'train'), os.path.join(fold_dir, 'val')

def setup_training_callbacks(fold: int) -> List[tf.keras.callbacks.Callback]:
    """Set up callbacks for model training.
    
    This function creates callbacks for early stopping, model checkpointing,
    TensorBoard logging, and memory monitoring.
    
    Args:
        fold: Current fold number for naming saved models
        
    Returns:
        List of TensorFlow callbacks
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create directory for saving models if it doesn't exist
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    # Create directory for TensorBoard logs if it doesn't exist
    logs_dir = os.path.join(config.LOGS_DIR, f"fold_{fold + 1}_{timestamp}")
    os.makedirs(logs_dir, exist_ok=True)
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODELS_DIR, f'fold_{fold + 1}_{timestamp}.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=logs_dir,
            histogram_freq=1
        ),
        MemoryMonitor(print_interval=1)
    ]
    
    return callbacks

def clean_fold_data(fold: int) -> None:
    """Clean up temporary fold data after training.
    
    Args:
        fold: Fold number to clean up
    """
    fold_dir = os.path.join('fold_data', f'fold_{fold}')
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
        print(f"Cleaned up temporary data for fold {fold + 1}")

def print_cross_validation_results(histories: List[Dict[str, List[float]]]) -> None:
    """Calculate and print cross-validation metrics.
    
    Args:
        histories: List of training histories for each fold
    """
    # Extract final validation metrics from each fold
    final_val_losses = [h['val_loss'][-1] for h in histories]
    final_val_accuracies = [h['val_accuracy'][-1] for h in histories]
    
    # Extract best validation metrics from each fold
    best_val_losses = [min(h['val_loss']) for h in histories]
    best_val_accuracies = [max(h['val_accuracy']) for h in histories]
    
    # Calculate statistics
    mean_final_loss = np.mean(final_val_losses)
    std_final_loss = np.std(final_val_losses)
    mean_final_acc = np.mean(final_val_accuracies)
    std_final_acc = np.std(final_val_accuracies)
    
    mean_best_loss = np.mean(best_val_losses)
    std_best_loss = np.std(best_val_losses)
    mean_best_acc = np.mean(best_val_accuracies)
    std_best_acc = np.std(best_val_accuracies)
    
    # Print results
    print("\n========== Cross-Validation Results ==========")
    print(f"Final metrics (last epoch):")
    print(f"  Validation Loss: {mean_final_loss:.4f} ± {std_final_loss:.4f}")
    print(f"  Validation Accuracy: {mean_final_acc:.4f} ± {std_final_acc:.4f}")
    
    print(f"\nBest metrics (best epoch for each fold):")
    print(f"  Validation Loss: {mean_best_loss:.4f} ± {std_best_loss:.4f}")
    print(f"  Validation Accuracy: {mean_best_acc:.4f} ± {std_best_acc:.4f}")
    print("=============================================")

def train_final_model(
    architecture: str = 'resnet50',
    use_features: bool = False,
    features_db: Optional[Dict[str, Any]] = None,
    epochs: int = config.EPOCHS,
    learning_rate: float = config.LEARNING_RATE
) -> tf.keras.Model:
    """Train a final model on the full training dataset.
    
    After performing cross-validation, this function can be used to train
    a final model on the entire training dataset for deployment.
    
    Args:
        architecture: Base model architecture ('vgg16' or 'resnet50')
        use_features: Whether to use extracted features pathway
        features_db: Dictionary of extracted features
        epochs: Maximum number of training epochs
        learning_rate: Learning rate for optimizer
        
    Returns:
        The trained TensorFlow model
    """
    # Memory cleanup
    gc.collect()
    tf.keras.backend.clear_session()
    
    # Create datasets
    from training.dataset import create_dataset
    train_dataset, train_size = create_dataset(config.TRAIN_DIR, features_db, is_training=True)
    val_dataset, val_size = create_dataset(config.VAL_DIR, features_db, is_training=False)
    
    # Create model based on whether we're using features
    if use_features:
        model = create_dual_pathway_model(architecture, use_features=True)
    else:
        model = create_model(architecture)
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(config.MODELS_DIR, f'final_model_{timestamp}.keras'),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(config.LOGS_DIR, f"final_model_{timestamp}"),
            histogram_freq=1
        ),
        MemoryMonitor(print_interval=1)
    ]
    
    # Train the model
    print("\nTraining final model on full training dataset...")
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # Evaluate on validation set
    val_loss, val_acc = model.evaluate(val_dataset)
    print(f"\nFinal model validation metrics:")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Accuracy: {val_acc:.4f}")
    
    # Save the final model
    final_model_path = os.path.join(config.MODELS_DIR, f'final_model_{timestamp}.keras')
    model.save(final_model_path)
    print(f"Final model saved to: {final_model_path}")
    
    return model