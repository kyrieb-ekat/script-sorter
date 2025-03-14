# dataset creation and loading

"""
Dataset creation, loading, and validation functions for manuscript classification.
"""
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import config

def verify_data_directories() -> Dict[str, int]:
    """Verify directory structure and count images in each class.
    
    This function checks that all expected class directories exist 
    and counts the number of valid images in each class.
    
    Returns:
        Dictionary mapping class names to image counts
    """
    class_counts = {}
    for directory in [config.TRAIN_DIR, config.VAL_DIR, config.TEST_DIR]:
        dir_name = os.path.basename(directory)
        print(f"\nChecking {dir_name} directory:")
        class_dirs = sorted(os.listdir(directory))
        print(f"Found classes: {class_dirs}")

        if not all(cls in class_dirs for cls in config.EXPECTED_CLASSES):
            raise ValueError(f"Missing some class directories in {dir_name}!")

        for class_name in class_dirs:
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                n_images = len([f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = n_images
                print(f"Class {class_name}: {n_images} images")
        
        # Verify minimum samples per class
        verify_dataset_size(directory)

    return class_counts

def verify_dataset_size(directory: str, min_samples: int = config.MIN_SAMPLES_PER_CLASS) -> None:
    """Verify dataset has enough samples per class and check image integrity.
    
    Args:
        directory: Path to the dataset directory
        min_samples: Minimum required samples per class
        
    Raises:
        ValueError: If any class has fewer than min_samples valid images
    """
    print(f"\nVerifying minimum samples in {os.path.basename(directory)}...")
    for class_name in config.EXPECTED_CLASSES:
        class_path = os.path.join(directory, class_name)
        valid_images = []
        for f in os.listdir(class_path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, f)
                try:
                    # Try to open the image to verify it's not corrupted
                    with Image.open(img_path) as img:
                        img.verify()
                    valid_images.append(f)
                except Exception as e:
                    print(f"Warning: Corrupted or invalid image found: {img_path}")
                    print(f"Error: {str(e)}")
                    continue

        n_samples = len(valid_images)
        if n_samples < min_samples:
            raise ValueError(
                f"Class {class_name} in {os.path.basename(directory)} "
                f"has insufficient samples: {n_samples}. "
                f"Minimum required: {min_samples}"
            )

def preprocess_image(path: str) -> tf.Tensor:
    """Read and preprocess a single image.
    
    Args:
        path: Path to the image file
        
    Returns:
        Preprocessed image as a TensorFlow tensor
        
    Raises:
        RuntimeError: If image processing fails
    """
    try:
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, config.IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image
    except Exception as e:
        print(f"Error processing image {path}: {str(e)}")
        raise RuntimeError(f"Failed to process image {path}: {str(e)}")

def augment(image: tf.Tensor) -> tf.Tensor:
    """Apply data augmentation to an image.
    
    This function applies random transformations to increase 
    dataset variety during training.
    
    Args:
        image: Input image tensor
        
    Returns:
        Augmented image tensor
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.clip_by_value(image, 0, 1)
    return image

def create_dataset(
    directory: str, 
    features_db: Optional[Dict[str, Any]] = None, 
    is_training: bool = False
) -> Tuple[tf.data.Dataset, int]:
    """Create a TensorFlow dataset from directory.
    
    This function builds a tf.data.Dataset with optional augmentation 
    and extracted features integration.
    
    Args:
        directory: Path to the data directory
        features_db: Optional dictionary of extracted features
        is_training: Whether this is a training dataset (enables augmentation)
        
    Returns:
        A tuple containing (tf.data.Dataset, number_of_samples)
    """
    # Get all image paths and labels
    image_paths = []
    labels = []
    extracted_features = []
    
    for class_idx, class_name in enumerate(config.EXPECTED_CLASSES):
        class_path = os.path.join(directory, class_name)
        class_images = [os.path.join(class_path, fname) 
                       for fname in os.listdir(class_path)
                       if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))
        
        # Add extracted features if available
        if features_db:
            for path in class_images:
                if path in features_db:
                    # Create feature vector 
                    features_dict = features_db[path]
                    feature_vector = [
                        features_dict['staffline_confidence'],
                        features_dict['layout_features']['text_ratio'],
                        features_dict['layout_features']['top_margin'],
                        features_dict['layout_features']['bottom_margin'],
                        features_dict['layout_features']['left_margin'],
                        features_dict['layout_features']['right_margin']
                    ]
                    extracted_features.append(feature_vector)
                else:
                    # Use default values if features not available
                    extracted_features.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    # Create dataset (with or without features)
    if features_db and extracted_features:
        dataset = tf.data.Dataset.from_tensor_slices((
            image_paths, extracted_features, labels))
        
        # Map function for dataset with features
        def process_with_features(path, features, label):
            image = preprocess_image(path)
            if is_training:
                image = augment(image)
            return (image, features), label
        
        dataset = dataset.map(
            process_with_features, 
            num_parallel_calls=tf.data.AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
        
        # Map function for dataset without features
        def process_without_features(path, label):
            image = preprocess_image(path)
            if is_training:
                image = augment(image)
            return image, label
        
        dataset = dataset.map(
            process_without_features, 
            num_parallel_calls=tf.data.AUTOTUNE)

    # Common dataset preparation
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(config.BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset, len(image_paths)

def prepare_fold_data(
    train_paths: np.ndarray, 
    train_labels: np.ndarray, 
    val_paths: np.ndarray, 
    val_labels: np.ndarray, 
    fold: int
) -> Tuple[str, str]:
    """Create temporary directories for current fold.
    
    Args:
        train_paths: Array of training image paths
        train_labels: Array of training labels
        val_paths: Array of validation image paths
        val_labels: Array of validation labels
        fold: Current fold number
        
    Returns:
        Tuple of (train_dir_path, val_dir_path) for this fold
    """
    import shutil
    
    fold_dir = f'fold_data/fold_{fold}'
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