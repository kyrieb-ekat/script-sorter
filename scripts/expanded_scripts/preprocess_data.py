# preprocess_data.py - Main preprocessing script that calls preprocessing modules

import argparse
import os
import sys
from typing import Optional, Dict, Any
from pathlib import Path

from preprocessing.normalize import normalize_images
from preprocessing.augmentation import generate_augmented_data, create_custom_augmentations
from preprocessing.feature_extraction import extract_and_save_features
from preprocessing.preprocess_main import split_dataset, process_sequential_pages
import config

def main() -> None:
    """Execute the main preprocessing pipeline.
    
    This function orchestrates the complete preprocessing workflow:
    1. Parse command-line arguments
    2. Split the dataset into train/validation/test sets (optional)
    3. Normalize images for consistent size and contrast
    4. Generate augmented data to expand the training set (optional)
    5. Create specialized augmentations for the rejection class
    6. Extract and save features for use in the dual-pathway model
    7. Process sequential pages to create linkage features
    """
    parser = argparse.ArgumentParser(description='Preprocess manuscript dataset')
    parser.add_argument('--raw_dir', type=str, default=os.path.join(config.DATASET_DIR, 'raw_dataset'),
                      help='Directory containing raw dataset')
    parser.add_argument('--output_dir', type=str, default=config.DATASET_DIR,
                      help='Output directory for processed data')
    parser.add_argument('--skip_split', action='store_true',
                      help='Skip the dataset splitting step')
    parser.add_argument('--skip_augmentation', action='store_true',
                      help='Skip data augmentation step')
    parser.add_argument('--skip_features', action='store_true',
                      help='Skip feature extraction step')
    parser.add_argument('--train_size', type=float, default=0.7,
                      help='Proportion of data to use for training (default: 0.7)')
    parser.add_argument('--val_size', type=float, default=0.15,
                      help='Proportion of data to use for validation (default: 0.15)')
    args = parser.parse_args()
    
    # Validate directories
    validate_directories(args.raw_dir, args.output_dir)
    
    try:
        # Step 1: Split dataset
        if not args.skip_split:
            print("Splitting dataset into train, validation, and test sets...")
            split_dataset(args.raw_dir, args.output_dir, 
                          train_size=args.train_size, 
                          val_size=args.val_size)
        
        # Step 2: Normalize images
        print("Normalizing images for consistent size and contrast...")
        normalize_images(args.output_dir, target_size=config.IMAGE_SIZE)
        
        # Step 3: Generate augmented data for training
        if not args.skip_augmentation:
            print("Generating augmented data for training...")
            generate_augmented_data(args.output_dir)
            
            # Step 4: Create specialized augmentations for class 4
            print("Creating specialized augmentations for class 4...")
            create_custom_augmentations(args.output_dir)
        
        # Step 5: Extract and save features
        if not args.skip_features:
            print("Extracting features from images...")
            features_db = extract_and_save_features(args.output_dir)
            print(f"Extracted features for {len(features_db) if features_db else 0} images")
        
        # Step 6: Process sequential pages if available
        print("Processing sequential pages...")
        process_sequential_pages(args.raw_dir, args.output_dir)
        
        print("Preprocessing complete!")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

def validate_directories(raw_dir: str, output_dir: str) -> None:
    """Validate that required directories exist or can be created.
    
    Args:
        raw_dir: Path to directory containing raw dataset
        output_dir: Path to output directory for processed data
        
    Raises:
        ValueError: If raw_dir doesn't exist or output_dir can't be created
    """
    # Check raw directory exists
    if not os.path.exists(raw_dir):
        raise ValueError(f"Raw dataset directory does not exist: {raw_dir}")
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            raise ValueError(f"Failed to create output directory {output_dir}: {e}")
    
    # Create subdirectories if they don't exist
    for subdir in ['train', 'validation', 'test']:
        subdir_path = os.path.join(output_dir, subdir)
        if not os.path.exists(subdir_path) and not args.skip_split:
            try:
                os.makedirs(subdir_path)
            except OSError as e:
                raise ValueError(f"Failed to create {subdir} directory: {e}")

if __name__ == "__main__":
    sys.exit(main())