import os
import shutil
import random
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as A  # For advanced augmentation

def split_dataset(dataset_dir, output_dir, train_size=0.7, val_size=0.15):
    # splitting code...
    
    for subset in ['train', 'validation', 'test']:
        subset_dir = os.path.join(output_dir, subset)
        if not os.path.exists(subset_dir):
            os.makedirs(subset_dir)
    
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        
        # Split dataset into train, validation, and test
        train, temp = train_test_split(images, train_size=train_size, random_state=42)
        val, test = train_test_split(temp, test_size=(val_size / (1 - train_size)), random_state=42)
        
        for subset, subset_name in [(train, "train"), (val, "validation"), (test, "test")]:
            subset_dir = os.path.join(output_dir, subset_name, class_dir)
            os.makedirs(subset_dir, exist_ok=True)
            for img in subset:
                try:
                    shutil.copy(img, subset_dir)
                except Exception as e:
                    print(f"Error copying {img} to {subset_dir}: {e}")


def normalize_images(output_dir, target_size=(512, 512)):
    """Normalize all images in the dataset for consistent size and contrast."""
    for subset in ['train', 'validation', 'test']:
        subset_path = os.path.join(output_dir, subset)
        for class_dir in os.listdir(subset_path):
            class_path = os.path.join(subset_path, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Could not read {img_path}")
                        continue
                        
                    # Convert to grayscale if not already
                    if len(img.shape) == 3:
                        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = img
                        
                    # Apply adaptive histogram equalization for better contrast
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    normalized = clahe.apply(gray)
                    
                    # Resize to target dimensions
                    resized = cv2.resize(normalized, target_size, interpolation=cv2.INTER_AREA)
                    
                    # Save the normalized image, overwriting the original
                    cv2.imwrite(img_path, resized)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")


def detect_stafflines(img_path):
    """Detect presence of stafflines in an image and return a confidence score."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
        
    # Apply horizontal line detection
    # Method 1: Using Hough Transform
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, 
                           minLineLength=100, maxLineGap=10)
    
    if lines is None:
        return 0.0
        
    # Count horizontal lines (roughly horizontal with small angle tolerance)
    horizontal_count = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
        
        # Lines that are roughly horizontal (within 5 degrees)
        if angle < 5 or angle > 175:
            horizontal_count += 1
    
    # Simple scoring based on number of horizontal lines
    # This would need fine-tuning based on your specific data
    confidence = min(1.0, horizontal_count / 10)  # Assume 10+ horizontal lines is very likely stafflines
    
    return confidence


def extract_layout_features(img_path):
    """Extract layout features like text-space ratio, margins, etc."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
        
    features = {}
    
    # Calculate text-to-space ratio (using simple thresholding)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    text_pixels = np.sum(binary > 0)
    total_pixels = img.shape[0] * img.shape[1]
    features['text_ratio'] = text_pixels / total_pixels
    
    # Calculate margin sizes (approximation)
    # Horizontal projection to find top and bottom margins
    h_proj = np.sum(binary, axis=1)
    v_proj = np.sum(binary, axis=0)
    
    # Find first and last non-zero rows/columns
    non_zero_rows = np.where(h_proj > 0)[0]
    non_zero_cols = np.where(v_proj > 0)[0]
    
    if len(non_zero_rows) > 0 and len(non_zero_cols) > 0:
        top_margin = non_zero_rows[0] / img.shape[0]
        bottom_margin = (img.shape[0] - non_zero_rows[-1]) / img.shape[0]
        left_margin = non_zero_cols[0] / img.shape[1]
        right_margin = (img.shape[1] - non_zero_cols[-1]) / img.shape[1]
        
        features['top_margin'] = top_margin
        features['bottom_margin'] = bottom_margin
        features['left_margin'] = left_margin
        features['right_margin'] = right_margin
    
    return features


def generate_augmented_data(output_dir, augmentation_factor=3):
    """Generate augmented data for training set only."""
    train_dir = os.path.join(output_dir, 'train')
    
    # Define augmentation pipeline
    transform = A.Compose([
        A.RandomRotate(limit=5, p=0.5),                # Slight rotation
        A.RandomBrightnessContrast(p=0.5),             # Brightness/contrast
        A.GaussNoise(var_limit=(10, 50), p=0.3),       # Add noise
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),  # Page distortion
        A.OneOf([
            A.Blur(blur_limit=3, p=0.5),               # Blur
            A.MedianBlur(blur_limit=3, p=0.5),         # Median blur
        ], p=0.2),
    ])
    
    # Create directory for augmented images if it doesn't exist
    augmented_dir = os.path.join(output_dir, 'augmented')
    if not os.path.exists(augmented_dir):
        os.makedirs(augmented_dir)
    
    for class_dir in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
            
        # Create directory for this class in augmented
        aug_class_dir = os.path.join(augmented_dir, class_dir)
        if not os.path.exists(aug_class_dir):
            os.makedirs(aug_class_dir)
        
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            try:
                # Read image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Generate augmented samples
                for i in range(augmentation_factor):
                    augmented = transform(image=img)['image']
                    
                    # Create unique name for augmented image
                    base_name, ext = os.path.splitext(img_name)
                    aug_name = f"{base_name}_aug{i}{ext}"
                    aug_path = os.path.join(aug_class_dir, aug_name)
                    
                    # Save augmented image
                    cv2.imwrite(aug_path, augmented)
                    
                    # Also copy to the original train directory
                    train_aug_path = os.path.join(class_path, aug_name)
                    cv2.imwrite(train_aug_path, augmented)
                    
            except Exception as e:
                print(f"Error augmenting {img_path}: {e}")


def create_custom_augmentations_for_class4(output_dir):
    """Create specialized augmentations for class 4 subcategories."""
    # This function would implement the specialized augmentations for your reject class
    # such as simulating damaged manuscripts, non-manuscript images, etc.
    
    # For example, to create damaged manuscript simulations:
    train_dir = os.path.join(output_dir, 'train')
    class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d)) and d != '4']
    
    # Create class 4 directory if it doesn't exist
    class4_dir = os.path.join(train_dir, '4')
    if not os.path.exists(class4_dir):
        os.makedirs(class4_dir)
    
    # Create damage transforms
    damage_transform = A.Compose([
        A.CoarseDropout(max_holes=8, max_height=64, max_width=64, p=0.8),  # Simulate tears/holes
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.3, p=0.7),  # Faded ink
        A.GaussNoise(var_limit=(30, 80), p=0.6),  # Noise/degradation
    ])
    
    # Sample from other classes and create damaged versions
    for class_dir in class_dirs:
        source_dir = os.path.join(train_dir, class_dir)
        images = os.listdir(source_dir)
        
        # Take a subset of images to transform
        sample_size = min(10, len(images))
        sampled_images = random.sample(images, sample_size)
        
        for img_name in sampled_images:
            img_path = os.path.join(source_dir, img_name)
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                # Apply damage transform
                damaged = damage_transform(image=img)['image']
                
                # Save to class 4 directory
                damaged_name = f"damaged_{class_dir}_{img_name}"
                damaged_path = os.path.join(class4_dir, damaged_name)
                cv2.imwrite(damaged_path, damaged)
                
            except Exception as e:
                print(f"Error creating damaged version of {img_path}: {e}")


def extract_and_save_features(output_dir):
    """Extract features from all images and save to a features database."""
    features_db = {}
    
    for subset in ['train', 'validation', 'test']:
        subset_path = os.path.join(output_dir, subset)
        for class_dir in os.listdir(subset_path):
            class_path = os.path.join(subset_path, class_dir)
            if not os.path.isdir(class_path):
                continue
                
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                
                # Extract features
                staffline_confidence = detect_stafflines(img_path)
                layout_features = extract_layout_features(img_path)
                
                if layout_features:
                    # Store features
                    features_db[img_path] = {
                        'staffline_confidence': staffline_confidence,
                        'layout_features': layout_features,
                        'class': class_dir,
                        'subset': subset
                    }
    
    # Save features database
    feature_file = os.path.join(output_dir, 'features_db.pkl')
    import pickle
    with open(feature_file, 'wb') as f:
        pickle.dump(features_db, f)
    
    return features_db


def process_sequential_pages(dataset_dir, output_dir):
    """Process sequential pages to create linkage features."""
    # Implement if you have sequential manuscript pages
    # This would identify sequential pages and create linkage information
    pass


def main():
    """Main preprocessing pipeline."""
    dataset_dir = '/Users/kyriebouressa/Documents/script-sorter/dataset/raw_dataset'
    output_dir = '/Users/kyriebouressa/Documents/script-sorter/dataset'
    
    # Step 1: Split dataset
    print("Splitting dataset into train, validation, and test sets...")
    split_dataset(dataset_dir, output_dir)
    
    # Step 2: Normalize images
    print("Normalizing images for consistent size and contrast...")
    normalize_images(output_dir)
    
    # Step 3: Generate augmented data for training
    print("Generating augmented data for training...")
    generate_augmented_data(output_dir)
    
    # Step 4: Create specialized augmentations for class 4 (reject class)
    print("Creating specialized augmentations for class 4...")
    create_custom_augmentations_for_class4(output_dir)
    
    # Step 5: Extract and save features
    print("Extracting features from images...")
    features_db = extract_and_save_features(output_dir)
    
    # Step 6: Process sequential pages if available
    print("Processing sequential pages...")
    process_sequential_pages(dataset_dir, output_dir)
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()