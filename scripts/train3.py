import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import KFold
import numpy as np
import os
import datetime
from PIL import Image
import shutil
from pathlib import Path
import psutil
import gc

#tensorboard version is 2.18.0

# Memory and GPU Configuration
tf.config.set_soft_device_placement(True)
tf.config.optimizer.set_jit(True)  # Enable XLA optimization

class MemoryMonitor(tf.keras.callbacks.Callback):
    def __init__(self, print_interval=1):
        super(MemoryMonitor, self).__init__()
        self.print_interval = print_interval
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.print_interval == 0:
            process = psutil.Process(os.getpid())
            ram_usage = process.memory_info().rss / 1024 / 1024  # MB
            print(f"\nRAM Memory Usage: {ram_usage:.2f} MB")
            if gpus:
                try:
                    if tf.__version__ >= "2.0":
                        gpu = gpus[0]  # Assuming first GPU
                        memory_info = tf.config.get_logical_device_configuration(gpu)
                        print(f"GPU Memory Info: {memory_info}")
                    else:
                        gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                        print(f"GPU Memory Usage - Peak: {gpu_memory['peak'] / 1024**2:.2f} MB, "
                              f"Current: {gpu_memory['current'] / 1024**2:.2f} MB")
                except Exception as e:
                    print(f"Unable to get GPU memory info: {str(e)}")

# Memory management for GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Memory and GPU Configuration
def get_available_memory():
    """Get available GPU memory in bytes"""
    if gpus:
        try:
            gpu = tf.config.experimental.get_gpu_device_properties('GPU:0')
            return gpu.memory_limit
        except:
            return 4e9  # Default to 4GB if can't get GPU memory
    return psutil.virtual_memory().available

def optimize_batch_size(available_memory, image_size, precision=32):
    """Calculate optimal batch size based on available memory"""
    image_bytes = np.prod(image_size) * 3 * (precision / 8)
    model_memory = 500e6  # Approximate model size in bytes
    available_memory = available_memory * 0.8  # Use 80% of available memory
    optimal_batch = int((available_memory - model_memory) / (image_bytes * 2))
    return max(min(optimal_batch, 128), 8)  # Clamp between 8 and 128

# Constants and Configuration
IMAGE_SIZE = (224, 224)
AUTOTUNE = tf.data.AUTOTUNE
NUM_FOLDS = 5
MIN_SAMPLES_PER_CLASS = 50  # Minimum samples required per class

# Calculate optimal batch size
BATCH_SIZE = optimize_batch_size(get_available_memory(), IMAGE_SIZE)

# Directory setup
dataset_dir = 'dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')
expected_classes = ['0', '1', '2', '3', '4']

# Create necessary directories
os.makedirs('models', exist_ok=True)
os.makedirs('fold_data', exist_ok=True)

def verify_data_directories():
    """Verify directory structure and count images in each class."""
    def verify_dataset_size(directory, min_samples=MIN_SAMPLES_PER_CLASS):
        """Verify dataset has enough samples per class and check image integrity"""
        print(f"\nVerifying minimum samples in {os.path.basename(directory)}...")
        for class_name in expected_classes:
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
    class_counts = {}
    for directory in [train_dir, val_dir, test_dir]:
        dir_name = os.path.basename(directory)
        print(f"\nChecking {dir_name} directory:")
        class_dirs = sorted(os.listdir(directory))
        print(f"Found classes: {class_dirs}")
        
        if not all(cls in class_dirs for cls in expected_classes):
            raise ValueError(f"Missing some class directories in {dir_name}!")
        
        for class_name in class_dirs:
            class_path = os.path.join(directory, class_name)
            if os.path.isdir(class_path):
                n_images = len([f for f in os.listdir(class_path) 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = n_images
                print(f"Class {class_name}: {n_images} images")
    return class_counts

def create_dataset(directory, is_training=False):
    """Create a tf.data.Dataset from directory."""
    def preprocess_image(path):
        try:
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, IMAGE_SIZE)
            image = tf.cast(image, tf.float32) / 255.0
            return image
        except Exception as e:
            print(f"Error processing image {path}: {str(e)}")
            # Return a placeholder image or raise the exception based on your needs
            raise RuntimeError(f"Failed to process image {path}: {str(e)}")

    def augment(image):
        """Apply augmentation to images."""
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.clip_by_value(image, 0, 1)
        return image

    # Get all image paths and labels
    image_paths = []
    labels = []
    for class_idx, class_name in enumerate(expected_classes):
        class_path = os.path.join(directory, class_name)
        class_images = [os.path.join(class_path, fname) 
                       for fname in os.listdir(class_path)
                       if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(lambda x, y: (preprocess_image(x), y), 
                         num_parallel_calls=AUTOTUNE)

    if is_training:
        dataset = dataset.map(lambda x, y: (augment(x), y), 
                            num_parallel_calls=AUTOTUNE)
        dataset = dataset.shuffle(buffer_size=1000)

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset, len(image_paths)

def create_model(architecture='resnet50'):
    """Create model with specified base architecture."""
    if architecture == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=(*IMAGE_SIZE, 3))
    elif architecture == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, 
                            input_shape=(*IMAGE_SIZE, 3))
    else:
        raise ValueError("Unsupported architecture")
    
    base_model.trainable = False
    regularizer = tf.keras.regularizers.l2(0.01)
    
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(0.6),
        layers.BatchNormalization(),
        layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
        layers.Dropout(0.5),
        layers.Dense(len(expected_classes), activation='softmax', 
                    kernel_regularizer=regularizer)
    ])
    return model

def prepare_fold_data(train_paths, train_labels, val_paths, val_labels, fold):
    """Create temporary directories for current fold."""
    fold_dir = f'fold_data/fold_{fold}'
    if os.path.exists(fold_dir):
        shutil.rmtree(fold_dir)
    
    for split in ['train', 'val']:
        for class_name in expected_classes:
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

def train_with_kfold():
    """Train model using K-fold cross validation."""
    # Memory cleanup before training
    gc.collect()
    if gpus:
        tf.keras.backend.clear_session()
    
    # Monitor initial memory state
    process = psutil.Process(os.getpid())
    print(f"\nInitial RAM usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
    if gpus:
        print("Initial GPU memory state:", 
              tf.config.experimental.get_memory_info('GPU:0'))

    # Get all image paths and labels
    image_paths = []
    labels = []
    for class_idx, class_name in enumerate(expected_classes):
        class_path = os.path.join(train_dir, class_name)
        class_images = [os.path.join(class_path, fname) 
                       for fname in os.listdir(class_path)
                       if fname.lower().endswith(('.png', '.jpg', '.jpeg'))]
        image_paths.extend(class_images)
        labels.extend([class_idx] * len(class_images))

    # Convert to numpy arrays
    image_paths = np.array(image_paths)
    labels = np.array(labels)

    # Initialize K-fold
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_histories = []

    # K-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(image_paths)):
        print(f"\nTraining fold {fold + 1}/{NUM_FOLDS}")
        
        # Prepare data for this fold
        train_paths, val_paths = image_paths[train_idx], image_paths[val_idx]
        train_labels, val_labels = labels[train_idx], labels[val_idx]
        
        # Create temporary directories and datasets for this fold
        fold_train_dir, fold_val_dir = prepare_fold_data(
            train_paths, train_labels, val_paths, val_labels, fold)
        
        train_dataset, train_size = create_dataset(fold_train_dir, is_training=True)
        val_dataset, val_size = create_dataset(fold_val_dir, is_training=False)
        
        # Create and compile model
        model = create_model('vgg16')
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Set up callbacks
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                verbose=1,
                restore_best_weights=True,
                mode='min'
            ),
            ModelCheckpoint(
                f'models/fold_{fold + 1}_{timestamp}.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1,
                mode='min'
            ),
            TensorBoard(
                log_dir=f"logs/fit/fold_{fold + 1}_{timestamp}",
                histogram_freq=1
            ),
            MemoryMonitor(print_interval=1)
        ]
        
        # Train the model
        history = model.fit(
            train_dataset,
            epochs=20,
            validation_data=val_dataset,
            callbacks=callbacks
        )
        
        fold_histories.append(history.history)
        
        # Clean up fold data
        shutil.rmtree(f'fold_data/fold_{fold}')
    
    return fold_histories

if __name__ == "__main__":
    print(f"\nOptimized batch size: {BATCH_SIZE}")
    print("Verifying dataset requirements...")
    
    # Verify directory structure, minimum samples, and get class counts
    class_counts = verify_data_directories()
    
    # Verify minimum samples in each split
    for directory in [train_dir, val_dir, test_dir]:
        verify_dataset_size(directory)
    
    # Print memory usage monitoring message
    print("\nMonitoring memory usage...")
    if gpus:
        print("GPU memory usage:", tf.config.experimental.get_memory_info('GPU:0'))
    
    # Train model with K-fold cross validation
    histories = train_with_kfold()
    
    # Calculate and print final metrics
    final_val_accuracies = [h['val_accuracy'][-1] for h in histories]
    mean_accuracy = np.mean(final_val_accuracies)
    std_accuracy = np.std(final_val_accuracies)
    print(f"\nFinal Cross-Validation Accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")