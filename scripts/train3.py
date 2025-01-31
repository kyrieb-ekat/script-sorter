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

# Memory and GPU Configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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
def get_available_memory():
    """Get available GPU memory in bytes"""
    if gpus:
        try:
            gpu = tf.config.experimental.get_gpu_device_properties('GPU:0')
            return gpu.memory_limit
        except Exception as e:
            print(f"Error retrieving GPU memory properties: {e}")
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

def verify_dataset_size(directory):
    """Verify dataset has enough samples per class and check image integrity"""
    def log_and_continue(img_path, error_message):
        """Log error and continue with the next image"""
        print(f"Error processing {img_path}: {error_message}")
        # Optionally move problematic images to a separate folder
        error_folder = 'error_images'
        os.makedirs(error_folder, exist_ok=True)
        shutil.move(img_path, os.path.join(error_folder, os.path.basename(img_path)))
    
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
                    log_and_continue(img_path, str(e))
                    continue
            
        n_samples = len(valid_images)
        if n_samples < MIN_SAMPLES_PER_CLASS:
            raise ValueError(
                f"Class {class_name} in {os.path.basename(directory)} "
                f"has insufficient samples: {n_samples}. "
                f"Minimum required: {MIN_SAMPLES_PER_CLASS}"
            )

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

def get_disk_space():
    """Check available disk space"""
    total, used, free = shutil.disk_usage("/")
    return free / (1024 * 1024 * 1024)  # In GB

def create_model(architecture='vgg16'):
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
    
    disk_space = get_disk_space()
    print(f"Available disk space: {disk_space:.2f} GB")

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
    
    # Perform K-fold cross-validation
    kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    fold_accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(image_paths, labels)):
        print(f"\nTraining Fold {fold+1}/{NUM_FOLDS}")
        
        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        train_labels = [labels[i] for i in train_idx]
        val_labels = [labels[i] for i in val_idx]
        
        # Prepare fold data
        train_fold_dir, val_fold_dir = prepare_fold_data(
            train_paths, train_labels, val_paths, val_labels, fold)
        
        # Load datasets for the current fold
        train_dataset, _ = create_dataset(train_fold_dir, is_training=True)
        val_dataset, _ = create_dataset(val_fold_dir, is_training=False)

        # Model creation
        model = create_model()
        
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        # Callbacks for monitoring
        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5)
        model_checkpoint = ModelCheckpoint(f'models/fold_{fold+1}.h5', 
                                           save_best_only=True)
        tensorboard = TensorBoard(log_dir=f'logs/fold_{fold+1}')
        
        model.fit(train_dataset, validation_data=val_dataset, epochs=50, 
                  callbacks=[early_stopping, model_checkpoint, tensorboard])

        # Store fold accuracy
        val_accuracy = model.evaluate(val_dataset, verbose=0)[1]
        fold_accuracies.append(val_accuracy)
        print(f"Fold {fold+1} Validation Accuracy: {val_accuracy:.4f}")

    print(f"\nMean Validation Accuracy: {np.mean(fold_accuracies):.4f}")
    print(f"Standard Deviation: {np.std(fold_accuracies):.4f}")

if __name__ == '__main__':
    verify_data_directories()
    train_with_kfold()
