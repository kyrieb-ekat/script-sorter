#significant difference here have a lot to do with tweaks to simplify early stopping,
# add test set evaluation; trains with k-fold cross validation, uses simpler early stopping
# saves the best model based on validation loss, and evaluates on the test set after training
# it will then provide detailed per-class accuracy metrics for the test set. 
# things I'm thinking about: TensorBoard apparently is a highly interesting
# and very cool tool for visualizing model training and evaluation metrics.
# it can be implemented through a TensorBoard callback in the model.fit() method.
# from tensorflow.keras.callbacks import TensorBoard
# tensorboard callback:
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
# which then gets passed back to model.fit() as a callback. : model.fit(..., callbacks=[callbacks, tensorboard_callback])
# this will create a log directory with timestamped logs that can be visualized with TensorBoard.
# which can be launched from the command line through the command: tensorboard --logdir logs/fit
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from sklearn.model_selection import KFold
from sklearn.manifold import TSNE # type: ignore
import numpy as np
import os
import math
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Disable the decompression bomb warning
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# First, verify the directory structure
print("\nVerifying directory structure...")
dataset_dir = 'dataset'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'test')

print(f"""
Expected directory structure:
dataset/
  train/
    0/
    1/
    2/
    3/
    4/  # Added rejection class
  validation/
    0/
    1/
    2/
    3/
    4/  # Added rejection class
  test/
    0/
    1/
    2/
    3/
    4/  # Added rejection class
""")

# Updated to include rejection class
expected_classes = ['0', '1', '2', '3', '4']

# Check for class directories in all splits
for directory in [train_dir, val_dir, test_dir]:
    dir_name = os.path.basename(directory)
    print(f"\nChecking {dir_name} directory:")
    class_dirs = sorted(os.listdir(directory))
    print(f"Found classes: {class_dirs}")
    
    if not all(cls in class_dirs for cls in expected_classes):
        raise ValueError(f"Missing some class directories in {dir_name}!")
    
    # Print number of images in each class
    for class_name in class_dirs:
        class_path = os.path.join(directory, class_name)
        if os.path.isdir(class_path):
            n_images = len([f for f in os.listdir(class_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"Class {class_name}: {n_images} images")

# Set up data generator with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.2,
    fill_mode='nearest'
)

# Validation generator should only rescale, no augmentation
val_datagen = ImageDataGenerator(rescale=1./255)

batch_size = 32

# Create the generators with updated class configuration
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=expected_classes,
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=expected_classes,
    shuffle=False
)

test_data = val_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    classes=expected_classes,
    shuffle=False
)

print("\nTest data distribution:")
for class_name, class_index in test_data.class_indices.items():
    n_samples = len([f for f in test_data.filenames if f.startswith(f"{class_name}/")])
    print(f"Class {class_name}: {n_samples} images")

# Function to create model with choice of architecture
def create_model(architecture='vgg16'):
    """
    Create model with specified base architecture.
    Args:
        architecture: 'vgg16' or 'resnet50'
    Returns:
        Compiled Keras model
    """
    if architecture == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False, 
                          input_shape=(224, 224, 3))
    elif architecture == 'resnet50':
        base_model = ResNet50(weights='imagenet', include_top=False, 
                            input_shape=(224, 224, 3))
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
        layers.Dense(5, activation='softmax', kernel_regularizer=regularizer)  # 5 classes
    ])
    return model

# Function to visualize feature clusters
def visualize_clusters(model, data_generator, layer_name='dense_1'):
    """
    Extract features from a specific layer and visualize clusters using t-SNE
    """
    intermediate_model = tf.keras.Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )
    
    features = []
    labels = []
    batch_count = 0
    max_batches = len(data_generator)
    
    for x, y in data_generator:
        if batch_count >= max_batches:
            break
        batch_features = intermediate_model.predict(x)
        features.append(batch_features)
        labels.append(np.argmax(y, axis=1))
        batch_count += 1
    
    features = np.vstack(features)
    labels = np.concatenate(labels)
    
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels, cmap='tab10')
    plt.colorbar(scatter)
    plt.title(f't-SNE visualization of {layer_name} features')
    plt.savefig(f'cluster_viz_{layer_name}.png')
    plt.close()

# Get all filenames and their corresponding labels
filenames = np.array(train_data.filenames)
labels = np.array(train_data.labels)

# Set up KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

# Set up TensorBoard callback
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1,
    write_graph=True,
    write_images=True
)

# KFold Cross-validation loop
for fold, (train_index, val_index) in enumerate(kf.split(filenames)):
    print(f"\nTraining fold {fold+1}/5...")
    
    # Split data for this fold
    train_files = filenames[train_index]
    train_labels = labels[train_index]
    val_files = filenames[val_index]
    val_labels = labels[val_index]
    
    # Calculate steps
    steps_per_epoch = math.ceil(len(train_files) / batch_size)
    validation_steps = math.ceil(len(val_files) / batch_size)
    
    # Create model for this fold
    model = create_model('vgg16')  # or 'resnet50'
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Get timestamp for this training run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_version = f"v{timestamp}"
    
    # Set up callbacks with versioned saving
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True,
            mode='min'
        ),
        ModelCheckpoint(
            f'models/vgg16_fold{fold+1}_{model_version}.keras',
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
            mode='min'
        ),
        tensorboard_callback
    ]
    
    # Initial training
    print("\nStarting initial training phase...")
    history = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callbacks
    )
    
    # Fine-tuning phase
    print("\nStarting fine-tuning phase...")
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:15]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune the model
    fine_tune_history = model.fit(
        train_data,
        epochs=5,
        validation_data=val_data,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callbacks
    )
    
    # Generate cluster visualization
    visualize_clusters(model, test_data)
    
    # Evaluate the model
    print("\nEvaluating model...")
    val_loss, val_accuracy = model.evaluate(val_data)
    print(f"Validation Accuracy for Fold {fold+1}: {val_accuracy:.4f}")
    fold_accuracies.append(val_accuracy)

# Calculate and print final metrics
average_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"\nTraining completed!")
print(f"Average Validation Accuracy over 5 folds: {average_accuracy:.4f}")
print(f"Standard Deviation: {std_accuracy:.4f}")

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(test_data, verbose=1)
print(f"\nFinal Test Set Accuracy: {test_accuracy:.4f}")
print(f"Final Test Set Loss: {test_loss:.4f}")

# Generate predictions and detailed metrics for test set
test_predictions = model.predict(test_data)
test_predicted_classes = np.argmax(test_predictions, axis=1)
true_classes = test_data.classes

# Print per-class accuracy including rejection class
for class_name, class_index in test_data.class_indices.items():
    class_mask = (true_classes == class_index)
    class_accuracy = np.mean(test_predicted_classes[class_mask] == true_classes[class_mask])
    
    if class_name == '4':  # Rejection class
        print(f"Rejection Class Accuracy: {class_accuracy:.4f}")
        # Calculate false rejection rate
        false_rejection = np.mean(
            test_predicted_classes[true_classes != class_index] == class_index
        )
        print(f"False Rejection Rate: {false_rejection:.4f}")
    else:
        print(f"Class {class_name} Accuracy: {class_accuracy:.4f}")