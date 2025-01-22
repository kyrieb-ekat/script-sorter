import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import KFold
import numpy as np
import os
import math

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
  validation/
    0/
    1/
    2/
    3/
  test/
    0/
    1/
    2/
    3/
""")

# Check for class directories in all splits
expected_classes = ['0', '1', '2', '3']
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

# Set up data generator
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # We'll use this for the k-fold validation
)

batch_size = 32

# Create the generators
train_data = datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

val_data = datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

print("\nClass mapping:", train_data.class_indices)

# Get all filenames and their corresponding labels
filenames = np.array(train_data.filenames)
labels = np.array(train_data.labels)

# Set up KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

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
    
    # Load and set up the model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),  # Added dropout for regularization
        layers.Dense(4, activation='softmax')  # 4 classes (0,1,2,3)
    ])
    
    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Set up callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=1,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            f'models/vgg16_fold{fold+1}.keras',
            save_best_only=True,
            verbose=1
        )
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
    base_model.trainable = True
    
    # Freeze early layers
    for layer in base_model.layers[:15]:
        layer.trainable = False
    
    # Recompile with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune the model
    fine_tune_history = model.fit(
        train_data,
        epochs=5,  # Reduced epochs for fine-tuning
        validation_data=val_data,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=1,
        callbacks=callbacks
    )
    
    # Evaluate the model
    print("\nEvaluating model...")
    val_loss, val_accuracy = model.evaluate(val_data)
    print(f"Validation Accuracy for Fold {fold+1}: {val_accuracy:.4f}")
    fold_accuracies.append(val_accuracy)

# Calculate and print final metrics
average_accuracy = np.mean(fold_accuracies)
std_accuracy = np.std(fold_accuracies)

print(f"\nTraining completed!")
print(f"Average Accuracy over 5 folds: {average_accuracy:.4f}")
print(f"Standard Deviation: {std_accuracy:.4f}")