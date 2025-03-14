import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import warnings

warnings.simplefilter("ignore", Image.DecompressionBombWarning)

# Paths
model_path = '/Users/kyriebouressa/Documents/script-sorter/scripts/models/fold_4_20250131_190118.keras'
test_dir = '/Users/kyriebouressa/Documents/script-sorter/dataset/test'
results_dir = Path('/Users/kyriebouressa/Documents/script-sorter/results')
results_dir.mkdir(parents=True, exist_ok=True)

# Debug: Check image files directly
for root, dirs, files in os.walk(test_dir):
    for file in files:
        # Check if file is an image (you could further refine this)
        if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
            print(f"Skipping non-image file: {file}")
            continue
        full_path = os.path.join(root, file)
        try:
            img = Image.open(full_path)
            img.verify()
        except Exception as e:
            print(f"Problem with file: {full_path} ({e})")


# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load model
model = tf.keras.models.load_model(model_path)

# Load test data
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_data = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Predictions
y_prob = model.predict(test_data)  # Probability outputs
y_pred = np.argmax(y_prob, axis=1)  # Predicted labels
y_true = test_data.classes  # True labels
class_names = list(test_data.class_indices.keys())

# Metrics: Classification Report
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
report_text = classification_report(y_true, y_pred, target_names=class_names)
Path(results_dir / 'classification_report.txt').write_text(report_text)
print(report_text)

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot and save confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(results_dir / 'confusion_matrix.png')
plt.show()

# Compute per-class accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)
accuracy_text = "\n".join([f"{class_names[i]}: {class_accuracies[i]:.4f}" for i in range(len(class_names))])
Path(results_dir / 'per_class_accuracy.txt').write_text(accuracy_text)
print("Per-Class Accuracy:")
print(accuracy_text)

# ROC Curves (One-vs-Rest for Multi-Class)
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc='lower right')
plt.savefig(results_dir / 'roc_curves.png')
plt.show()
