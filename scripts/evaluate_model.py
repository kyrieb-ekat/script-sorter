import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and test data
model_path = '/Users/ekaterina/documents/script-sorter/models/vgg16_fold5_v20250121_213938.keras' # in future change this to .keras
test_dir = '/Users/ekaterina/documents/script-sorter/dataset/test'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

model = tf.keras.models.load_model(model_path)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_data = test_datagen.flow_from_directory(test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

# Predictions
predictions = model.predict(test_data)
y_pred = np.argmax(predictions, axis=1)
y_true = test_data.classes

# Metrics
print(classification_report(y_true, y_pred, target_names=test_data.class_indices.keys()))
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_data.class_indices.keys(), yticklabels=test_data.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('/Users/ekaterina/documents/script-sorter/results/confusion_matrix.png')
