import tensorflow as tf

print("TensorFlow version:", tf.__version__)

# Test individual imports
try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras import layers, models
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
    print("All TensorFlow submodules imported successfully.")
except ImportError as e:
    print("ImportError:", e)
