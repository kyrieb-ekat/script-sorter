from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

model_path = '/Users/ekaterina/documents/script-sorter/models/vgg16_fold5_v20250121_213938.keras'  # /path/to/models/vgg16_finetuned.h5
IMG_SIZE = (224, 224)
model = tf.keras.models.load_model(model_path)

def classify_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index]
    return class_index, confidence

image_path = '/Users/ekaterina/documents/script-sorter/dataset/z_newimage/Madrid Codex 30- 20v.jpg'
# correctly assigned Einsie611 to class 3 (confidence .38)
# incorrectly assigned Ross chunk to class 3- should've been class 2. confidence .63
# correctly assigned SG484 to class 0. confidence .50.
# incorrectly assigned Madrid Codex 30 to class 0, confidence of .39. Tricky one, however!
class_index, confidence = classify_image(image_path)
print(f"Predicted class: {class_index}, Confidence: {confidence:.2f}")
