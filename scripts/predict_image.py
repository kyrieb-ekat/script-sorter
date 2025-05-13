from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import tensorflow as tf

model_path = '/Users/kyriebouressa/documents/script-sorter/scripts/models/fold_3_20250131_180038.keras' #DDMAL machine ; old model PATH/label = vgg16_fold5_v20250121_213938.keras
# '/Users/ekaterina/documents/script-sorter/models/vgg16_fold5_v20250121_213938.keras'  # /path/to/models/vgg16_finetuned.h5 #laptop path
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

image_path = '/Users/kyriebouressa/documents/script-sorter/dataset/stranger_danger/Ott.lat.145_0010_m copy.jpg'
# in lab, use this PATH: /Users/kyriebouressa/documents/script-sorter/dataset/stranger_danger/[IMAGE]
# /Users/ekaterina/documents/script-sorter/dataset/z_newimage/Madrid Codex 30- 20v.jpg NOTE: this path is default to angantyr path; fix if in lab

# correctly assigned Einsie611 to class 3 (confidence .38)
# incorrectly assigned Ross chunk to class 3- should've been class 2. confidence .63
# correctly assigned SG484 to class 0. confidence .50.
# incorrectly assigned Madrid Codex 30 to class 0, confidence of .39. Tricky one, however!
#incorrectly classified SG_484_15r as class 3, confidence 23%. Should've been class 0.
#correctly classified SktG390_116 as class 0. 86% confidence  
# fold 4 correctly classified einside007v; fold 5 could not. 
#fold 4 misclassified Madrid30_20v as class 2 (Beneventan) not class 1 (Hispanic) with 58% confidence


class_index, confidence = classify_image(image_path)
print(f"Predicted class: {class_index}, Confidence: {confidence:.2f}")
