import os
import shutil
import random
from sklearn.model_selection import train_test_split # type: ignore

def split_dataset(dataset_dir, output_dir, train_size=0.7, val_size=0.15):
    for class_dir in os.listdir(dataset_dir):
        class_path = os.path.join(dataset_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        train, temp = train_test_split(images, train_size=train_size, random_state=42)
        val, test = train_test_split(temp, test_size=(val_size / (1 - train_size)), random_state=42)
        
        for subset, subset_name in [(train, "train"), (val, "validation"), (test, "test")]:
            subset_dir = os.path.join(output_dir, subset_name, class_dir)
            os.makedirs(subset_dir, exist_ok=True)
            for img in subset:
                shutil.copy(img, subset_dir)

dataset_dir = '/Users/ekaterina/documents/script-sorter/dataset/raw_dataset'
output_dir = '/Users/ekaterina/documents/script-sorter/dataset/'
split_dataset(dataset_dir, output_dir)
