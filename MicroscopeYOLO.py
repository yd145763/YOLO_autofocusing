import os
import random
import shutil
from pathlib import Path


import yaml



# Path to your YAML file
yaml_file_path = "/home/grouptan/Documents/yudian/yolo_microscope/datasetMicroscope.yaml"  # Update with the path to your YAML file

# Load the existing YAML file
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file)

# Extract the train and val paths
train_path = data.get('train', 'Not found')
val_path = data.get('val', 'Not found')

# Print the train and val paths as strings
print(f"Train path: {train_path}")
print(f"Val path: {val_path}")

images_path = Path(train_path).parent
labels_path = Path(images_path).parent / "labels"


images_train_path = images_path / "train"
images_val_path = images_path / "val"
labels_train_path = labels_path / "train"
labels_val_path = labels_path / "val"

# Create directories if they do not exist
images_train_path.mkdir(parents=True, exist_ok=True)
images_val_path.mkdir(parents=True, exist_ok=True)
labels_train_path.mkdir(parents=True, exist_ok=True)
labels_val_path.mkdir(parents=True, exist_ok=True)

# Function to delete all files in a directory
def delete_files_in_directory(directory_path):
    for file_path in directory_path.glob('*'):
        try:
            if file_path.is_file() or file_path.is_symlink():
                os.unlink(file_path)
            elif file_path.is_dir():
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

# Delete files in the specified directories
delete_files_in_directory(images_train_path)
delete_files_in_directory(images_val_path)
delete_files_in_directory(labels_train_path)
delete_files_in_directory(labels_val_path)

print("All files in the specified directories have been deleted.")

# Get all image files and label files
image_files = list(images_path.glob("*.jpg")) + list(images_path.glob("*.jpeg")) + list(images_path.glob("*.png"))
label_files = list(labels_path.glob("*.txt"))

# Create a dictionary to match images with their corresponding labels
image_label_pairs = {}
for image_file in image_files:
    label_file = labels_path / (image_file.stem + ".txt")
    if label_file.exists():
        image_label_pairs[image_file] = label_file

# Shuffle and split into training and validation sets
image_label_items = list(image_label_pairs.items())
random.shuffle(image_label_items)
split_index = int(len(image_label_items) * 0.5)
train_set = image_label_items[:split_index]
val_set = image_label_items[split_index:]

# Function to copy files
def copy_files(file_pairs, dest_image_path, dest_label_path):
    for image_file, label_file in file_pairs:
        shutil.copy(image_file, dest_image_path / image_file.name)
        shutil.copy(label_file, dest_label_path / label_file.name)

# Copy the files to their respective directories
copy_files(train_set, images_train_path, labels_train_path)
copy_files(val_set, images_val_path, labels_val_path)

print("Files have been copied successfully.")


from ultralytics import YOLO

import tensorflow as tf
print(tf.__version__)





# Initialize the YOLO model
model = YOLO()

# Train the model with the specified dataset and number of epochs
model.train(data=yaml_file_path, epochs=200,
    flipud=0.5,          # Vertical flip probability
    fliplr=0.5,          # Horizontal flip probability
    scale=0.5,           # Scale
    translate=0.1,       # Translate
    shear=0.1,           # Shear
    perspective=0.0,     # Perspective
    hsv_h=0.015,         # HSV hue augmentation (fraction)
    hsv_s=0.7,           # HSV saturation augmentation (fraction)
    hsv_v=0.4 )



