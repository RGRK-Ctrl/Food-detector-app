import os
import numpy as np
import cv2
import json

# Paths
MASTER_DATASET_PATH = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food Classification dataset'
FEEDBACK_PATH = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Feedback dataset'
REPORTS_DIR = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model data/Reports'
CHARTS_DIR = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model data/Charts'
MODEL_PATH = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model data/Food detector model.keras"
TRAINING_LOG = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model data/logs/training_log.csv'
LABELS_PATH = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model data/logs/class_labels.json"
CHECKPOINTS_DIR = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model data/Checkpoints"
TEST_SPLIT_PATH = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model data/test_split.npz'
TEST_DIR = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food images test set'

os.makedirs(TEST_DIR, exist_ok=True)



import os
import shutil

if os.path.exists(TEST_DIR):
    for item in os.listdir(TEST_DIR):
        item_path = os.path.join(TEST_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            os.remove(item_path)
    print(f"🧹 Cleared contents of: {TEST_DIR}")



# Load labels (for mapping integer class indices to class names if needed)
with open(LABELS_PATH, "r") as f:
    class_info = json.load(f)
class_labels = class_info["labels"]

# Load test split
print(f"✅ Loading test split images from Food Images Test set directory")

data = np.load(TEST_SPLIT_PATH)
# See available keys
print(data.files)  # ['test_paths', 'test_labels']

# Access using the correct keys
test_paths = data['test_paths']  # list or array of file paths to test images
test_labels = data['test_labels']  # corresponding labels

# You cannot access 'X_test' directly since image data is not stored in the npz
# Instead, you should load images from disk using these paths when you need to.

print(f"Number of test samples: {len(test_paths)}")
print(f"Example test image path: {test_paths[0]}")




# Parameters: image size expected by your model
IMG_SIZE = (224, 224)

# Load images from disk, preprocess
X_test = []
for img_path in test_paths:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Warning: Could not read image {img_path}")
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMG_SIZE)
    # Normalize pixel values between 0 and 1 if your model expects that
    img = img.astype('float32') / 255.0
    X_test.append(img)

X_test = np.array(X_test)
y_test = np.array(test_labels[:len(X_test)])  # ensure labels length matches loaded images

print("Test image data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)



# Convert to integer labels if needed
if len(y_test.shape) > 1 and y_test.shape[1] > 1:
    labels = np.argmax(y_test, axis=1)
else:
    labels = y_test.astype(int)

# --- Save Images ---
for idx, (img, label) in enumerate(zip(X_test, labels)):
    class_folder = os.path.join(TEST_DIR, class_labels[label])  # Use class name, not number
    os.makedirs(class_folder, exist_ok=True)
    save_path = os.path.join(class_folder, f"img_{idx:06d}.jpg")
    img_uint8 = np.clip(img * 255.0, 0, 255).astype('uint8')
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, img_bgr)

print(f"✅ All test images saved under: {TEST_DIR}")



"""
# Step 2: Count images per class & save to file
# Path to save counts file
counts_file_path = os.path.join(TEST_DIR, "test_classes_count.txt")

if os.path.exists(TEST_DIR):
    with open(counts_file_path, "w") as f:
        total_images = 0
        for class_folder in sorted(os.listdir(TEST_DIR), key=str.lower):
            folder_path = os.path.join(TEST_DIR, class_folder)
            if os.path.isdir(folder_path):
                image_count = len([
                    file for file in os.listdir(folder_path)
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                ])
                total_images += image_count
                f.write(f"{class_folder}: {image_count} images\n")
                print(f"{class_folder}: {image_count} images")
        f.write(f"\nTotal images: {total_images}\n")
        print(f"\nTotal images: {total_images}")

print(f"✅ Saved test classes count to {counts_file_path}")
"""

import os
from datetime import datetime
import sys
import tensorflow as tf

# Step 1: Prepare save path
counts_file_path = os.path.join(TEST_DIR, "test_classes_count.txt")

# Step 2: Collect metadata
generated_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if os.path.exists(TEST_SPLIT_PATH):
    npz_timestamp = datetime.fromtimestamp(os.path.getmtime(TEST_SPLIT_PATH)).strftime("%Y-%m-%d %H:%M:%S")
else:
    npz_timestamp = "NPZ file not found"

python_version = sys.version.split()[0]
tensorflow_version = tf.__version__

# Step 3: Write to file
if os.path.exists(TEST_DIR):
    with open(counts_file_path, "w") as f:
        # Metadata section
        f.write(f"===== Test Set Report =====\n")
        f.write(f"Generated on: {generated_timestamp}\n")
        f.write(f"NPZ file created on: {npz_timestamp}\n")
        f.write(f"Python Version: {python_version}\n")
        f.write(f"TensorFlow Version: {tensorflow_version}\n\n")

        # Counts section
        total_images = 0
        f.write("===== Image Counts per Class =====\n")
        for class_folder in sorted(os.listdir(TEST_DIR), key=str.lower):
            folder_path = os.path.join(TEST_DIR, class_folder)
            if os.path.isdir(folder_path):
                image_count = len([
                    file for file in os.listdir(folder_path)
                    if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))
                ])
                total_images += image_count
                f.write(f"{class_folder}: {image_count} images\n")
                print(f"{class_folder}: {image_count} images")
        f.write(f"\nTotal images: {total_images}\n")
        print(f"\nTotal images: {total_images}")

print(f"✅ Saved test classes count with metadata to {counts_file_path}")


#----------------------------------
# Sending telegram message---------
#----------------------------------
import requests
from send_telegram import send_message
send_message("✅ Test set data is saved in the TEST_DIR")
