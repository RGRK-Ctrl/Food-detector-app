import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from datetime import datetime
from send_telegram import send_message

send_message("Test-Val-Split: Running Train-test-val split script ✅")
# -----------------------------
# Config
# -----------------------------
MASTER_DATASET_PATH = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food Classification dataset"
TRAIN_DIR = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food images Training set"
VAL_DIR   = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food images Validation set"
TEST_DIR  = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food images Test set"
TRAIN_VAL_TEST_SPLIT_FILE = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/train_val_test_split.npz"

# -----------------------------
# Helper function: clear & recreate split dirs
# -----------------------------
def prepare_dir(base_dir, class_labels):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    os.makedirs(base_dir, exist_ok=True)
    for cls in class_labels:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)

def save_summary(base_dir, split_paths, split_labels, class_labels, split_name):
    summary_file = os.path.join(base_dir, f"{split_name}_summary.txt")
    counts = {cls: 0 for cls in class_labels}
    for lbl in split_labels:
        counts[class_labels[lbl]] += 1

    total_images = len(split_paths)
    with open(summary_file, "w") as f:
        f.write(f"Summary for {split_name} split\n")
        f.write(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        for cls, count in counts.items():
            f.write(f"{cls}: {count} images\n")
        f.write(f"\nTotal images: {total_images}\n")

    print(f"✅ Saved {split_name} summary")
    send_message(f"✅ Saved {split_name} summary")

# -----------------------------
# Load dataset (image paths + labels)
# -----------------------------

image_paths, image_labels = [], []
class_labels = sorted(
    [d for d in os.listdir(MASTER_DATASET_PATH) if os.path.isdir(os.path.join(MASTER_DATASET_PATH, d))],
    key=str.lower
)
print(f"Test-Val-Split: Master dataset loaded ✅ ")


for label_idx, cls in enumerate(class_labels):
    folder = os.path.join(MASTER_DATASET_PATH, cls)
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            image_paths.append(os.path.join(folder, fname))
            image_labels.append(label_idx)

image_paths, image_labels = np.array(image_paths), np.array(image_labels)

# -----------------------------
# Train/Val/Test Split (64/16/20)
# -----------------------------
print(f"Test-Val-Split: Started proceeding with test split 80:15:5 as intended ✅ ")

train_paths, temp_paths, train_labels, temp_labels = train_test_split(
    image_paths, image_labels, train_size=0.80, test_size=0.20, stratify=image_labels, random_state=42
)

val_paths, test_paths, val_labels, test_labels = train_test_split(
    temp_paths, temp_labels, train_size=0.75, test_size=0.25, stratify=temp_labels, random_state=42
)

print(f"✅ Training set={len(train_paths)} images")
send_message(f" Training set={len(train_paths)} images")

print(f"✅ Validation set={len(val_paths)} images")
send_message(f" Validation set={len(val_paths)} images")

print(f"✅ Test set={len(test_paths)} images")
send_message(f" Test set={len(test_paths)} images")

# -----------------------------
# Save split info
# -----------------------------
np.savez(TRAIN_VAL_TEST_SPLIT_FILE,
         train_paths=train_paths, train_labels=train_labels,
         val_paths=val_paths, val_labels=val_labels,
         test_paths=test_paths, test_labels=test_labels,
         class_labels=class_labels)

print(f"Test-Val-Split: Saved train/val/test split paths & Labels to NPZ file ✅")

import shutil

# Clean split directories before copying
for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
    if os.path.exists(split_dir):
        for item in os.listdir(split_dir):
            item_path = os.path.join(split_dir, item)
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)  # remove old class subfolders
            elif os.path.isfile(item_path):
                os.remove(item_path)  # remove stray files

# -----------------------------
# Copy images into respective dirs
# -----------------------------
print("Test-Val-Split: Started copying files to directories")

for split_name, split_paths, split_labels, target_dir in [
    ("train", train_paths, train_labels, TRAIN_DIR),
    ("val",   val_paths,   val_labels,   VAL_DIR),
    ("test",  test_paths,  test_labels,  TEST_DIR),
]:
    prepare_dir(target_dir, class_labels)

    for path, lbl in zip(split_paths, split_labels):
        cls = class_labels[lbl]
        dest_path = os.path.join(target_dir, cls, os.path.basename(path))
        shutil.copy(path, dest_path)

    save_summary(target_dir, split_paths, split_labels, class_labels, split_name)

print("Test-Val-Split: Successfully ran the Train-test-val split script :✅")
send_message("Test-Val-Split: Successfully ran the Train-test-val split script ✅")
