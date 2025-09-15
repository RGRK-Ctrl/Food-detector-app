import os
import shutil
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import json
import matplotlib.pyplot as plt
import requests
from Image_Sequence import ImageSequence
from send_telegram import send_message


# -----------------------------
# Paths
# -----------------------------
MASTER_DATASET_PATH = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food Classification dataset'
FEEDBACK_PATH = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Feedback dataset'
REPORTS_DIR = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Reports'
CHARTS_DIR = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Charts'
MODEL_PATH = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Food detector model.keras"
TRAINING_LOG = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/logs/training_log.csv'
LABELS_PATH = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/logs/class_labels.json"
CHECKPOINTS_DIR = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Checkpoints"
TRAIN_VAL_TEST_SPLIT_FILE = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/train_val_test_split.npz'
TEST_DIR = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food images Test set'
TRAIN_DIR = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food images Training set"
VAL_DIR   = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food images Validation set"
REPORTS_DIR_JSON = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Reports/model_param_summary.json"
MODEL_SUMMARY = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Reports/model_summary.txt"
LOGS_DIR = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/logs"
DATASET_ANALYSIS="/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Dataset Analysis"
os.makedirs(LOGS_DIR, exist_ok=True)


os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)




# -----------------------------
# Merge Feedback Dataset
# -----------------------------
print("🔄 Starting feedback dataset merge...")
merged_images = 0
for class_name in os.listdir(FEEDBACK_PATH):
    feedback_class_path = os.path.join(FEEDBACK_PATH, class_name)
    master_class_path = os.path.join(MASTER_DATASET_PATH, class_name)

    if not os.path.isdir(feedback_class_path):
        continue

    feedback_images = os.listdir(feedback_class_path)
    if not feedback_images:
        os.rmdir(feedback_class_path)
        print(f"🗑️ Removed empty folder: {feedback_class_path}")
        continue

    os.makedirs(master_class_path, exist_ok=True)
    for img_file in feedback_images:
        src = os.path.join(feedback_class_path, img_file)
        dst = os.path.join(master_class_path, img_file)
        shutil.move(src, dst)
        merged_images += 1

    print(f"✅ Merged {len(feedback_images)} feedback images into '{class_name}'")
    os.rmdir(feedback_class_path)
    print(f"🗑️ Removed merged folder: {feedback_class_path}")

print(f"✅ Feedback merge completed. Total merged images: {merged_images}")

# -----------------------------
# Trigger Data analysis Script
# -----------------------------
import subprocess
print("🚀 Running Data analysis script...")
send_message("🚀 Data analysis script triggered")

subprocess.run([
    "python",
    "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/dataanalysis.py"
])


#---------------------------
#   Size Inputs
#---------------------------
img_rows, img_cols = 224, 224
batch_size = 20
num_epoch_stage1 = 15
num_epoch_stage2 = 15
CONFIDENCE_THRESHOLD = 50


# -----------------------------
# Class labels (labels + stats)
# -----------------------------
data_dir_list = sorted([
    d for d in os.listdir(MASTER_DATASET_PATH)
    if os.path.isdir(os.path.join(MASTER_DATASET_PATH, d))],
    key=str.lower
)

class_labels = data_dir_list



#----------------------------------
# -------- Save class info --------
#----------------------------------
df_counts = pd.DataFrame({
    "Class": class_labels,
    "Image_Count": [
        len([f for f in os.listdir(os.path.join(MASTER_DATASET_PATH, c))
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))])
        for c in class_labels
    ]
})
class_info = {
    "total_images": int(df_counts["Image_Count"].sum()),
    "total_classes": len(df_counts),
    "labels": df_counts["Class"].tolist(),
    "counts": dict(zip(df_counts["Class"], df_counts["Image_Count"]))
}
with open(LABELS_PATH, "w") as f:
    json.dump(class_info, f, indent=4)
print(f"✅ Saved class info with counts to {LABELS_PATH}")





# ----------------------------------
# -------- Gather all image paths and labels --------
# ----------------------------------
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
image_paths = []
image_labels = []
for idx, dataset in enumerate(data_dir_list):
    class_folder = os.path.join(MASTER_DATASET_PATH, dataset)
    for fname in os.listdir(class_folder):
        if fname.lower().endswith(valid_extensions):
            image_paths.append(os.path.join(class_folder, fname))
            image_labels.append(idx)
image_paths, image_labels = np.array(image_paths), np.array(image_labels)
num_classes = len(class_labels)



# ----------------------------------
# SPlit the TRAIN-VALIDATION_TEST data into respective directrories
# ----------------------------------

print(f"✅ Train-Val-Test split began")
send_message("Train-Val-Test_split.py script triggered")

import subprocess
subprocess.run([
    "python",
    "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Train-Val-Test_split.py"
])
print(f"✅ Successfully saved Training data in {TRAIN_DIR},Validation data in {VAL_DIR},Test data in {TEST_DIR}")





# -------- Stopping the script execution --------
"""
import sys
sys.exit("🚫 Stopping script execution here.")
"""


# ----------------------------------
# -------- Class weights for imbalance--------
# ----------------------------------
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------
# Load train/val/test split
# ------------------------------
data = np.load(TRAIN_VAL_TEST_SPLIT_FILE, allow_pickle=True)

train_paths, train_labels = data["train_paths"], data["train_labels"]
val_paths, val_labels     = data["val_paths"], data["val_labels"]
test_paths, test_labels   = data["test_paths"], data["test_labels"]
class_labels              = data["class_labels"]

print(f"✅ Loaded split: {len(train_paths)} train, {len(val_paths)} val, {len(test_paths)} test")


# ------------------------------
# Compute class weights on train
# ------------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights = dict(enumerate(class_weights))
send_message("✅ Class weights computed:")
print("✅ Class weights computed:")
Class_weights_file = os.path.join(DATASET_ANALYSIS, "class_weights_file.txt")
with open(Class_weights_file, "w") as f:
    f.write("📊 Class weights Data Summary\n\n")
    for cls_idx, weight in class_weights.items():
        f.write(f"  Class {cls_idx}: {weight:.4f}\n")
send_message("✅ Class weights saved in DATASET_ANALYSIS/class_weights_file.txt:")

# ----------------------------------
# -------- Create generators --------
# ----------------------------------
from Image_Sequence import ImageSequence

print("✅ Creating Training, Validation and Test sets via generators")

# Helper to collect paths and labels from a directory
def get_paths_and_labels(base_dir, class_labels):
    paths, labels = [], []
    for idx, cls in enumerate(class_labels):
        class_dir = os.path.join(base_dir, cls)
        if not os.path.exists(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                paths.append(os.path.join(class_dir, fname))
                labels.append(idx)
    return paths, labels

# Collect train/val/test paths and labels
train_paths, train_labels = get_paths_and_labels(TRAIN_DIR, class_labels)
val_paths, val_labels     = get_paths_and_labels(VAL_DIR, class_labels)
test_paths, test_labels   = get_paths_and_labels(TEST_DIR, class_labels)

# Generators
train_gen = ImageSequence(
    train_paths, train_labels, batch_size,
    (img_rows, img_cols), num_classes,
    shuffle=True, augment=True   # ✅ Augmentation ON
)

val_gen = ImageSequence(
    val_paths, val_labels, batch_size,
    (img_rows, img_cols), num_classes,
    shuffle=False, augment=False # ✅ Clean validation set
)

test_gen = ImageSequence(
    test_paths, test_labels, batch_size,
    (img_rows, img_cols), num_classes,
    shuffle=False, augment=False # ✅ Final evaluation set
)



#-----------------------------
# -------- Build Model: EfficientNetV2B0 --------
#-----------------------------

print("🔄 Building EfficientNetV2B0 model...")

base_model = EfficientNetV2B0(weights="imagenet", include_top=False, input_shape=(img_rows, img_cols, 3))

# -------Freeze base for Stage 1------
print("🔄 Freezing base model for Stage 1...")

for layer in base_model.layers:
    layer.trainable = False
print("✅ Model built and base layers frozen.")

# Add pooling, dropout for regularization
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)  # try 0.2,0.4
x = Dense(128, activation='relu')(x)
x = Dropout(0.4)(x)
predictions = Dense(num_classes, activation='softmax')(x)
transfer_model = Model(inputs=base_model.input, outputs=predictions)

from send_telegram import send_message
send_message("✅ Freezing base model for Stage 1")





# -------- CALLBACKS --------
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
import time
from tensorflow.keras.callbacks import Callback
import csv

import time, csv
from keras.callbacks import Callback

class EpochTimeLogger(Callback):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.csv_file = None
        self.writer = None
        self.epoch_times = []   # ✅ make sure this line is here

    def on_train_begin(self, logs=None):
        self.csv_file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        header = ['epoch', 'duration', 'train_loss', 'train_acc', 'val_loss', 'val_acc']
        self.writer.writerow(header)
        self.csv_file.flush()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        duration = time.time() - self.epoch_start
        mins = int(duration // 60)
        secs = int(duration % 60)
        duration_str = f"{mins}m {secs}s"

        # ✅ append to list
        self.epoch_times.append(duration)

        avg_time = sum(self.epoch_times) / len(self.epoch_times)
        total_epochs = self.params.get("epochs", 0)
        remaining_epochs = total_epochs - (epoch + 1)
        eta_seconds = int(remaining_epochs * avg_time)
        eta_mins = eta_seconds // 60
        eta_secs = eta_seconds % 60
        eta_str = f"{eta_mins}m {eta_secs}s"

        # Metrics
        train_loss = logs.get("loss")
        train_acc = logs.get("accuracy")
        val_loss = logs.get("val_loss")
        val_acc = logs.get("val_accuracy")

        row = [
            epoch + 1,
            duration_str,
            f"{train_loss:.4f}" if train_loss is not None else "",
            f"{train_acc:.4f}" if train_acc is not None else "",
            f"{val_loss:.4f}" if val_loss is not None else "",
            f"{val_acc:.4f}" if val_acc is not None else ""
        ]
        self.writer.writerow(row)
        self.csv_file.flush()

        message = (
            f"⏱ Epoch {epoch+1} took {mins}m {secs}s\n"
            f"📉 Loss: {train_loss:.4f} | ✅ Accuracy: {train_acc:.4f}\n"
            f"🧪 Val Loss: {val_loss:.4f} | 🎯 Val Accuracy: {val_acc:.4f}"
        )
        print(message)
        if 'send_message' in globals():
            send_message(message)

    def on_train_end(self, logs=None):
        if self.csv_file:
            self.csv_file.close()

# Stage 1 checkpoint
checkpoint_stage1_filepath = os.path.join(CHECKPOINTS_DIR, "stage1_epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.keras")
checkpoint_stage1_callback = ModelCheckpoint(
    filepath=checkpoint_stage1_filepath,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1,
)

# Stage 2 checkpoint
checkpoint_stage2_filepath = os.path.join(CHECKPOINTS_DIR, "stage2_epoch_{epoch:02d}_valacc_{val_accuracy:.4f}.keras")
checkpoint_stage2_callback = ModelCheckpoint(
    filepath=checkpoint_stage2_filepath,
    save_best_only=True,
    monitor="val_accuracy",
    mode="max",
    verbose=1,
)

early_stopping_callback = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
    min_delta = 0.01,  # ✅ only stop/save when val_accuracy improves by ≥ 1%

)
reduce_lr_callback = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=2,
    min_lr=1e-6,
    verbose=1
)
from tensorflow.keras.callbacks import CSVLogger

csv_logger_stage1 = CSVLogger("/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/logs/training_log_stage1.csv", append=False)
csv_logger_stage2 = CSVLogger("/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/logs/training_log_stage2.csv", append=False)



# -----------------------------
# Training: Stage 1 (Frozen Layers)
# -----------------------------
send_message("-------------------------------")
print("\n\n🚀 Stage 1: Training with frozen base layers...")
send_message("✅ Stage 1: Began: Training with frozen base layers")

start_time1 = time.time()
stage1_log_path = os.path.join(LOGS_DIR, "stage1_epoch_logs.csv")
epoch_timer1 = EpochTimeLogger(filename=stage1_log_path)

transfer_model.compile(optimizer=Adam(learning_rate=1e-4), loss="categorical_crossentropy", metrics=["accuracy"])
hist1 = transfer_model.fit(
    train_gen,
    epochs=num_epoch_stage1,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[checkpoint_stage1_callback, early_stopping_callback, reduce_lr_callback, epoch_timer1,csv_logger_stage1]
)


# Stage 1 training Duration
duration_stage1 = time.time() - start_time1
hours1 = int(duration_stage1 // 3600)
minutes1 = int((duration_stage1 % 3600) // 60)
seconds1 = int(duration_stage1 % 60)

print(f"🚀 Stage 1: Training took {hours1}h {minutes1}m {seconds1}s")

import json
send_message("saving the stage 1 training metrics")
# Saves training metrics for each epoch
os.makedirs(LOGS_DIR, exist_ok=True)
with open(os.path.join(LOGS_DIR, "history_stage1.json"), "w") as f:
    json.dump(hist1.history, f)

# -----------------------------
# Training: Stage 1 Telegram update sent
# -----------------------------
# Stage 1 metrics
stage1_acc = hist1.history['accuracy'][-1]
stage1_loss = hist1.history['loss'][-1]
stage1_val_acc = hist1.history['val_accuracy'][-1]
stage1_val_loss = hist1.history['val_loss'][-1]

import subprocess

# Send Stage 1 summary
subprocess.run([
    "python", "send_telegram.py",
    f"🚀 Stage 1 Complete:\n"
    f"Train Acc: {stage1_acc:.4f}, Loss: {stage1_loss:.4f}\n"
    f"Val Acc: {stage1_val_acc:.4f}, Val Loss: {stage1_val_loss:.4f}\n"
    f"Duration: {hours1}h {minutes1}m {seconds1}s"
])
# ----------------------------------------------------------




# -------- Stage 2: Unfreezing & begin Fine-tuning --------
print("\n\n\n🚀 Stage 2: Unfreezing: Began Fine-tuning top layers")
send_message("✅ Stage 2: Unfreezing: Began Fine-tuning top layers")

start_time2 = time.time()
stage1_log_path = os.path.join(LOGS_DIR, "stage2_epoch_logs.csv")
epoch_timer2 = EpochTimeLogger(filename=stage1_log_path)

for layer in base_model.layers:
    layer.trainable = True
transfer_model.compile(optimizer=Adam(learning_rate=3e-5), loss="categorical_crossentropy", metrics=["accuracy"])
hist2 = transfer_model.fit(
    train_gen,
    epochs=num_epoch_stage2,
    validation_data=val_gen,
    class_weight=class_weights,
    callbacks=[checkpoint_stage2_callback, early_stopping_callback, reduce_lr_callback, epoch_timer2, csv_logger_stage2])


# Stage 2 training Duration
duration_stage2 = time.time() - start_time2
hours2 = int(duration_stage2 // 3600)
minutes2 = int((duration_stage2 % 3600) // 60)
seconds2 = int(duration_stage2 % 60)


print(f"🚀 Stage 2: Training took {hours2}h {minutes2}m {seconds2}s")
message2=f"🚀 Stage 2: Training took {hours2}h {minutes2}m {seconds2}s"
send_message(message2)

# -----------------------------
# Training: Stage 1 & 2 Final Summary
# -----------------------------
total_duration = duration_stage1 + duration_stage2
total_hours = int(total_duration // 3600)
total_minutes = int((total_duration % 3600) // 60)
total_seconds = int(total_duration % 60)

print(f"\n🏁 Summary of Total Training Time: {total_hours}h {total_minutes}m {total_seconds}s")


#----------------------------------------
# Training summary saving
#---------------------------------------

# Convert duration to human-readable string
def format_duration(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"

training_summary = {
    "stage1": {
        "duration": format_duration(duration_stage1),
        "epochs": len(hist1.history["loss"]),
        "final_train_acc": hist1.history["accuracy"][-1],
        "final_train_loss": hist1.history["loss"][-1],
        "final_val_acc": hist1.history["val_accuracy"][-1],
        "final_val_loss": hist1.history["val_loss"][-1]
    },
    "stage2": {
        "duration": format_duration(duration_stage2),
        "epochs": len(hist2.history["loss"]),
        "final_train_acc": hist2.history["accuracy"][-1],
        "final_train_loss": hist2.history["loss"][-1],
        "final_val_acc": hist2.history["val_accuracy"][-1],
        "final_val_loss": hist2.history["val_loss"][-1]
    },
    "batch_size": batch_size,
    "confidence_threshold": CONFIDENCE_THRESHOLD,
    "total_duration": format_duration(duration_stage1 + duration_stage2)
}

with open(os.path.join(LOGS_DIR, "training_summary.json"), "w") as f:
    json.dump(training_summary, f, indent=2)



# Saves training metrics for each epoch
send_message("saving the stage 2 training metrics")
with open(os.path.join(LOGS_DIR, "history_stage2.json"), "w") as f:
    json.dump(hist2.history, f)


# Plotting training metrics for Stage 1 and Stage 2
def plot_history(history, stage_name):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    epochs_range = range(1, len(acc) + 1)

    # Save history dict as JSON for later use
    with open(f"/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/logs/history_{stage_name.lower()}.json", "w") as f:
        json.dump(history.history, f)

    # Accuracy plot
    plt.figure(figsize=(6,4))
    plt.plot(epochs_range, acc, label="Train Accuracy")
    plt.plot(epochs_range, val_acc, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(f"{stage_name} Accuracy")
    plt.legend()
    plt.tight_layout()

    # Save in static folder (for web app)
    plt.savefig(
        f"/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/static/images/{stage_name.lower()}_accuracy.png")

    # Save in logs folder (for reference)
    plt.savefig(
        f"/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/logs/{stage_name.lower()}_accuracy.png")
    plt.close()


    # Loss plot
    plt.figure(figsize=(6, 4))
    plt.plot(epochs_range, loss, label="Train Loss")
    plt.plot(epochs_range, val_loss, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(f"{stage_name} Loss")
    plt.legend()
    plt.tight_layout()

    # Save in static folder (for web app)
    plt.savefig(
        f"/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/static/images/{stage_name.lower()}_loss.png")

    # Save in logs folder (for reference)
    plt.savefig(
        f"/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/logs/{stage_name.lower()}_loss.png")

    plt.close()

# Example usage
plot_history(hist1, "Stage1")
plot_history(hist2, "Stage2")



#-----------------
# Saving the model
#-----------------
print("\n✅ Training complete. Saving model...")
transfer_model.save(MODEL_PATH, include_optimizer=True)
print("✅ Model saved at", MODEL_PATH)


#-----------------------------
# Saving the Model & parameter summary
#-----------------------------
import numpy as np
import json

# Save model summary to a text file

with open(MODEL_SUMMARY, "w") as f:
    transfer_model.summary(print_fn=lambda x: f.write(x + "\n"))

print(f"✏️ Model summary saved to {MODEL_SUMMARY}")

#Send Message---------------------------------------
message3=f"✏️ Model summary saved to {MODEL_SUMMARY}"
send_message(message3)


# Save Parameter summary to a JSON file
total_params = transfer_model.count_params()
trainable_params = np.sum([np.prod(v.shape) for v in transfer_model.trainable_variables])
non_trainable_params = total_params - trainable_params
param_summary = {
    "Total Parameters": int(total_params),
    "Trainable Parameters": int(trainable_params),
    "Non-Trainable Parameters": int(non_trainable_params)
}


with open(REPORTS_DIR_JSON, "w") as f:
    json.dump(param_summary, f, indent=4)

print("✏️ Model parameter summary saved to")
print(f"{REPORTS_DIR_JSON}")



# -----------------------------
# Training: Stage 2 Telegram update sent
# -----------------------------
# Stage 2 metrics
stage2_acc = hist2.history['accuracy'][-1]
stage2_loss = hist2.history['loss'][-1]
stage2_val_acc = hist2.history['val_accuracy'][-1]
stage2_val_loss = hist2.history['val_loss'][-1]

# Send Stage 2 summary
subprocess.run([
    "python", "send_telegram.py",
    f"🚀 Stage 2 Complete:\n"
    f"Train Acc: {stage2_acc:.4f}, Loss: {stage2_loss:.4f}\n"
    f"Val Acc: {stage2_val_acc:.4f}, Val Loss: {stage2_val_loss:.4f}\n"
    f"Duration: {hours2}h {minutes2}m {seconds2}s"
])


# Send final training summary #-----------------------------
subprocess.run([
    "python", "send_telegram.py",
    f"🏁 Training Completed:\n"
    f"Total Duration: {total_hours}h {total_minutes}m {total_seconds}s\n"
    f"Best Val Acc: {max(stage1_val_acc, stage2_val_acc):.4f}"
])

# -----------------------------
# Trigger Evaluation Script
# -----------------------------
import subprocess
print("\n🚀 Running evaluation script...")
send_message("-------------------------------")
send_message("🚀 Evaluation script triggered")

subprocess.run([
    "python",
    "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/model evaluation.py"
])

# -----------------------------
# Trigger Prediction analysis Script
# -----------------------------
import subprocess
print("\n🚀 Running Prediction analysis script...")
send_message("-------------------------------")
send_message("🚀 Prediction analysis  triggered")

subprocess.run([
    "python",
    "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Predictions_analysis.py"
])


send_message("Food detector training script execution completed ✅")

