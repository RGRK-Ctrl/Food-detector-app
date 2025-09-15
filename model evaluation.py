
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import pandas as pd
import os
import json
import numpy as np
from Image_Sequence import ImageSequence
from send_telegram import send_message

send_message("✅ Running evaluation script")
print("Model evaluation script:  Started running ✅")

MASTER_DATASET_PATH = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food Classification dataset'
FEEDBACK_PATH = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Feedback dataset'
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
REPORTS_DIR = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Reports"
MODEL_SUMMARY = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Reports/model_summary.txt"

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Load class labels saved during training
print(f"Model evaluation script:  Loading labels data from {LABELS_PATH} ✅")

with open(LABELS_PATH, "r") as f:
    class_info = json.load(f)
class_labels = class_info["labels"]
print(class_labels)


# Load test paths and labels from .npz
val_data = np.load(TRAIN_VAL_TEST_SPLIT_FILE, allow_pickle=True)
val_paths, val_labels     = val_data["val_paths"], val_data["val_labels"]

# ------------------------------
# Validation data summary
# ------------------------------
from collections import Counter

val_counts = Counter(val_labels)
summary_file = os.path.join(REPORTS_DIR, "validation_data_summary.txt")

with open(summary_file, "w") as f:
    f.write("📊 Validation Data Summary\n\n")
    f.write(f"Total validation samples: {len(val_labels)}\n")
    send_message(f"Total validation samples: {len(val_labels)}")

    f.write(f"Number of classes: {len(class_labels)}\n\n")
    send_message(f"Number of classes: {len(class_labels)}")

    print("📊 Validation class distribution:")
    for idx, count in val_counts.items():
        class_name = class_labels[idx]
        f.write(f"{class_name}: {count} samples\n")
        print(f"   {class_name}: {count}")

    f.write(f"\n✅ Total validation samples: {len(val_labels)}\n")

#print(f"✅ Validation data summary saved to {summary_file}")




# Use same custom ImageSequence generator
val_gen = ImageSequence(
    val_paths, val_labels,
    batch_size=16,
    img_size=(224, 224),
    num_classes=len(class_labels),
    shuffle=False,
    augment=False
)


# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model evaluation script:  Model loaded  ✅")
send_message("Model evaluation script:  Model loaded  ✅")

# Predict on the Validation dataset
print("Model evaluation script:  starting validation dataset predictions ✅")
send_message("Model evaluation script:  starting with validation dataset predictions ✅")

y_pred = model.predict(val_gen)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = val_labels  # True labels in order of val_gen


# Build class_indices manually from JSON
class_indices = {label: idx for idx, label in enumerate(class_labels)}
actual_class_labels = [None] * len(class_indices)
for class_name, idx in class_indices.items():
    actual_class_labels[idx] = class_name

# ✅ Overall metrics
print("Model evaluation script: Calculating overall metrics ✅")
send_message("Model evaluation script: Calculating overall metrics ✅")

accuracy = accuracy_score(y_true, y_pred_classes)
precision = precision_score(y_true, y_pred_classes, average="weighted")  # handles class imbalance
recall = recall_score(y_true, y_pred_classes, average="weighted")
f1 = f1_score(y_true, y_pred_classes, average="weighted")

import json
import os

# Your metric values
metrics = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4)
}

# Specify folder and filename
os.makedirs(REPORTS_DIR, exist_ok=True)  # create folder if it doesn't exist
json_file_path = os.path.join(REPORTS_DIR, "final_metrics.json")

# Save to file
with open(json_file_path, "w") as f:
    json.dump(metrics, f, indent=4)

print(f"Final metrics saved to REPORTS_DIR: {json_file_path}")


"""
print(f"📊 Accuracy:  {accuracy:.4f}")
print(f"🎯 Precision: {precision:.4f}")
print(f"🔄 Recall:    {recall:.4f}")
print(f"⚡ F1-score:  {f1:.4f}")
"""
send_message(f"Accuracy: {accuracy:.4f}")
send_message(f"Precision: {precision:.4f}")
send_message(f"Recall: {recall:.4f}")
send_message(f"F1 score: {f1:.4f}")
send_message("saved Final metrics into REPORTS_DIR")


# Generate classification report with correct label ordering
report_txt = classification_report(
    y_true,
    y_pred_classes,
    target_names=actual_class_labels,
    labels=list(range(len(actual_class_labels))),
    digits=4
)


#print("Model evaluation script: 📊 Classification Report:\n", report_txt)
with open(os.path.join(REPORTS_DIR, "Food detector classification_report.txt"), "w") as f:
    f.write(report_txt)
print("Model evaluation script: 📊 Classification Report Saved to:", REPORTS_DIR)


# Calculate precision and F1 scores for each class
precision_scores = precision_score(
    y_true, y_pred_classes, labels=list(range(len(actual_class_labels))), average=None
)
f1_scores = f1_score(
    y_true, y_pred_classes, labels=list(range(len(actual_class_labels))), average=None
)
metrics_data = {
    "Class": actual_class_labels,
    "Precision": precision_scores,
    "F1-Score": f1_scores
}
pd.DataFrame(metrics_data).to_csv(
    os.path.join(REPORTS_DIR, "precision_f1.csv"), index=False
)

print("Model evaluation script: Precision & F1 scores saved in Reports ✅")


# Confusion matrix plot
cm = confusion_matrix(y_true, y_pred_classes, labels=list(range(len(actual_class_labels))))
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=actual_class_labels, yticklabels=actual_class_labels
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(REPORTS_DIR, "Food detector_confusion_matrix.png"), dpi=300)
plt.close()
print("Model evaluation script: Confusion matrix Chart saved ✅.")


#----------------------------------
# Sending telegram message---------
#----------------------------------

send_message("Model evaluation script: Evaluation script ran successfully ✅")