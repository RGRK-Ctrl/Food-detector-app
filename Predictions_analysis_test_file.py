import numpy as np
import cv2
import json
from collections import Counter
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import pandas as pd
from Image_Sequence import ImageSequence
from send_telegram import send_message
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

send_message("Prediction analysis script:  Started running ✅")
print("Prediction analysis script:  Started running ✅")

# ===== PATHS =====
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
PREDICTIONS_DIR = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Predictions"

# ===== CREATE TIMESTAMPED OUTPUT FOLDER =====
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# ===== CREATE OUTPUT FOLDER and update the files =====
#os.makedirs(PREDICTIONS_DIR, exist_ok=True)
results_file = os.path.join(PREDICTIONS_DIR, f"testdata_inference_report.txt")
conf_matrix_file = os.path.join(PREDICTIONS_DIR, f"confusion_matrix.png")
per_class_accuracy = os.path.join(PREDICTIONS_DIR, f"per_class_accuracy.png")
csv_file = os.path.join(PREDICTIONS_DIR, f"Testdata_bulk_inference_analysis.csv")
per_class_csv = os.path.join(PREDICTIONS_DIR, f"Per_class_accuracy.csv")

# ===== LOAD MODEL =====
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# ===== LOAD LABELS =====
print(f"Loading labels from {LABELS_PATH}...")
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)["labels"]

# ===== Clear the directory =====
import os
import shutil

if os.path.exists(PREDICTIONS_DIR):
    for item in os.listdir(PREDICTIONS_DIR):
        item_path = os.path.join(PREDICTIONS_DIR, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
        elif os.path.isfile(item_path):
            os.remove(item_path)
    print(f"🧹 Cleared contents of: {PREDICTIONS_DIR}")



# ===== 🔍 DEBUGGING SECTION =====
send_message("Started debugging the script")
print(f"Loading Train-Val-Test Split file from {TRAIN_VAL_TEST_SPLIT_FILE}...")
data = np.load(TRAIN_VAL_TEST_SPLIT_FILE)
test_paths = data["test_paths"]
test_labels = data["test_labels"]
IMG_SIZE = (224, 224)


print("\n" + "=" * 50)
print("🔍 DEBUGGING: Class Index Alignment Check")
print("=" * 50)
# Step 1: Verify class index mapping

print("\n📋 Current class_labels.json mapping:")
for idx, label in enumerate(labels):
    print(f"Index {idx}: {label}")
print(f"\n📊 Total classes in JSON: {len(labels)}")
print(f"📊 Model output shape: {model.output_shape}")
print(f"📊 Expected vs Actual: {len(labels)} vs {model.output_shape[1]}")
if len(labels) != model.output_shape[1]:
    print("❌ MISMATCH: Number of classes in JSON doesn't match model output!")
else:
    print("\n✅ Class count matches model output")




# ===== DEBUGGING WITH 10 RANDOM TEST IMAGES =====
send_message("Prediction analysis script:  Started testing 10 samples ✅")
print("Prediction analysis script:  Started testing 10 samples ✅")

n_sample = min(10, len(test_paths))
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(test_paths), size=n_sample, replace=False)
sample_paths = test_paths[sample_idx]
sample_labels = test_labels[sample_idx]

# Using a generator only for these sample images
sample_gen = ImageSequence(
    sample_paths, sample_labels,
    batch_size=n_sample,
    img_size=(224, 224),
    num_classes=len(labels),
    shuffle=False,
    augment=False
)
#Run predictions
sample_preds = model.predict(sample_gen, verbose=0)

for i, (img_path, true_idx, probs) in enumerate(zip(sample_paths, sample_labels, sample_preds)):
    pred_idx = np.argmax(probs)
    top3_idx = probs.argsort()[-3:][::-1]
    print(f"\nImage: {os.path.basename(img_path)}")
    print(f"True label: {labels[true_idx]} (index {true_idx})")
    print(f"Predicted: {labels[pred_idx]} (index {pred_idx})")
    print("Top 3 predictions:")
    for j, idx in enumerate(top3_idx):
        print(f"  {j + 1}. {labels[idx]}: {probs[idx]:.4f}")

print("\n" + "=" * 50)
print("🔍 End Debugging Section")
print("=" * 50 + "\n")
send_message("Prediction analysis script:  Ended testing 10 samples ✅")
print("Prediction analysis script:  Ended testing 10 samples ✅")


# ===== FULL EVALUATION ON TEST DATA =====
send_message("Prediction analysis script:  Loading test data ✅")
print(f"Loading test data from {TRAIN_VAL_TEST_SPLIT_FILE}")

test_gen = ImageSequence(
    test_paths, test_labels,
    batch_size=16,
    img_size=(224,224),
    num_classes=len(labels),
    shuffle=False,
    augment=False
)

print("Prediction analysis script: Starting test data predictions ✅")
send_message("Prediction analysis script:  Starting test data predictions ✅")

preds = model.predict(test_gen, verbose=1)
pred_classes = preds.argmax(axis=1)
y_test = test_labels

# ===== CLASS DISTRIBUTION =====
class_counts = Counter(pred_classes)
print("\n📊 Predicted Class Distribution:")
for idx, count in class_counts.items():
    print(f"{labels[idx]}: {count}")

class_dist_msg = "📊 Predicted Class Distribution:\n"
for idx, count in class_counts.items():
    class_dist_msg += f"- {labels[idx]}: {count}\n"
send_message(class_dist_msg)

# ===== METRICS =====
acc = accuracy_score(y_test, pred_classes)
report = classification_report(y_test, pred_classes, target_names=labels, digits=4)
prec = precision_score(y_test, pred_classes, average="weighted")
rec = recall_score(y_test, pred_classes, average="weighted")
f1 = f1_score(y_test, pred_classes, average="weighted")

print("\n" + "="*50)
print("📊 Evaluation Metrics on Test Data")
print("="*50)
print(report)
print(f"\n✅🎯 Bulk Prediction Accuracy: {acc:.4f}")
print(f"✅🎯 Bulk Prediction Precision: {prec:.4f}")
print(f"✅🎯 Bulk Prediction Recall (weighted): {rec:.4f}")
print(f"✅🎯 Bulk Prediction F1-score (weighted): {f1:.4f}")

# === FIXED: Precision bug in Telegram message ===
summary_msg = (
    f"📊 Prediction Report\n"
    f"🎯 Bulk Prediction Accuracy: {acc:.4f}\n"
    f"🎯 Bulk Prediction Precision: {prec:.4f}\n"
    f"🎯 Bulk Prediction Recall (weighted): {rec:.4f}\n"
    f"🎯 Bulk Prediction F1-score (weighted): {f1:.4f}\n"
    f"✅ Total Test Images: {len(y_test)}\n"
)
send_message(summary_msg)

# ===== SAVE RESULTS TO FILE =====
with open(results_file, "w") as f:
    f.write(f"Prediction Run - {timestamp}\n")
    f.write("=" * 50 + "\n\n")
    f.write("📊 Predicted Class Distribution:\n")
    for idx, count in class_counts.items():
        f.write(f"{labels[idx]}: {count}\n")
    f.write("\n📊 Bulk Prediction metrics:\n")
    f.write(f"\n✅🎯 Bulk Prediction Accuracy: {acc:.4f}")
    f.write(f"\n✅🎯 Bulk Prediction Precision: {prec:.4f}")
    f.write(f"\n✅🎯 Bulk Prediction Recall (weighted): {rec:.4f}")
    f.write(f"\n✅🎯 Bulk Prediction F1-score (weighted): {f1:.4f}")

print(f"Bulk prediction test data: Metrics saved here → {results_file}")
send_message("✅🎯 Bulk prediction test data : Metrics saved in testdata_inference_report.txt")

# ===== SAVE DETAILED CSV =====
detailed_results = []
for img_path, true_idx, pred_idx, probs in zip(test_paths, y_test, pred_classes, preds):
    top3_idx = probs.argsort()[-3:][::-1]
    row = {
        "image_path": img_path,
        "true_label": labels[true_idx],
        "predicted_label": labels[pred_idx],
        "is_correct": true_idx == pred_idx,
    }
    # expand top-3 into columns instead of tuple list
    for j, idx in enumerate(top3_idx):
        row[f"top{j+1}_label"] = labels[idx]
        row[f"top{j+1}_confidence"] = round(probs[idx] * 100, 2)
    detailed_results.append(row)

df = pd.DataFrame(detailed_results)
df.to_csv(csv_file, index=False)
print(f"✅ Test data: Predictions CSV saved → {csv_file}")


import matplotlib.pyplot as plt
import pandas as pd

# Convert to DataFrame for easier grouping
results_df = df[["true_label", "is_correct"]].copy()
results_df = results_df.rename(columns={"true_label": "True_Label", "is_correct": "Is_Correct"})

# Count correct and failed per class
class_results = results_df.groupby(["True_Label", "Is_Correct"]).size().unstack(fill_value=0)
class_results = class_results.rename(columns={True: "Correct", False: "Failed"})

# Per class stacked bar chart
class_results[["Correct", "Failed"]].plot(
    kind="bar",
    stacked=True,
    figsize=(10,6)
)

plt.title("Class-wise Prediction Results")
plt.xlabel("Class")
plt.ylabel("Number of Predictions")
plt.legend(title="Prediction Outcome")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(per_class_accuracy)
plt.show()
plt.close()
print(f"Test data: Per class accuracy bar chart saved → {per_class_accuracy}")



# ===== PER-CLASS ACCURACY =====
per_class_acc = {}
for idx, label in enumerate(labels):
    mask = (y_test == idx)
    if mask.any():
        per_class_acc[label] = (pred_classes[mask] == idx).mean()
    else:
        per_class_acc[label] = None
pd.DataFrame(list(per_class_acc.items()), columns=["Class", "Accuracy"]).to_csv(per_class_csv, index=False)
print(f"Per-class accuracy CSV file is saved → {per_class_csv}")

# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - {timestamp}")
plt.tight_layout()
plt.savefig(conf_matrix_file)
plt.close()

print(f"Test data: Confusion matrix saved → {conf_matrix_file}")
print("Prediction analysis script: Execution Successful  ✅")
send_message("Prediction analysis script: Execution Successful ✅")