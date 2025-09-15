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
STATIC_IMAGES = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/static/images"

# ===== CREATE OUTPUT FOLDER and update the files =====
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file = os.path.join(PREDICTIONS_DIR, f"testdata_inference_report.txt")
conf_matrix_file = os.path.join(PREDICTIONS_DIR, f"confusion_matrix.png")
predictions_csv_file = os.path.join(PREDICTIONS_DIR, f"Predictions.csv")
Testdata_bulk_inference_analysis= os.path.join(PREDICTIONS_DIR, f"Testdata_bulk_inference_analysis.csv")
per_class_acuracy_bar = os.path.join(PREDICTIONS_DIR, f"per_class_acuracy_bar.png")
top_k_misclassifications = os.path.join(PREDICTIONS_DIR, f"top_k_misclassifications.png")
class_index_mapping = os.path.join(PREDICTIONS_DIR, f"class_index_mapping.txt")
classification_report_file = os.path.join(PREDICTIONS_DIR, f"classification_report.txt")
ten_sample_debugging_predictions_report = os.path.join(PREDICTIONS_DIR, f"ten_sample_debugging_predictions_report.txt")

top_k_misclassifications_static = os.path.join(STATIC_IMAGES, f"top_k_misclassifications.png")
per_class_acuracy_bar_static = os.path.join(STATIC_IMAGES, f"per_class_acuracy_bar.png")


# ===== LOAD MODEL =====
print(f"Loading model from {MODEL_PATH}...")
model = load_model(MODEL_PATH)

# ===== LOAD LABELS =====
print(f"Loading labels from {LABELS_PATH}...")
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)["labels"]
#----------------------------------------------------------------------------



# ===== 🔍 DEBUGGING SECTION (ADD THIS) =====
send_message("Started debugging the script")
print(f"Loading Train-Val-Test Split file from {TRAIN_VAL_TEST_SPLIT_FILE}...")
data = np.load(TRAIN_VAL_TEST_SPLIT_FILE)
test_paths = data["test_paths"]
test_labels = data["test_labels"]

IMG_SIZE = (224, 224)


#------DEBUGGING: Class Index Alignment Check------
print("\n" + "=" * 50)
print("🔍 DEBUGGING: Class Index Alignment Check")
print("=" * 50)

# Step 1: Verify class index mapping
mapping_lines = []
mapping_lines.append("\n📋 Current class_labels.json mapping:")

for idx, label in enumerate(labels):
    line = f"Index {idx}: {label}"
    #print(line)
    mapping_lines.append(line)

# Step 2: Extra info
extra_info = [
    f"\n📊 Total classes in JSON: {len(labels)}",
    f"📊 Model output shape: {model.output_shape}",
    f"📊 Expected vs Actual: {len(labels)} vs {model.output_shape[1]}"
]

for info in extra_info:
    print(info)
    mapping_lines.append(info)

# Step 3: Check alignment
if len(labels) != model.output_shape[1]:
    msg = "❌ MISMATCH: Number of classes in JSON doesn't match model output!"
else:
    msg = "\n✅ Class count matches model output"

print(msg)
mapping_lines.append(msg)

# Step 4: Save to file
os.makedirs(os.path.dirname(class_index_mapping), exist_ok=True)
with open(class_index_mapping, "w", encoding="utf-8") as f:
    f.write("\n".join(mapping_lines))

print(f"\n💾 Saved class index mapping to: {class_index_mapping}")





# Load class labels saved during training
print(f"\nModel evaluation script:  Loading labels data from {LABELS_PATH} ✅")

with open(LABELS_PATH, "r") as f:
    class_info = json.load(f)
class_labels = class_info["labels"]
#print(class_labels)



send_message("Prediction analysis script:  Started testing 10 samples ✅")
print("Prediction analysis script:  Started testing 10 samples ✅")

# ===== DEBUGGING WITH 10 RANDOM TEST IMAGES =====
import numpy as np

n_sample = min(10, len(test_paths))
rng = np.random.default_rng(42)  # reproducibility
sample_idx = rng.choice(len(test_paths), size=n_sample, replace=False)
sample_paths = test_paths[sample_idx]
sample_labels = test_labels[sample_idx]

# Create a generator only for these sample images
sample_gen = ImageSequence(
    sample_paths, sample_labels,
    batch_size=n_sample,
    img_size=(224, 224),
    num_classes=len(labels),
    shuffle=False,
    augment=False
)

# Run predictions
results = []
sample_preds = model.predict(sample_gen, verbose=0)

# Collect results
for i, (img_path, true_idx, probs) in enumerate(zip(sample_paths, sample_labels, sample_preds)):
    pred_idx = np.argmax(probs)
    top3_idx = probs.argsort()[-3:][::-1]

    entry = []
    entry.append(f"\nImage: {os.path.basename(img_path)}")
    entry.append(f"True label: {labels[true_idx]} (index {true_idx})")
    entry.append(f"Predicted: {labels[pred_idx]} (index {pred_idx})")
    entry.append("Top 3 predictions:")
    for j, idx in enumerate(top3_idx):
        entry.append(f"  {j + 1}. {labels[idx]}: {probs[idx]:.4f}")

    # Print to console
    print("\n".join(entry))

    # Save for later
    results.append("\n".join(entry))

# Write everything into a file
with open(ten_sample_debugging_predictions_report, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\n💾 Saved random sample predictions to {ten_sample_debugging_predictions_report}")

print("\n" + "="*50)
print("🔍 End Debugging Section")
print("="*50 + "\n")
send_message("Prediction analysis script:  Ended testing 10 samples  ✅")
print("Prediction analysis script:  Ended testing 10 samples ✅")




#----------------------------------------------------------------------------
# ===== LOAD TEST DATA =====
send_message("Prediction analysis script:  Loading test data ✅")

print("Loading test data from")
print(f"{TRAIN_VAL_TEST_SPLIT_FILE}")

data = np.load(TRAIN_VAL_TEST_SPLIT_FILE)
test_paths = data["test_paths"]
test_labels = data["test_labels"]

IMG_SIZE = (224, 224)
# ===== FULL EVALUATION ON TEST DATA =====
test_gen = ImageSequence(
    test_paths, test_labels,
    batch_size=32,
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


# From here: keep your metrics, reports, confusion matrix generation unchanged
# ===== CLASS DISTRIBUTION =====
class_counts = Counter(pred_classes)
"""
print("\n📊 Predicted Class Distribution:")
for idx, count in class_counts.items():
    print(f"{labels[idx]}: {count}")

# ===== Send to Telegram =====
class_dist_msg = "📊 Predicted Class Distribution:\n"
for idx, count in class_counts.items():
    class_dist_msg += f"- {labels[idx]}: {count}\n"

send_message(class_dist_msg)
"""


# ===== METRICS =====
acc = accuracy_score(y_test, pred_classes)
report = classification_report(y_test, pred_classes, target_names=labels, digits=4)
prec = precision_score(y_test, pred_classes, average="weighted")
rec = recall_score(y_test, pred_classes, average="weighted")
f1 = f1_score(y_test, pred_classes, average="weighted")


print("\n" + "="*50)
print("📊 Evaluation Metrics on Test Data")
print("="*50)



with open(classification_report_file, "w", encoding="utf-8") as f:
    f.write(report)

print(f"💾 Saved classification report to: {classification_report_file}")

print(f"\n✅🎯 Bulk Prediction Accuracy: {acc:.4f}")
print(f"\n✅🎯 Bulk Prediction Precision: {prec:.4f}")
print(f"\n✅🎯 Bulk Prediction Recall (weighted): {rec:.4f}")
print(f"\n✅🎯 Bulk Prediction F1-score (weighted): {f1:.4f}")

# ===== Send Telegram Notification =====
summary_msg = (
    f"📊 Prediction Report\n"
    f"🎯 Bulk Prediction Accuracy: {acc:.4f}\n"
    f"🎯 Bulk Prediction Precision: {prec:.4f}\n"
    f"🎯 Bulk Prediction Recall (weighted): {rec:.4f}"
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
    f.write("\n")

    f.write("📊 Bulk Prediction metrics:\n")
    f.write(f"\n✅🎯 Bulk Prediction Accuracy: {acc:.4f}")
    f.write(f"\n✅🎯 Bulk Prediction Precision: {prec:.4f}")
    f.write(f"\n✅🎯 Bulk Prediction Recall (weighted): {rec:.4f}")
    f.write(f"\n✅🎯 Bulk Prediction F1-score (weighted): {f1:.4f}")

print("Bulk prediction test data : Metrics saved here in testdata_inference_report.txt")
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
df.to_csv(Testdata_bulk_inference_analysis, index=False)
print(f"✅ Test data: Predictions CSV saved → {Testdata_bulk_inference_analysis}")





# Load CSV
df = pd.read_csv("/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Predictions/Testdata_bulk_inference_analysis.csv")

if os.path.exists(Testdata_bulk_inference_analysis) and os.path.getsize(Testdata_bulk_inference_analysis) > 0:
    df = pd.read_csv(Testdata_bulk_inference_analysis)
    print("✅ CSV loaded:", df.shape)
else:
    print("⚠️ CSV is empty or missing, skipping analysis.")
    df = None

# Create output directory
output_dir = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Predictions"
os.makedirs(output_dir, exist_ok=True)

# --- Misclassifications ---
misclassified = df[df["is_correct"] == False]

# 1. Total misclassifications count
total_misclassifications = len(misclassified)

# 2. Top 10 most confused classes (true_label → predicted_label)
confusions = (
    misclassified.groupby(["true_label", "predicted_label"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

top10_confusions = confusions.head(10)

# Save as CSV
summary_path = os.path.join(output_dir, "misclassification_summary.csv")
with open(summary_path, "w") as f:
    f.write(f"Total Misclassifications,{total_misclassifications}\n\n")
    f.write(f"Total 10 confused labels\n")
    top10_confusions.to_csv(f, index=False)



# ===== SAVE DETAILED CSV =====
detailed_results = []
for img_path, true_idx, pred_idx, probs in zip(test_paths, y_test, pred_classes, preds):
    top3_idx = probs.argsort()[-3:][::-1]  # top 3 indices
    top3_info = [(labels[i], round(probs[i] * 100, 2)) for i in top3_idx]
    detailed_results.append({
        "image_path": img_path,
        "true_label": labels[true_idx],
        "predicted_label": labels[pred_idx],
        "is_correct": true_idx == pred_idx,
        "top3_predictions": top3_info
    })

df = pd.DataFrame(detailed_results)
df.to_csv(predictions_csv_file, index=False)
print("Test data : Predictions.csv file saved here")
print(f"✅ {predictions_csv_file}")




# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Confusion Matrix - {timestamp}")
plt.tight_layout()
plt.savefig(conf_matrix_file)
plt.close()
print("Test data : Confusion matrix saved here")
print(f"✅ {conf_matrix_file}")


# ===== CONFUSION MATRIX =====
cm = confusion_matrix(y_test, pred_classes)
num_classes = len(labels)

# ---- 1) Per-Class Accuracy Bar Plot ----
# Compute per-class accuracy
acc_per_class = cm.diagonal() / cm.sum(axis=1)

# Sort accuracies + labels together
sorted_idx = np.argsort(acc_per_class)[::-1]  # descending order
acc_sorted = acc_per_class[sorted_idx]
labels_sorted = np.array(labels)[sorted_idx]

# Plot per-class accuracy (sorted)
plt.figure(figsize=(8, 12))
sns.barplot(
    x=acc_sorted,
    y=labels_sorted,
    orient="h",
    hue=labels_sorted,  # assign hue
    dodge=False,
    legend=False,
    palette="Blues_d"
)
plt.xlabel("Accuracy")
plt.ylabel("Class")
plt.title("Per-Class Accuracy (Sorted)")
plt.tight_layout()
plt.savefig(per_class_acuracy_bar)
plt.savefig(per_class_acuracy_bar_static)

print("Test data : Per class accuracy bar chart saved here")
print(f"✅ {per_class_acuracy_bar}")

# ---- 2) Top-K Misclassifications Heatmap ----
K = 10  # number of top misclassifications
misclassified = []

for i in range(num_classes):
    for j in range(num_classes):
        if i != j and cm[i, j] > 0:
            misclassified.append((labels[i], labels[j], cm[i, j]))

# Sort by count of misclassifications
misclassified_sorted = sorted(misclassified, key=lambda x: x[2], reverse=True)[:K]

# Convert to DataFrame for heatmap
df_mis = pd.DataFrame(misclassified_sorted, columns=["True", "Predicted", "Count"])
# Convert counts to int
pivot_mis = df_mis.pivot(index="True", columns="Predicted", values="Count").fillna(0).astype(int)

# Plot heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(pivot_mis, annot=True, fmt="d", cmap="Reds")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Top-{K} Misclassifications Heatmap")
plt.tight_layout()
plt.savefig(top_k_misclassifications)
plt.savefig(top_k_misclassifications_static)

plt.close()

print(f"Test data : Saved Top-{K} Misclassifications plot here")
print(f"✅ {top_k_misclassifications}")

send_message("Prediction analysis script:  Successfully ran the prediction script ✅")
