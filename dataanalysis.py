import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
from scipy.stats import chisquare

from send_telegram import send_message
send_message("🚀 Data analysis script started execution")

# -----------------------------
# Configurations
# -----------------------------
MASTER_DATASET_PATH = '/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Food Classification dataset'

img_rows, img_cols = 128, 128
save_dir = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Dataset Analysis"
os.makedirs(save_dir, exist_ok=True)
LABELS_PATH = "/Users/goutham-18258/PycharmProjects/Image classification/Food detector app/Model_data/Dataset Analysis/class_labels.json"

valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')



# -----------------------------
# Dataset Loading
# -----------------------------
class_counts = {}
image_shapes = []
failed_images = []
"""
for cls in sorted(os.listdir(MASTER_DATASET_PATH)):
    cls_path = os.path.join(MASTER_DATASET_PATH, cls)
    if not os.path.isdir(cls_path):
        continue
"""
for cls in sorted(os.listdir(MASTER_DATASET_PATH), key=str.lower):
    cls_path = os.path.join(MASTER_DATASET_PATH, cls)
    if not os.path.isdir(cls_path):
        continue
    count = 0
    for file in os.listdir(cls_path):
        if not file.lower().endswith(valid_extensions):
            continue
        try:
            img_path = os.path.join(cls_path, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                failed_images.append(img_path)
                continue
            count += 1
            image_shapes.append(img.shape)
        except Exception as e:
            failed_images.append(img_path)
            continue

    class_counts[cls] = count

# -----------------------------
# Save Class Distribution
# -----------------------------
df_counts = pd.DataFrame(list(class_counts.items()), columns=["Class", "Image_Count"])
df_counts["Imbalance_Ratio"] = df_counts["Image_Count"] / df_counts["Image_Count"].sum()
df_counts.to_csv(os.path.join(save_dir, "class_distribution.csv"), index=False)

# Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x="Class", y="Image_Count", data=df_counts, palette="viridis", hue="Class", legend=False)
plt.title("Class Distribution")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "class_distribution.png"), dpi=300)
plt.close()

# Pie Chart
plt.figure(figsize=(7,7))
plt.pie(df_counts["Image_Count"], labels=df_counts["Class"], autopct='%1.1f%%', startangle=140, colors=sns.color_palette("viridis", len(df_counts)))
plt.title("Class Proportion (Imbalance)")
plt.savefig(os.path.join(save_dir, "class_proportion.png"), dpi=300)
plt.close()

print("\n✅ Saved class distribution and proportion charts")

# -----------------------------
# Imbalance Analysis
# -----------------------------
majority_class = df_counts.loc[df_counts["Image_Count"].idxmax()]
minority_class = df_counts.loc[df_counts["Image_Count"].idxmin()]

imbalance_ratio = round(majority_class["Image_Count"] / minority_class["Image_Count"], 2) if minority_class["Image_Count"] > 0 else "N/A"

# Chi-square test for balance
chi2, p_value = chisquare(df_counts["Image_Count"])

imbalance_report = {
    "Majority_Class": majority_class["Class"],
    "Majority_Count": int(majority_class["Image_Count"]),
    "Minority_Class": minority_class["Class"],
    "Minority_Count": int(minority_class["Image_Count"]),
    "Imbalance_Ratio": imbalance_ratio,
    "ChiSquare_Statistic": round(chi2, 4),
    "ChiSquare_pValue": round(p_value, 4)
}


pd.DataFrame([imbalance_report]).to_csv(os.path.join(save_dir, "imbalance_report.csv"), index=False)

print("\n📊 Imbalance Analysis:")
for k, v in imbalance_report.items():
    print(f"{k}: {v}")

# -----------------------------
# Image Size Analysis
# -----------------------------
if image_shapes:
    heights = [h for h, w in image_shapes]
    widths = [w for h, w in image_shapes]

    plt.figure(figsize=(8, 6))
    sns.histplot(heights, bins=20, kde=True, color="blue", label="Heights")
    sns.histplot(widths, bins=20, kde=True, color="green", label="Widths")
    plt.legend()
    plt.title("Image Dimension Distribution")
    plt.xlabel("Pixels")
    plt.ylabel("Count")
    plt.savefig(os.path.join(save_dir, "image_dimensions.png"), dpi=300)
    plt.close()

    print("\n✅ Saved image dimension distribution")

# -----------------------------
# Failed Images Report
# -----------------------------
if failed_images:
    with open(os.path.join(save_dir, "failed_images.txt"), "w") as f:
        for path in failed_images:
            f.write(path + "\n")
    print(f"\n⚠️ {len(failed_images)} images failed to load. See failed_images.txt")

# -----------------------------
# Insights
# -----------------------------
print("\n📊 Dataset Insights:")
print(df_counts)

print(f"\nTotal images: {df_counts['Image_Count'].sum()}")
Total_images= df_counts['Image_Count'].sum()
send_message(f"Total dataset {Total_images}")

print(f"\nAverage per class: {df_counts['Image_Count'].mean():.2f}")
Average_per_class= df_counts['Image_Count'].mean()
send_message(f"Avg images per class {Average_per_class}")

print(f"\nClasses: {', '.join(df_counts['Class'])}")


# -----------------------------
# Save class labels and counts to JSON
# -----------------------------
import json

class_info = {
    "total_images": int(df_counts["Image_Count"].sum()),  # total number of images
    "total_classes": len(df_counts),  # <-- add total number of classes
    "labels": df_counts["Class"].tolist(),
    "counts": dict(zip(df_counts["Class"], df_counts["Image_Count"]))

}

with open(LABELS_PATH, "w") as f:
    json.dump(class_info, f, indent=4)

print(f"\n✅ Saved class labels and counts to {LABELS_PATH}")
send_message("Data analysis script execution is finished")
