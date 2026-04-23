# ================================================
# ARCHITECTURE 1: SINGLE-STREAM CNN BASELINE
# COMPONENT: MODEL 2 (CNN 10-CLASS CLASSIFICATION)
# ================================================

import os
import time
import zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from matplotlib.colors import LinearSegmentedColormap

# 1. Define Paths
zip_path = '/content/drive/MyDrive/Thesis_Project/Base_Test.zip'
extract_dir = '/content/Base_Test_Data'
model_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
output_dir = '/content/drive/MyDrive/Thesis_Project/Base_Test_Results'

os.makedirs(output_dir, exist_ok=True)

# 2. Extract Data
if not os.path.exists(extract_dir):
    print(f"Extracting test dataset to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if len(dirs) == 10:
        test_dir = root
        break

print(f"Data located at: {test_dir}")

# 3. Load Model
print("Loading Model 2 (MobileNetV2 - 10 Classes)...")
baseline_model = tf.keras.models.load_model(model_path, compile=False)

# 4. Prepare Data Pipeline
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1, # Batch size 1 to simulate real-world sequential frame processing
    class_mode='categorical',
    shuffle=False
)

y_true = test_generator.classes
class_labels = list(test_generator.class_indices.keys())
total_samples = test_generator.samples

# 5. System Inference Simulation & Runtime Measurement
print(f"Starting inference pipeline on {total_samples} samples...")
y_pred = []

start_time = time.perf_counter()

for i in range(total_samples):
    img, _ = test_generator[i]
    prediction = baseline_model.predict(img, verbose=0)
    y_pred.append(np.argmax(prediction[0]))

end_time = time.perf_counter()

# 6. Performance & Speed Metrics
total_time = end_time - start_time
avg_time_per_image_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_per_image_ms

print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time : {total_time:.4f} seconds")
print(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms")
print(f"Frames Per Second    : {fps:.2f} FPS")

# 7. Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=class_labels, digits=4)
print(report_str)

# Save text report for thesis documentation
report_path = os.path.join(output_dir, 'Architecture_1_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 1: SINGLE-STREAM CNN BASELINE\n")
    f.write("="*60 + "\n")
    f.write(f"Average Time / Image: {avg_time_per_image_ms:.2f} ms\n")
    f.write(f"Frames Per Second: {fps:.2f} FPS\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 8. Visualizations

# 8.1 Confusion Matrix

cm = confusion_matrix(y_true, y_pred)

colors = ["#ffffff", "#e3f2fd", "#90caf9", "#42a5f5", "#1565c0"]
custom_blue_cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")

ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_blue_cmap,
                 xticklabels=class_labels, yticklabels=class_labels,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 1: Single-Stream CNN - Confusion Matrix',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#0d47a1')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch1_Confusion_Matrix_Blue.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

# 8.2 F1-Score Bar Chart

f1_scores = f1_score(y_true, y_pred, average=None)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(class_labels, f1_scores, color='#42a5f5', edgecolor='#1565c0', linewidth=1.5)

plt.title('Architecture 1: F1-Score per Class', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch1_F1_Scores_Chart.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Process Complete, successfully saved to: {output_dir}")
