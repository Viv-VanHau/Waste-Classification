# =========================================================
# ARCHITECTURE 2: DUAL-STAGE SEQUENTIAL CNN
# WORKFLOW: STAGE 1 (CNN 8-CLASS) -> STAGE 2 (CNN 10-CLASS)
# =========================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from matplotlib.colors import LinearSegmentedColormap

# 1. Define Paths & Hardcoded Classes (BULLETPROOF MAPPING)
extract_dir = 'your_path'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Backbone_Stage1_Final_9429.h5'
model_stage2_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
output_dir = 'your_path'

os.makedirs(output_dir, exist_ok=True)

s1_classes = ['battery', 'glass', 'metal', 'organic_waste', 'paper_cardboard', 'plastic', 'textiles', 'trash']

s2_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
              'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
              'plastic_Grade_B', 'textiles', 'trash']

# 2. Smart Directory Finder
test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if 'battery' in dirs and 'glass' in dirs and 'trash' in dirs:
        test_dir = root
        break

print(f"Data directory locked at: {test_dir}")

# 3. Load Models
print("Loading Stage 1 Model (CNN 8-Class)...")
stage1_model = tf.keras.models.load_model(model_stage1_path, compile=False)

print("Loading Stage 2 Model (CNN 10-Class)...")
stage2_model = tf.keras.models.load_model(model_stage2_path, compile=False)

# 4. Prepare Data Pipeline
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    classes=s2_classes,
    class_mode='categorical',
    shuffle=False
)

total_samples = test_generator.samples
filenames = test_generator.filenames

# 5. System Inference Simulation & Routing Logic
print(f"\nStarting inference pipeline on {total_samples} samples...")
print("-" * 110)
print(f"{'ITEM':<10} | {'ACTUAL':<17} | {'S1 PREDICT':<15} | {'S2 PREDICT':<17} | {'FINAL PREDICT':<17} | {'STATUS'}")
print("-" * 110)

y_true = []
y_pred = []
routing_logs = []
stage2_usage_count = 0
rescued_count = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img, label_batch = test_generator[i]
    true_class = s2_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # STAGE 1: PRIMARY SORTING
    pred1 = stage1_model.predict(img, verbose=0)
    c1 = s1_classes[np.argmax(pred1[0])]

    c2 = "N/A"
    route_status = "Bypassed"

    # CONDITIONAL ROUTING TO STAGE 2
    if c1 in ['metal', 'plastic']:
        stage2_usage_count += 1
        pred2 = stage2_model.predict(img, verbose=0)
        c2 = s2_classes[np.argmax(pred2[0])]
        final_decision = c2

        # Track rescues: If Stage 1 said metal/plastic, but Stage 2 outputs something else
        if c2 not in ['metal_Grade_A', 'metal_Grade_B', 'plastic_Grade_A', 'plastic_Grade_B']:
            route_status = "Rescued by S2"
            rescued_count += 1
        else:
            route_status = "Graded by S2"
    else:
        final_decision = c1

    y_pred.append(final_decision)

    # Determine final correctness status
    if final_decision == true_class:
        if route_status == "Rescued by S2":
            final_status = "CORRECT (RESCUED)"
        else:
            final_status = "CORRECT"
    else:
        final_status = "INCORRECT"

    # Record detailed log
    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "Stage1_Predict": c1,
        "Stage2_Predict": c2,
        "Final_Decision": final_decision,
        "Routing_Action": route_status,
        "Status": final_status
    })

    # Print real-time system trace
    print(f"Item {i+1:03d}  | {true_class:<17} | {c1:<15} | {c2:<17} | {final_decision:<17} | {final_status}")

end_time = time.perf_counter()
print("-" * 110)

# 6. Export Routing Log
df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch2_Inference_Routing_Log.csv')
df_log.to_csv(csv_path, index=False)

# 7. Performance & Speed Metrics
total_time = end_time - start_time
avg_time_per_image_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_per_image_ms

print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time : {total_time:.4f} seconds")
print(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms")
print(f"Frames Per Second    : {fps:.2f} FPS")
print(f"Stage 2 Utilization  : {stage2_usage_count}/{total_samples} samples processed by Stage 2")
print(f"Rescued Count        : {rescued_count} classification errors fixed by Stage 2")

# 8. Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=s2_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Architecture_2_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 2: DUAL-STAGE SEQUENTIAL CNN\n")
    f.write("="*60 + "\n")
    f.write(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms\n")
    f.write(f"Frames Per Second    : {fps:.2f} FPS\n")
    f.write(f"Stage 2 Utilization  : {stage2_usage_count}/{total_samples}\n")
    f.write(f"Rescued Count        : {rescued_count}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 9. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=s2_classes)
green_colors = ["#ffffff", "#e8f5e9", "#a5d6a7", "#4caf50", "#2e7d32"]
custom_green_cmap = LinearSegmentedColormap.from_list("custom_green", green_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_green_cmap,
                 xticklabels=s2_classes, yticklabels=s2_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 2: Dual-Stage CNN - Confusion Matrix',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#2e7d32')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch2_Confusion_Matrix_Green.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

# 9.2 F1-Score Bar Chart
f1_scores = f1_score(y_true, y_pred, average=None, labels=s2_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(s2_classes, f1_scores, color='#66bb6a', edgecolor='#2e7d32', linewidth=1.5)

plt.title('Architecture 2: F1-Score per Class', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch2_F1_Scores_Chart.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Process Complete, outputs saved to: {output_dir}")
