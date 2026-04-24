# ============================================================
# ARCHITECTURE 3: CONFIDENCE-ROUTED HYBRID SYSTEM
# WORKFLOW: STAGE 1 (CNN 10-CLASS) -> STAGE 2 (VLM HARDCASES)
# THRESHOLD: CONFIDENCE < 0.85
# ============================================================

import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from transformers import AutoImageProcessor, ViTForImageClassification
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from matplotlib.colors import LinearSegmentedColormap

# 0. Memory Management: Prevent TF from hogging all VRAM
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. Define Paths & Hardcoded Classes
extract_dir = '/content/Base_Test_Data'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
vlm_base_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_LoRA_Output'
output_dir = '/content/drive/MyDrive/Thesis_Project/Base_Test_Results/Architecture_3'

os.makedirs(output_dir, exist_ok=True)
CONFIDENCE_THRESHOLD = 0.85

s2_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
              'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
              'plastic_Grade_B', 'textiles', 'trash']

# 2. Smart Directory Finder
test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if 'battery' in dirs and 'glass' in dirs and 'trash' in dirs:
        test_dir = root
        break

print(f"Data directory located at: {test_dir}")

# 3. Load Models
print("Loading Stage 1 Model (CNN 10-Class)...")
cnn_model = tf.keras.models.load_model(model_stage1_path, compile=False)

print("Loading Stage 2 Model (TrashVLM HardCases)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

model_stage2_path = vlm_base_dir
if not os.path.exists(os.path.join(vlm_base_dir, 'adapter_config.json')):
    print("adapter_config.json not found in root. Searching for latest checkpoint...")
    checkpoints = glob.glob(os.path.join(vlm_base_dir, "checkpoint-*"))
    if checkpoints:
        model_stage2_path = max(checkpoints, key=os.path.getmtime)
        print(f"Found checkpoint! Routing VLM path to: {model_stage2_path}")
    else:
        raise FileNotFoundError(f"CRITICAL ERROR: No adapter_config.json or checkpoints found in {vlm_base_dir}. Please check your Google Drive path!")

base_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10,
    ignore_mismatched_sizes=True
)
vlm_model = PeftModel.from_pretrained(base_model, model_stage2_path)
vlm_model.to(device)
vlm_model.eval()

# 4. Prepare Data Pipeline
test_datagen = ImageDataGenerator()
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
print(f"Routing Threshold: Confidence < {CONFIDENCE_THRESHOLD}")
print("-" * 125)
print(f"{'ITEM':<6} | {'ACTUAL':<17} | {'S1 PREDICT':<17} | {'CONF':<6} | {'ACTION':<17} | {'FINAL PREDICT':<17} | {'STATUS'}")
print("-" * 125)

y_true = []
y_pred = []
routing_logs = []
vlm_usage_count = 0
rescued_count = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = s2_classes[np.argmax(label_batch[0])]

    y_true.append(true_class)

    filename = filenames[i]

    # Preprocess specifically for CNN
    img_cnn = preprocess_input(img_raw.copy())

    # STAGE 1: CNN PREDICTION
    pred1 = cnn_model.predict(img_cnn, verbose=0)
    conf_score = np.max(pred1[0])
    c1 = s2_classes[np.argmax(pred1[0])]

    final_decision = c1
    route_status = "Trusted S1"
    c2 = "N/A"

    # CONDITIONAL ROUTING TO VLM
    if conf_score < CONFIDENCE_THRESHOLD:
        vlm_usage_count += 1
        route_status = "Routed to VLM"

        # Preprocess for VLM
        img_vlm = img_raw[0].astype(np.uint8)
        inputs = processor(images=img_vlm, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = vlm_model(**inputs)
            c2_idx = outputs.logits.argmax(-1).item()
            c2 = s2_classes[c2_idx]

        final_decision = c2

        # Track rescues: VLM predicted correctly while CNN was wrong
        if c2 == true_class and c1 != true_class:
            route_status = "Rescued by VLM"
            rescued_count += 1

    y_pred.append(final_decision)

    # Determine status
    if final_decision == true_class:
        if route_status == "Rescued by VLM":
            final_status = "CORRECT (RESCUED)"
        else:
            final_status = "CORRECT"
    else:
        final_status = "INCORRECT"

    # Logging
    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "Stage1_Predict": c1,
        "S1_Confidence": float(conf_score),
        "Stage2_Predict": c2,
        "Routing_Action": route_status,
        "Final_Decision": final_decision,
        "Status": final_status
    })

    print(f"{i+1:03d}    | {true_class:<17} | {c1:<17} | {conf_score:.2f} | {route_status:<17} | {final_decision:<17} | {final_status}")

end_time = time.perf_counter()
print("-" * 125)

# 6. Export Routing Log
df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch3_Inference_Routing_Log.csv')
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
print(f"VLM Utilization      : {vlm_usage_count}/{total_samples} samples routed to VLM")
print(f"VLM Rescued Count    : {rescued_count} classification errors fixed by VLM")

# 8. Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=s2_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Architecture_3_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 3: CONFIDENCE-ROUTED HYBRID SYSTEM\n")
    f.write(f"Threshold: {CONFIDENCE_THRESHOLD}\n")
    f.write("="*60 + "\n")
    f.write(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms\n")
    f.write(f"Frames Per Second    : {fps:.2f} FPS\n")
    f.write(f"VLM Utilization      : {vlm_usage_count}/{total_samples}\n")
    f.write(f"VLM Rescued Count    : {rescued_count}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 9. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=s2_classes)
purple_colors = ["#ffffff", "#f3e5f5", "#ce93d8", "#ab47bc", "#7b1fa2"]
custom_purple_cmap = LinearSegmentedColormap.from_list("custom_purple", purple_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_purple_cmap,
                 xticklabels=s2_classes, yticklabels=s2_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 3: Confidence-Routed Hybrid - Confusion Matrix',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#7b1fa2')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch3_Confusion_Matrix_Purple.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

# 9.2 F1-Score Bar Chart
f1_scores = f1_score(y_true, y_pred, average=None, labels=s2_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(s2_classes, f1_scores, color='#ab47bc', edgecolor='#7b1fa2', linewidth=1.5)

plt.title('Architecture 3: F1-Score per Class', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch3_F1_Scores_Chart.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Process Complete, outputs saved to: {output_dir}")

# =================================================================================
# EXTRA EVALUATION ARCH 3: PURE WASTE CLASSIFICATION (8 CLASSES - IGNORING GRADES)
# =================================================================================

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

print("\n" + "="*80)
print("EXTRA EVALUATION: PURE WASTE CLASSIFICATION (NO GRADING)")
print("="*80)

# 1. Define the mapping logic (Merging grades into parent materials)
category_mapping = {
    'battery': 'battery',
    'glass': 'glass',
    'metal_Grade_A': 'metal',     # Merge A
    'metal_Grade_B': 'metal',     # Merge B
    'organic_waste': 'organic_waste',
    'paper_cardboard': 'paper_cardboard',
    'plastic_Grade_A': 'plastic', # Merge A
    'plastic_Grade_B': 'plastic', # Merge B
    'textiles': 'textiles',
    'trash': 'trash'
}

# 2. y_true and y_pred are already Strings in Arch 3, map directly!
y_true_base = [category_mapping[label] for label in y_true]
y_pred_base = [category_mapping[label] for label in y_pred]

# 3. Get unique base classes for the new report (8 classes)
base_classes = sorted(list(set(category_mapping.values())))

# 4. Calculate new metrics
base_accuracy = accuracy_score(y_true_base, y_pred_base)
print(f"Accuracy (Classification Only - No Grading): {base_accuracy * 100:.2f}%\n")

base_report = classification_report(y_true_base, y_pred_base, target_names=base_classes, digits=4)
print(base_report)

# 5. Append this finding to the existing Arch 3 text report
with open(report_path, 'a') as f:
    f.write("\n\n" + "*"*60 + "\n")
    f.write("EXTRA: PURE WASTE CLASSIFICATION (8 CLASSES - NO GRADING)\n")
    f.write("*"*60 + "\n")
    f.write(f"Pure Classification Accuracy: {base_accuracy * 100:.2f}%\n\n")
    f.write(base_report)

# 6. Generate Visualizations for 8-Class Evaluation

# 6.1 Confusion Matrix
cm_base = confusion_matrix(y_true_base, y_pred_base, labels=base_classes)

teal_colors = ["#ffffff", "#e0f2f1", "#80cbc4", "#26a69a", "#00695c"]
custom_teal_cmap = LinearSegmentedColormap.from_list("custom_teal", teal_colors, N=256)

plt.figure(figsize=(12, 9))
sns.set_theme(style="white")

ax = sns.heatmap(cm_base, annot=True, fmt='d', cmap=custom_teal_cmap,
                 xticklabels=base_classes, yticklabels=base_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 3: Pure Classification (8 Classes)',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#004d40')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_base_save_path = os.path.join(output_dir, 'Arch3_BaseClassification_CM_Teal.png')
plt.savefig(cm_base_save_path, dpi=300, bbox_inches='tight')
plt.close()

# 6.2 F1-Score Bar Chart (8 Classes)
print("Generating 8-Class F1-Score Chart for Arch 3...")
f1_scores_base = f1_score(y_true_base, y_pred_base, average=None, labels=base_classes)

plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")
bars = plt.bar(base_classes, f1_scores_base, color='#26a69a', edgecolor='#00695c', linewidth=1.5)

plt.title('Architecture 3: Pure Classification F1-Score', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()

f1_base_save_path = os.path.join(output_dir, 'Arch3_BaseClassification_F1_Chart.png')
plt.savefig(f1_base_save_path, dpi=300, bbox_inches='tight')
plt.close()

