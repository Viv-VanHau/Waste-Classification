# ====================================================================
# ARCHITECTURE 6: CROSS-ARCHITECTURAL HIERARCHICAL PIPELINE
# WORKFLOW: STAGE 1 (CNN 8-CLASS) -> STAGE 2 (VLM 10-CLASS MONOLITHIC)
# MECHANISM: CONDITIONAL ROUTING & ERROR CORRECTION
# ====================================================================

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

# 0. Memory Management
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 1. Define Paths
extract_dir = 'your_path'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Backbone_Stage1_Final_9429.h5'
vlm_10class_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
output_dir = 'your_path'

os.makedirs(output_dir, exist_ok=True)

s1_classes = ['battery', 'glass', 'metal', 'organic_waste', 'paper_cardboard', 'plastic', 'textiles', 'trash']
final_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
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
cnn_model = tf.keras.models.load_model(model_stage1_path, compile=False)

print("Loading Stage 2 Model (TrashVLM 10-Class)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# SMART CHECKPOINT FINDER
model_stage2_path = vlm_10class_dir
if not os.path.exists(os.path.join(vlm_10class_dir, 'adapter_config.json')):
    print("adapter_config.json not found in root. Searching for latest checkpoint...")
    checkpoints = glob.glob(os.path.join(vlm_10class_dir, "checkpoint-*"))
    if checkpoints:
        model_stage2_path = max(checkpoints, key=os.path.getmtime)
        print(f"Found checkpoint! Routing VLM path to: {model_stage2_path}")
    else:
        raise FileNotFoundError(f"CRITICAL ERROR: No adapter_config.json or checkpoints found in {vlm_10class_dir}.")

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
    classes=final_classes,
    class_mode='categorical',
    shuffle=False
)

total_samples = test_generator.samples
filenames = test_generator.filenames

# 5. System Inference Simulation & Routing Logic
print(f"\nStarting hierarchical inference pipeline on {total_samples} samples...")
print("-" * 120)
print(f"{'ITEM':<6} | {'ACTUAL':<17} | {'S1 PREDICT':<15} | {'ACTION':<20} | {'FINAL PREDICT':<17} | {'STATUS'}")
print("-" * 120)

y_true = []
y_pred = []
routing_logs = []
stage2_usage_count = 0
rescued_count = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = final_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # Preprocess specifically for CNN
    img_cnn = preprocess_input(img_raw.astype(np.float32).copy())

    # STAGE 1: CNN PREDICTION
    pred1 = cnn_model.predict(img_cnn, verbose=0)
    c1 = s1_classes[np.argmax(pred1[0])]

    final_decision = c1
    route_status = "Bypassed S2"
    c2 = "N/A"

    # CONDITIONAL ROUTING TO VLM 10-CLASS
    if c1 in ['metal', 'plastic']:
        stage2_usage_count += 1

        # Preprocess for VLM
        img_vlm = img_raw[0].astype(np.uint8)
        inputs = processor(images=img_vlm, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = vlm_model(**inputs)
            c2_idx = outputs.logits.argmax(-1).item()
            c2 = final_classes[c2_idx]

        final_decision = c2

        # Track rescues: If Stage 1 said metal/plastic, but VLM 10-Class says it's something else
        if c2 not in ['metal_Grade_A', 'metal_Grade_B', 'plastic_Grade_A', 'plastic_Grade_B']:
            route_status = "Rescued by VLM"
            rescued_count += 1
        else:
            route_status = "Graded by VLM"

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
        "Stage2_Predict": c2,
        "Routing_Action": route_status,
        "Final_Decision": final_decision,
        "Status": final_status
    })

    # Trace
    print(f"{i+1:03d}    | {true_class:<17} | {c1:<15} | {route_status:<20} | {final_decision:<17} | {final_status}")

end_time = time.perf_counter()
print("-" * 120)

# 6. Export Routing Log
df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch6_Inference_Routing_Log.csv')
df_log.to_csv(csv_path, index=False)

# 7. Performance Metrics
total_time = end_time - start_time
avg_time_per_image_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_per_image_ms

print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time : {total_time:.4f} seconds")
print(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms")
print(f"Frames Per Second    : {fps:.2f} FPS")
print(f"Stage 2 Utilization  : {stage2_usage_count}/{total_samples} samples routed to VLM")
print(f"Rescued Count        : {rescued_count} classification errors fixed by VLM")

# 8. Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=final_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Architecture_6_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 6: CROSS-ARCHITECTURAL HIERARCHICAL PIPELINE\n")
    f.write("="*60 + "\n")
    f.write(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms\n")
    f.write(f"Frames Per Second    : {fps:.2f} FPS\n")
    f.write(f"Stage 2 Utilization  : {stage2_usage_count}/{total_samples}\n")
    f.write(f"Rescued Count        : {rescued_count}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 9. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
teal_colors = ["#ffffff", "#e0f2f1", "#80cbc4", "#26a69a", "#00695c"]
custom_teal_cmap = LinearSegmentedColormap.from_list("custom_teal", teal_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_teal_cmap,
                 xticklabels=final_classes, yticklabels=final_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 6: Hierarchical Hybrid - Confusion Matrix',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#00695c')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch6_Confusion_Matrix_Teal.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

# F1-Score Bar Chart
f1_scores = f1_score(y_true, y_pred, average=None, labels=final_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(final_classes, f1_scores, color='#4db6ac', edgecolor='#00695c', linewidth=1.5)

plt.title('Architecture 6: F1-Score per Class', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch6_F1_Scores_Chart.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Process Complete, outputs saved to: {output_dir}")
