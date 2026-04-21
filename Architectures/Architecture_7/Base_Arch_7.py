# =================================================================
# ARCHITECTURE 7: DOUBLE-EXPERT VLM PIPELINE (THE FINAL BOSS)
# WORKFLOW: STAGE 1 (VLM 10-CLASS) -> STAGE 2 (VLM GRADING 4-CLASS)
# MECHANISM: VLM GATEKEEPER + VLM SPECIALIST
# =================================================================

import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from transformers import AutoImageProcessor, ViTForImageClassification
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from matplotlib.colors import LinearSegmentedColormap

# 1. Define Paths
extract_dir = 'your_path'
vlm_10class_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
vlm_grading_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_Grading_Model'
output_dir = 'your_path'

os.makedirs(output_dir, exist_ok=True)

# Define label spaces
final_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
                 'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
                 'plastic_Grade_B', 'textiles', 'trash']
grading_classes = ['metal_Grade_A', 'metal_Grade_B', 'plastic_Grade_A', 'plastic_Grade_B']

# 2. Smart Directory Finder
test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if 'battery' in dirs and 'glass' in dirs and 'trash' in dirs:
        test_dir = root
        break

print(f"Data directory locked at: {test_dir}")

# 3. Load Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# LOAD STAGE 1: VLM 10-CLASS
print("Loading Stage 1 Model (TrashVLM 10-Class Gatekeeper)...")
s1_path = vlm_10class_dir
if not os.path.exists(os.path.join(s1_path, 'adapter_config.json')):
    s1_checkpoints = glob.glob(os.path.join(s1_path, "checkpoint-*"))
    s1_path = max(s1_checkpoints, key=os.path.getmtime)

base_model_1 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=10, ignore_mismatched_sizes=True)
vlm_stage1 = PeftModel.from_pretrained(base_model_1, s1_path)
vlm_stage1.to(device)
vlm_stage1.eval()

# LOAD STAGE 2: VLM GRADING
print("Loading Stage 2 Model (TrashVLM Grading Specialist)...")
s2_path = vlm_grading_dir
if not os.path.exists(os.path.join(s2_path, 'adapter_config.json')):
    s2_checkpoints = glob.glob(os.path.join(s2_path, "checkpoint-*"))
    s2_path = max(s2_checkpoints, key=os.path.getmtime)

base_model_2 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=4, ignore_mismatched_sizes=True)
vlm_stage2 = PeftModel.from_pretrained(base_model_2, s2_path)
vlm_stage2.to(device)
vlm_stage2.eval()

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
print(f"\nStarting Double-Expert inference pipeline on {total_samples} samples...")
print("-" * 125)
print(f"{'ITEM':<6} | {'ACTUAL':<17} | {'S1 (10-CLS)':<17} | {'ACTION':<22} | {'FINAL PREDICT':<17} | {'STATUS'}")
print("-" * 125)

y_true = []
y_pred = []
routing_logs = []
stage2_usage_count = 0
re_graded_count = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = final_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # Preprocess Image for VLM
    img_vlm = img_raw[0].astype(np.uint8)
    inputs = processor(images=img_vlm, return_tensors="pt").to(device)

    # STAGE 1: VLM 10-CLASS PREDICTION
    with torch.no_grad():
        out1 = vlm_stage1(**inputs)
        c1_idx = out1.logits.argmax(-1).item()
        c1 = final_classes[c1_idx]

    final_decision = c1
    route_status = "Trusted S1"
    c2 = "N/A"

    # CONDITIONAL ROUTING TO STAGE 2 (If S1 says it's Metal or Plastic)
    if c1 in grading_classes:
        stage2_usage_count += 1

        # STAGE 2: VLM GRADING SPECIALIST PREDICTION
        with torch.no_grad():
            out2 = vlm_stage2(**inputs)
            c2_idx = out2.logits.argmax(-1).item()
            c2 = grading_classes[c2_idx]

        final_decision = c2

        # Track changes: Did the Specialist disagree with the Gatekeeper's grading?
        if c2 != c1:
            route_status = "Re-graded by Specialist"
            re_graded_count += 1
        else:
            route_status = "Confirmed by Specialist"

    y_pred.append(final_decision)

    # Determine status
    if final_decision == true_class:
        if route_status == "Re-graded by Specialist":
            final_status = "CORRECT (RE-GRADED)"
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

    # Trace log
    print(f"{i+1:03d}    | {true_class:<17} | {c1:<17} | {route_status:<22} | {final_decision:<17} | {final_status}")

end_time = time.perf_counter()
print("-" * 125)

# 6. Export Routing Log
df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch7_Inference_Routing_Log.csv')
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
print(f"Stage 2 Utilization  : {stage2_usage_count}/{total_samples} samples sent to Specialist")
print(f"Re-graded Count      : {re_graded_count} times the Specialist overrode Stage 1 grading")

# 8. Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=final_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Architecture_7_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 7: DOUBLE-EXPERT VLM PIPELINE\n")
    f.write("="*60 + "\n")
    f.write(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms\n")
    f.write(f"Frames Per Second    : {fps:.2f} FPS\n")
    f.write(f"Stage 2 Utilization  : {stage2_usage_count}/{total_samples}\n")
    f.write(f"Re-graded Count      : {re_graded_count}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 9. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
gold_colors = ["#ffffff", "#fff8e1", "#ffe082", "#ffca28", "#ff8f00"]
custom_gold_cmap = LinearSegmentedColormap.from_list("custom_gold", gold_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_gold_cmap,
                 xticklabels=final_classes, yticklabels=final_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 7: Double-Expert VLM - Confusion Matrix',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#ff8f00')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch7_Confusion_Matrix_Gold.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

f1_scores = f1_score(y_true, y_pred, average=None, labels=final_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(final_classes, f1_scores, color='#ffca28', edgecolor='#ff8f00', linewidth=1.5)

plt.title('Architecture 7: F1-Score per Class', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch7_F1_Scores_Chart.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Process Complete, outputs saved to: {output_dir}")
