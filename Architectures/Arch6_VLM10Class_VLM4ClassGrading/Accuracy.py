# ==============================================================================
# ARCHITECTURE 8: DOUBLE-EXPERT VLM PIPELINE (THE FINAL BOSS)
# TEST 1: CONFIDENCE-GATED SPECIALIST ROUTING (THRESHOLD = 0.90)
# ==============================================================================
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from matplotlib.colors import LinearSegmentedColormap

# 1. Define Paths & Configurations
extract_dir = '/content/Base_Test_Data'
vlm_10class_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
vlm_grading_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_Grading_Model'
output_dir = '/content/drive/MyDrive/Thesis_Project/Arch8_Test1_Results'

os.makedirs(output_dir, exist_ok=True)

# THE GOLDEN THRESHOLD (From Sensitivity Analysis)
ROUTING_THRESHOLD = 0.90

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
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)

# =========================================================================
# CHECKPOINT CHO VLM 10-CLASS LÀ 7975
# =========================================================================
best_checkpoint_arch8_test1 = "checkpoint-7975"
s1_path = os.path.join(vlm_10class_dir, best_checkpoint_arch8_test1)

if not os.path.exists(os.path.join(s1_path, 'adapter_config.json')):
    raise FileNotFoundError(f"CRITICAL ERROR: Không tìm thấy checkpoint {s1_path}.")
print(f"Force loading Arch 8 Test 1 VLM checkpoint: {s1_path}")

base_model_1 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=10, ignore_mismatched_sizes=True)
vlm_stage1 = PeftModel.from_pretrained(base_model_1, s1_path).to(device).eval()

# --- LOAD STAGE 2: VLM GRADING (AUTO) ---
print("Loading Stage 2 Model (TrashVLM Grading Specialist)...")
s2_path = vlm_grading_dir
if not os.path.exists(os.path.join(s2_path, 'adapter_config.json')):
    s2_checkpoints = glob.glob(os.path.join(s2_path, "checkpoint-*"))
    s2_path = max(s2_checkpoints, key=os.path.getmtime)

base_model_2 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=4, ignore_mismatched_sizes=True)
vlm_stage2 = PeftModel.from_pretrained(base_model_2, s2_path).to(device).eval()

# 4. Prepare Data Pipeline
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=1,
    classes=final_classes, class_mode='categorical', shuffle=False
)

total_samples = test_generator.samples
filenames = test_generator.filenames

# 5. System Inference Simulation & Routing Logic
print(f"\nStarting Confidence-Gated Double-Expert pipeline on {total_samples} samples...")
print(f"Routing Rule: Call Specialist ONLY IF Stage 1 predicts Grading AND Conf < {ROUTING_THRESHOLD}")
print("-" * 135)
print(f"{'INDEX':<7} | {'ACTUAL':<17} | {'S1 PRED':<17} | {'CONF':<6} | {'ACTION':<22} | {'FINAL PREDICT':<17} | {'STATUS'}")
print("-" * 135)

y_true = []
y_pred = []
routing_logs = []

stage2_usage_count = 0
s1_fast_path_count = 0

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
        p_s1 = torch.nn.functional.softmax(out1.logits, dim=-1).cpu().numpy()[0]
        c1_idx = np.argmax(p_s1)
        c1 = final_classes[c1_idx]
        conf1 = np.max(p_s1)

    final_decision = c1
    route_action = "M1 Fast-Path (Confident)"
    c2 = "N/A"

    # CONDITIONAL ROUTING LOGIC
    requires_grading = c1 in grading_classes
    requires_specialist = requires_grading and (conf1 < ROUTING_THRESHOLD)

    if requires_specialist:
        stage2_usage_count += 1
        route_action = "Routed to Specialist"

        # STAGE 2: VLM GRADING SPECIALIST PREDICTION
        with torch.no_grad():
            out2 = vlm_stage2(**inputs)
            c2_idx = out2.logits.argmax(-1).item()
            c2 = grading_classes[c2_idx]

        final_decision = c2
    elif requires_grading:
        s1_fast_path_count += 1
        route_action = "M1 Fast-Path (Grading)"

    y_pred.append(final_decision)
    status = "CORRECT" if final_decision == true_class else "INCORRECT"

    # Logging
    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "Stage1_Predict": c1,
        "Stage1_Conf": float(conf1),
        "Stage2_Predict": c2,
        "Routing_Action": route_action,
        "Final_Decision": final_decision,
        "Is_Correct": status == "CORRECT"
    })

    print(f"{i+1:05d}   | {true_class:<17} | {c1:<17} | {conf1:.2f} | {route_action:<22} | {final_decision:<17} | {status}")

end_time = time.perf_counter()
print("-" * 135)

# 6. Export Routing Log
df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch8_Test1_Routing_Log.csv')
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
print(f"M1 Auto-Graded       : {s1_fast_path_count} grading cases handled by Gatekeeper directly")

# 8. Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=final_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Arch8_Test1_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 8: DOUBLE-EXPERT VLM PIPELINE\n")
    f.write(f"TEST 1: CONFIDENCE GATING (THRESHOLD = {ROUTING_THRESHOLD})\n")
    f.write("="*60 + "\n")
    f.write(f"Frames Per Second    : {fps:.2f} FPS\n")
    f.write(f"Stage 2 Utilization  : {stage2_usage_count}/{total_samples}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 9. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
gold_colors = ["#ffffff", "#fff8e1", "#ffe082", "#ffca28", "#ff8f00", "#e65100"]
custom_gold_cmap = LinearSegmentedColormap.from_list("custom_gold", gold_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_gold_cmap,
                 xticklabels=final_classes, yticklabels=final_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "sans-serif"},
                 linewidths=0, cbar=True)

plt.title(f'Arch 8: Gated Routing (T={ROUTING_THRESHOLD}) - Confusion Matrix',
          fontsize=18, fontweight='bold', pad=25, color='#e65100')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Arch8_Test1_Confusion_Matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

f1_scores = f1_score(y_true, y_pred, average=None, labels=final_classes)
plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(final_classes, f1_scores, color='#ffca28', edgecolor='#ff8f00', linewidth=1.5)

plt.title('Architecture 8: F1-Score per Class', fontsize=16, fontweight='bold', pad=20, color='#e65100')
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Arch8_Test1_F1_Scores.png'), dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# EXTRA EVALUATION ARCH 8 TEST 1: PURE WASTE CLASSIFICATION (8 CLASSES - IGNORING GRADES)
# ==============================================================================
print("\n" + "="*80)
print("EXTRA EVALUATION: PURE WASTE CLASSIFICATION (NO GRADING)")
print("="*80)

category_mapping = {
    'battery': 'battery',
    'glass': 'glass',
    'metal_Grade_A': 'metal',
    'metal_Grade_B': 'metal',
    'organic_waste': 'organic_waste',
    'paper_cardboard': 'paper_cardboard',
    'plastic_Grade_A': 'plastic',
    'plastic_Grade_B': 'plastic',
    'textiles': 'textiles',
    'trash': 'trash'
}

y_true_base = [category_mapping[label] for label in y_true]
y_pred_base = [category_mapping[label] for label in y_pred]

base_classes = sorted(list(set(category_mapping.values())))

base_accuracy = accuracy_score(y_true_base, y_pred_base)
print(f"Accuracy (Classification Only - No Grading): {base_accuracy * 100:.2f}%\n")

base_report = classification_report(y_true_base, y_pred_base, target_names=base_classes, digits=4)
print(base_report)

with open(report_path, 'a') as f:
    f.write("\n\n" + "*"*60 + "\n")
    f.write("EXTRA: PURE WASTE CLASSIFICATION (8 CLASSES - NO GRADING)\n")
    f.write("*"*60 + "\n")
    f.write(f"Pure Classification Accuracy: {base_accuracy * 100:.2f}%\n\n")
    f.write(base_report)

# 8-Class Visualizations
cm_base = confusion_matrix(y_true_base, y_pred_base, labels=base_classes)

teal_colors = ["#ffffff", "#e0f2f1", "#80cbc4", "#26a69a", "#00695c"]
custom_teal_cmap = LinearSegmentedColormap.from_list("custom_teal", teal_colors, N=256)

plt.figure(figsize=(12, 9))
sns.set_theme(style="white")

ax = sns.heatmap(cm_base, annot=True, fmt='d', cmap=custom_teal_cmap,
                 xticklabels=base_classes, yticklabels=base_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 8 (Test 1): Pure Classification (8 Classes)',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#004d40')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_base_save_path = os.path.join(output_dir, 'Arch8_Test1_BaseClassification_CM_Teal.png')
plt.savefig(cm_base_save_path, dpi=300, bbox_inches='tight')
plt.close()

# 8-Class F1-Score Chart
f1_scores_base = f1_score(y_true_base, y_pred_base, average=None, labels=base_classes)

plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")
bars = plt.bar(base_classes, f1_scores_base, color='#26a69a', edgecolor='#00695c', linewidth=1.5)

plt.title('Architecture 8 (Test 1): Pure Classification F1-Score', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_base_save_path = os.path.join(output_dir, 'Arch8_Test1_BaseClassification_F1_Chart.png')
plt.savefig(f1_base_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Process Complete. All analytical outputs saved to: {output_dir}")
