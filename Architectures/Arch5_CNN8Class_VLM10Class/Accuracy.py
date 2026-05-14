# ==============================================================================
# ARCHITECTURE 6: CROSS-ARCHITECTURAL HIERARCHICAL PIPELINE
# TEST 3: SELECTIVE SOFT FUSION & CALIBRATION
# ==============================================================================
import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from transformers import AutoImageProcessor, ViTForImageClassification
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from matplotlib.colors import LinearSegmentedColormap

# 1. Path Definitions & Configurations
extract_dir = '/content/Base_Test_Data/Base_Test'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Backbone_Stage1_Final_9429.h5'
vlm_10class_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
output_dir = '/content/drive/MyDrive/Thesis_Project/Base_Test_Results_Final'

os.makedirs(output_dir, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.90
WEIGHT_CNN = 0.40
WEIGHT_VLM = 0.60

s1_classes = ['battery', 'glass', 'metal', 'organic_waste', 'paper_cardboard', 'plastic', 'textiles', 'trash']
final_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
                 'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
                 'plastic_Grade_B', 'textiles', 'trash']

# Mapping S1 (8 classes) to S2 (10 classes) for expanding probabilities
s1_to_s2_mapping = {
    0: [0], 1: [1], 2: [2, 3], 3: [4], 4: [5], 5: [6, 7], 6: [8], 7: [9]
}

calibration_weights = np.ones(10)
calibration_weights[2] = 1.25  # Boost metal_Grade_A
calibration_weights[6] = 1.50  # Boost plastic_Grade_A

# 2. Automated Directory Discovery
test_dir = extract_dir
if not os.path.exists(test_dir):
    print(f"Warning: {test_dir} not found. Searching...")
    for root, dirs, files in os.walk('/content/Base_Test_Data'):
        if 'battery' in dirs and 'glass' and 'trash' in dirs:
            test_dir = root
            break

print(f"Data directory located at: {test_dir}")

# 3. Load Models
print("Loading Stage 1 Model (CNN 8-Class)...")
cnn_model = tf.keras.models.load_model(model_stage1_path, compile=False)

print("Loading Stage 2 Model (TrashVLM 10-Class)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)

# =========================================================================
best_checkpoint_arch6 = "checkpoint-7975"
# =========================================================================

model_stage2_path = os.path.join(vlm_10class_dir, best_checkpoint_arch6)

if not os.path.exists(model_stage2_path):
    raise FileNotFoundError(f"CRITICAL ERROR: Không tìm thấy {model_stage2_path}! Sếp check lại Drive nha.")

print(f"Force loading Arch 6 VLM checkpoint: {model_stage2_path}")

base_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=10, ignore_mismatched_sizes=True
)
vlm_model = PeftModel.from_pretrained(base_model, model_stage2_path).to(device).eval()

# 4. Data Pipeline Initialization
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=1,
    classes=final_classes, class_mode='categorical', shuffle=False
)

total_samples = test_generator.samples
filenames = test_generator.filenames

# 5. Inference Pipeline
print(f"\nStarting hierarchical inference pipeline on {total_samples} samples...")
print("-" * 135)
print(f"{'INDEX':<7} | {'ACTUAL':<17} | {'S1 PREDICT':<15} | {'CONF':<6} | {'ACTION':<25} | {'FINAL PREDICT':<17} | {'STATUS'}")
print("-" * 135)

y_true = []
y_pred = []
routing_logs = []

stage2_calls = 0
fusion_rescues = 0
grading_rescues = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = final_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # Stage 1: CNN Inference
    img_cnn = preprocess_input(img_raw.astype(np.float32).copy())
    p_cnn = cnn_model.predict(img_cnn, verbose=0)[0]

    s1_conf = np.max(p_cnn)
    c1 = s1_classes[np.argmax(p_cnn)]

    final_decision = c1
    route_action = "Fast-Path (Bypass)"
    c2 = "N/A"

    # Routing Logic
    requires_grading = c1 in ['metal', 'plastic']
    requires_fallback = (not requires_grading) and (s1_conf < CONFIDENCE_THRESHOLD)

    if requires_grading or requires_fallback:
        stage2_calls += 1

        img_vlm = img_raw[0].astype(np.uint8)
        inputs = processor(images=img_vlm, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = vlm_model(**inputs)
            p_vlm = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        if requires_grading:
            # Grading Route: VLM takes full control, apply calibration
            route_action = "VLM Grading + Calib"
            p_calibrated = p_vlm * calibration_weights
            final_idx = np.argmax(p_calibrated)
            final_decision = final_classes[final_idx]

            # Tracking
            if true_class in final_decision and c1 not in true_class:
                grading_rescues += 1

        elif requires_fallback:
            # Fallback Route: Soft Fusion between CNN and VLM, then calibration
            route_action = "Soft Fusion + Calib"

            # Expand CNN probabilities to 10 classes
            p_cnn_expanded = np.zeros(10)
            for old_idx, new_indices in s1_to_s2_mapping.items():
                val = p_cnn[old_idx] / len(new_indices)
                for new_idx in new_indices:
                    p_cnn_expanded[new_idx] = val

            p_fused = (WEIGHT_CNN * p_cnn_expanded) + (WEIGHT_VLM * p_vlm)
            p_calibrated = p_fused * calibration_weights

            final_idx = np.argmax(p_calibrated)
            final_decision = final_classes[final_idx]

            # Tracking
            if final_decision == true_class and c1 != true_class:
                fusion_rescues += 1

    y_pred.append(final_decision)
    status = "CORRECT" if final_decision == true_class else "INCORRECT"

    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "S1_Predict": c1,
        "S1_Confidence": float(s1_conf),
        "Routing_Action": route_action,
        "Final_Decision": final_decision,
        "Is_Correct": status == "CORRECT"
    })

    print(f"{i+1:05d}   | {true_class:<17} | {c1:<15} | {s1_conf:.2f} | {route_action:<25} | {final_decision:<17} | {status}")

end_time = time.perf_counter()
print("-" * 135)

# 6. Metrics & Performance Calculation
total_time = end_time - start_time
avg_time_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_ms

df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch6_Test3_Routing_Log.csv')
df_log.to_csv(csv_path, index=False)

# 7. Reporting
print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time    : {total_time:.4f} seconds")
print(f"Average Time / Image    : {avg_time_ms:.2f} ms")
print(f"Frames Per Second       : {fps:.2f} FPS")
print(f"Stage 2 Utilization     : {stage2_calls}/{total_samples} samples routed to VLM")
print(f"Grading Rescues         : {grading_rescues} material errors fixed during grading")
print(f"Fusion Rescues          : {fusion_rescues} errors fixed during fallback fusion")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=final_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Arch6_Test3_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 6: CROSS-ARCHITECTURAL HIERARCHICAL PIPELINE\n")
    f.write("TEST 3: SELECTIVE SOFT FUSION & CALIBRATION\n")
    f.write("="*60 + "\n")
    f.write(f"FPS                     : {fps:.2f} FPS\n")
    f.write(f"Stage 2 Utilization     : {stage2_calls}/{total_samples}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 8. Visualizations 
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
navy_colors = ["#ffffff", "#e8eaf6", "#c5cae9", "#7986cb", "#3f51b5", "#1a237e"]
custom_navy_cmap = LinearSegmentedColormap.from_list("custom_navy", navy_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_navy_cmap,
                 xticklabels=final_classes, yticklabels=final_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "sans-serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 6: Selective Fusion (Confusion Matrix)',
          fontsize=18, fontweight='bold', pad=25, color='#1a237e')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch6_Test3_Confusion_Matrix.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

f1_scores = f1_score(y_true, y_pred, average=None, labels=final_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(final_classes, f1_scores, color='#5c6bc0', edgecolor='#1a237e', linewidth=1.5)

plt.title('Architecture 6: F1-Score per Class', fontsize=16, fontweight='bold', pad=20, color='#1a237e')
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch6_Test3_F1_Scores.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# EXTRA EVALUATION ARCH 6 TEST 3: PURE WASTE CLASSIFICATION (8 CLASSES - NO GRADING)
# ==============================================================================
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import os

print("\n" + "="*80 + "\nEXTRA EVALUATION: PURE WASTE CLASSIFICATION (NO GRADING)\n" + "="*80)

output_dir = '/content/drive/MyDrive/Thesis_Project/Base_Test_Results_Final'
report_path = os.path.join(output_dir, 'Arch6_Test3_Classification_Report.txt')

category_mapping = {
    'battery': 'battery', 'glass': 'glass', 'metal_Grade_A': 'metal', 'metal_Grade_B': 'metal',
    'organic_waste': 'organic_waste', 'paper_cardboard': 'paper_cardboard',
    'plastic_Grade_A': 'plastic', 'plastic_Grade_B': 'plastic', 'textiles': 'textiles', 'trash': 'trash'
}

y_true_base = [category_mapping[label] for label in y_true]
y_pred_base = [category_mapping[label] for label in y_pred]
base_classes = sorted(list(set(category_mapping.values())))

base_accuracy = accuracy_score(y_true_base, y_pred_base)
base_report = classification_report(y_true_base, y_pred_base, labels=base_classes, target_names=base_classes, digits=4)

print(f"8-Class Accuracy: {base_accuracy * 100:.2f}%\n\n{base_report}")

with open(report_path, 'a') as f:
    f.write("\n\n" + "*"*60 + "\nEXTRA: PURE WASTE CLASSIFICATION (8 CLASSES - NO GRADING)\n" + "*"*60 + "\n")
    f.write(f"8-Class Accuracy: {base_accuracy * 100:.2f}%\n\n{base_report}")

# Visualization
cm_base = confusion_matrix(y_true_base, y_pred_base, labels=base_classes)
plt.figure(figsize=(12, 9))
sns.heatmap(cm_base, annot=True, fmt='d', cmap=LinearSegmentedColormap.from_list("navy", ["#ffffff", "#e8eaf6", "#3f51b5", "#1a237e"]),
            xticklabels=base_classes, yticklabels=base_classes, annot_kws={"weight": "bold"})
plt.title('Arch 6 Test 3: Pure Classification (8 Classes)', fontweight='bold', color='#1a237e')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Arch6_Test3_BaseClassification_CM.png'), dpi=300)
plt.show()

print(f"Process Complete. Outputs saved to: {output_dir}")
