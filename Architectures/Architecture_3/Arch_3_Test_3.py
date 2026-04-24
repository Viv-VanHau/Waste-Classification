# ==============================================================================
# ARCHITECTURE 3: CONFIDENCE-ROUTED HYBRID SYSTEM
# TEST 3: TOP-K CONSTRAINED CALIBRATED FUSION (TIE-BREAKER MECHANISM)
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
extract_dir = '/content/Base_Test_Data'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
vlm_base_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_LoRA_Output'
output_dir = '/content/drive/MyDrive/Thesis_Project/Base_Test_Results'

os.makedirs(output_dir, exist_ok=True)
CONFIDENCE_THRESHOLD = 0.85
WEIGHT_CNN = 0.40
WEIGHT_VLM = 0.60
TOP_K_CONSTRAINT = 2

s2_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
              'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
              'plastic_Grade_B', 'textiles', 'trash']

# Calibration Multipliers (From Test 2)
calibration_weights = np.ones(10)
calibration_weights[2] = 1.25  # Boost metal_Grade_A
calibration_weights[6] = 1.50  # Boost plastic_Grade_A

# 2. Automated Directory Discovery
test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if 'battery' in dirs and 'glass' in dirs and 'trash' in dirs:
        test_dir = root
        break

# 3. Model Instantiation
print("Loading Stage 1 Model (CNN 10-Class)...")
cnn_model = tf.keras.models.load_model(model_stage1_path, compile=False)

print("Loading Stage 2 Model (TrashVLM HardCases)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

model_stage2_path = vlm_base_dir
if not os.path.exists(os.path.join(vlm_base_dir, 'adapter_config.json')):
    checkpoints = glob.glob(os.path.join(vlm_base_dir, "checkpoint-*"))
    if checkpoints:
        model_stage2_path = max(checkpoints, key=os.path.getmtime)
    else:
        raise FileNotFoundError(f"No checkpoints found in {vlm_base_dir}.")

base_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=10, ignore_mismatched_sizes=True
)
vlm_model = PeftModel.from_pretrained(base_model, model_stage2_path).to(device).eval()

# 4. Data Pipeline Initialization
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=1,
    classes=s2_classes, class_mode='categorical', shuffle=False
)

total_samples = test_generator.samples
filenames = test_generator.filenames

# 5. Inference Pipeline
print(f"\nStarting inference pipeline on {total_samples} samples...")
print(f"Routing Threshold: Confidence < {CONFIDENCE_THRESHOLD}")
print("-" * 130)
print(f"{'INDEX':<7} | {'ACTUAL':<17} | {'CNN TOP-1':<17} | {'CONF':<6} | {'ACTION':<20} | {'FINAL PREDICT':<17} | {'STATUS'}")
print("-" * 130)

y_true = []
y_pred = []
routing_logs = []

vlm_calls = 0
fusion_rescues = 0
constrained_corrections = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = s2_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # Stage 1: CNN Inference
    img_cnn = preprocess_input(img_raw.copy().astype(np.float32))
    p_cnn = cnn_model.predict(img_cnn, verbose=0)[0]

    cnn_idx = np.argmax(p_cnn)
    cnn_conf = np.max(p_cnn)
    c1 = s2_classes[cnn_idx]

    final_decision = c1
    route_action = "Fast-Path (CNN)"

    # Stage 2: Top-K Constrained Fusion
    if cnn_conf < CONFIDENCE_THRESHOLD:
        vlm_calls += 1
        route_action = "Constrained Fusion"

        img_vlm = img_raw[0].astype(np.uint8)
        inputs = processor(images=img_vlm, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = vlm_model(**inputs)
            p_vlm = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

        # Extract Top-K indices from CNN
        top_k_indices = np.argsort(p_cnn)[-TOP_K_CONSTRAINT:]

        # Create mask and apply to VLM
        mask = np.zeros(10)
        mask[top_k_indices] = 1.0

        p_vlm_masked = p_vlm * mask
        p_vlm_sum = np.sum(p_vlm_masked)

        # Normalize masked VLM probabilities (fallback to uniform if sum is 0 to avoid division by zero)
        if p_vlm_sum > 1e-9:
            p_vlm_masked = p_vlm_masked / p_vlm_sum
        else:
            p_vlm_masked[top_k_indices] = 1.0 / TOP_K_CONSTRAINT

        # Soft Fusion
        p_fused = (WEIGHT_CNN * p_cnn) + (WEIGHT_VLM * p_vlm_masked)

        # Apply Calibration Multiplier
        p_calibrated = p_fused * calibration_weights
        final_idx = np.argmax(p_calibrated)
        final_decision = s2_classes[final_idx]

        # Tracking metrics
        unconstrained_vlm_idx = np.argmax(p_vlm)
        if unconstrained_vlm_idx not in top_k_indices and final_decision == true_class:
            constrained_corrections += 1

        if final_decision == true_class and c1 != true_class:
            fusion_rescues += 1

    y_pred.append(final_decision)
    status = "CORRECT" if final_decision == true_class else "INCORRECT"

    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "CNN_Predict": c1,
        "CNN_Confidence": float(cnn_conf),
        "Routing_Action": route_action,
        "Final_Decision": final_decision,
        "Is_Correct": status == "CORRECT"
    })

    print(f"{i+1:05d}   | {true_class:<17} | {c1:<17} | {cnn_conf:.2f} | {route_action:<20} | {final_decision:<17} | {status}")

end_time = time.perf_counter()
print("-" * 130)

# 6. Metrics & Performance Calculation
total_time = end_time - start_time
avg_time_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_ms

df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch3_Test3_Routing_Log.csv')
df_log.to_csv(csv_path, index=False)

# 7. Reporting
print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time    : {total_time:.4f} seconds")
print(f"Average Time / Image    : {avg_time_ms:.2f} ms")
print(f"Frames Per Second       : {fps:.2f} FPS")
print(f"VLM Utilization         : {vlm_calls}/{total_samples} samples processed by VLM")
print(f"Constrained Corrections : {constrained_corrections} out-of-domain VLM errors prevented")
print(f"Total Rescues           : {fusion_rescues} errors prevented in Stage 2")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=s2_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Arch3_Test3_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 3: CONFIDENCE-ROUTED HYBRID SYSTEM\n")
    f.write("TEST 3: TOP-K CONSTRAINED CALIBRATED FUSION\n")
    f.write("="*60 + "\n")
    f.write(f"FPS                     : {fps:.2f} FPS\n")
    f.write(f"VLM Utilization         : {vlm_calls}/{total_samples}\n")
    f.write(f"Constrained Corrections : {constrained_corrections}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 8. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=s2_classes)
indigo_colors = ["#ffffff", "#e8eaf6", "#c5cae9", "#7986cb", "#3f51b5", "#1a237e"]
custom_indigo_cmap = LinearSegmentedColormap.from_list("custom_indigo", indigo_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_indigo_cmap,
                 xticklabels=s2_classes, yticklabels=s2_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "sans-serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 3: Top-K Constrained Fusion (Confusion Matrix)',
          fontsize=18, fontweight='bold', pad=25, color='#1a237e')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch3_Test3_Confusion_Matrix.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

f1_scores = f1_score(y_true, y_pred, average=None, labels=s2_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(s2_classes, f1_scores, color='#5c6bc0', edgecolor='#1a237e', linewidth=1.5)

plt.title('Architecture 3: F1-Score per Class', fontsize=16, fontweight='bold', pad=20, color='#1a237e')
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch3_Test3_F1_Scores.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()
