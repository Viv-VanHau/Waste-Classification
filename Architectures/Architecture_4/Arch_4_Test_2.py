# ==============================================================================
# ARCHITECTURE 4: PRECISION GRADING PIPELINE
# TEST 2: DOMAIN-CONSTRAINED (HIERARCHICAL LOGIT MASKING) ---- domain expansion iykyk hahahahahahahahahahaha
# ==============================================================================

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

# 1. Path Definitions & Hardcoded Classes
extract_dir = '/content/Base_Test_Data'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Backbone_Stage1_Final_9429.h5'
vlm_grading_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_Grading_Model'
output_dir = '/content/drive/MyDrive/Thesis_Project/Base_Test_Results'

os.makedirs(output_dir, exist_ok=True)

s1_classes = ['battery', 'glass', 'metal', 'organic_waste', 'paper_cardboard', 'plastic', 'textiles', 'trash']
vlm_classes = ['metal_Grade_A', 'metal_Grade_B', 'plastic_Grade_A', 'plastic_Grade_B']
final_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
                 'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
                 'plastic_Grade_B', 'textiles', 'trash']

# Define Domain Constraints for VLM-4 based on CNN base material prediction
# Indices: 0: metal_A, 1: metal_B, 2: plastic_A, 3: plastic_B
domain_masks = {
    'metal': np.array([1.0, 1.0, 0.0, 0.0]),
    'plastic': np.array([0.0, 0.0, 1.0, 1.0])
}

# 2. Automated Directory Discovery
test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if 'battery' in dirs and 'glass' in dirs and 'trash' in dirs:
        test_dir = root
        break

# 3. Load Models
print("Loading Stage 1 Model (CNN 8-Class)...")
cnn_model = tf.keras.models.load_model(model_stage1_path, compile=False)

print("Loading Stage 2 Model (TrashVLM Grading 4-Class)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

model_stage2_path = vlm_grading_dir
if not os.path.exists(os.path.join(vlm_grading_dir, 'adapter_config.json')):
    checkpoints = glob.glob(os.path.join(vlm_grading_dir, "checkpoint-*"))
    if checkpoints:
        model_stage2_path = max(checkpoints, key=os.path.getmtime)
    else:
        raise FileNotFoundError(f"CRITICAL ERROR: No checkpoints found in {vlm_grading_dir}.")

base_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=4, ignore_mismatched_sizes=True
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
print(f"\nStarting inference pipeline on {total_samples} samples...")
print("-" * 125)
print(f"{'INDEX':<7} | {'ACTUAL':<17} | {'S1 PREDICT':<15} | {'ACTION':<20} | {'FINAL PREDICT':<17} | {'STATUS'}")
print("-" * 125)

y_true = []
y_pred = []
routing_logs = []
vlm_usage_count = 0
masked_corrections = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = final_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # STAGE 1: CNN PREDICTION (8 Classes)
    img_cnn = preprocess_input(img_raw.copy().astype(np.float32))
    pred1 = cnn_model.predict(img_cnn, verbose=0)[0]
    c1 = s1_classes[np.argmax(pred1)]

    final_decision = c1
    route_status = "Bypassed VLM"
    c2 = "N/A"

    # CONDITIONAL ROUTING TO VLM GRADING
    if c1 in ['metal', 'plastic']:
        vlm_usage_count += 1
        route_status = "Graded by VLM"

        img_vlm = img_raw[0].astype(np.uint8)
        inputs = processor(images=img_vlm, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = vlm_model(**inputs)
            p_vlm = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

            raw_c2_idx = np.argmax(p_vlm)

            # Apply Hierarchical Domain Mask
            mask = domain_masks[c1]
            p_vlm_masked = p_vlm * mask

            # Prevent division by zero if VLM was absolutely certain about out-of-domain classes
            sum_masked = np.sum(p_vlm_masked)
            if sum_masked > 1e-9:
                p_vlm_masked = p_vlm_masked / sum_masked
            else:
                p_vlm_masked = mask / np.sum(mask) # Fallback to uniform distribution over allowed classes

            masked_c2_idx = np.argmax(p_vlm_masked)
            c2 = vlm_classes[masked_c2_idx]

            # Track if the mask prevented a cross-domain hallucination
            if raw_c2_idx != masked_c2_idx:
                masked_corrections += 1
                route_status = "Masked Correction"

        final_decision = c2

    y_pred.append(final_decision)
    status = "CORRECT" if final_decision == true_class else "INCORRECT"

    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "Stage1_Predict": c1,
        "Stage2_Predict": c2,
        "Routing_Action": route_status,
        "Final_Decision": final_decision,
        "Is_Correct": status == "CORRECT"
    })

    print(f"{i+1:05d}   | {true_class:<17} | {c1:<15} | {route_status:<20} | {final_decision:<17} | {status}")

end_time = time.perf_counter()
print("-" * 125)

# 6. Metrics & Performance Calculation
total_time = end_time - start_time
avg_time_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_ms

df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch4_Test2_Routing_Log.csv')
df_log.to_csv(csv_path, index=False)

# 7. Reporting
print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time    : {total_time:.4f} seconds")
print(f"Average Time / Image    : {avg_time_ms:.2f} ms")
print(f"Frames Per Second       : {fps:.2f} FPS")
print(f"VLM Utilization         : {vlm_usage_count}/{total_samples} samples routed to VLM")
print(f"Masked Corrections      : {masked_corrections} out-of-domain errors prevented")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=final_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Arch4_Test2_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 4: PRECISION GRADING PIPELINE\n")
    f.write("TEST 2: DOMAIN-CONSTRAINED SPECIALIST\n")
    f.write("="*60 + "\n")
    f.write(f"FPS                     : {fps:.2f} FPS\n")
    f.write(f"VLM Utilization         : {vlm_usage_count}/{total_samples}\n")
    f.write(f"Masked Corrections      : {masked_corrections}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 8. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
crimson_colors = ["#ffffff", "#ffebee", "#ffcdd2", "#ef9a9a", "#e53935", "#8e0000"]
custom_crimson_cmap = LinearSegmentedColormap.from_list("custom_crimson", crimson_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_crimson_cmap,
                 xticklabels=final_classes, yticklabels=final_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "sans-serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 4: Domain-Constrained Specialist (Confusion Matrix)',
          fontsize=18, fontweight='bold', pad=25, color='#8e0000')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch4_Test2_Confusion_Matrix.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

f1_scores = f1_score(y_true, y_pred, average=None, labels=final_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(final_classes, f1_scores, color='#e53935', edgecolor='#8e0000', linewidth=1.5)

plt.title('Architecture 4: F1-Score per Class', fontsize=16, fontweight='bold', pad=20, color='#8e0000')
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch4_Test2_F1_Scores.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()
