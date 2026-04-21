# =================================================================================
# ARCHITECTURE 8: THE ULTIMATE ENSEMBLE (MULTI-MODAL WEIGHTED VOTING)
# WORKFLOW: CNN 10-Class + VLM 10-Class (Base) -> VLM 4-Class (Grading Tie-breaker)
# =================================================================================

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

print("Initializing Evaluation for Architecture 8: The Ultimate Ensemble...")

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
cnn_10class_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
vlm_10class_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
vlm_grading_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_Grading_Model'
output_dir = 'your_path'

os.makedirs(output_dir, exist_ok=True)

final_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
                 'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
                 'plastic_Grade_B', 'textiles', 'trash']
grading_classes = ['metal_Grade_A', 'metal_Grade_B', 'plastic_Grade_A', 'plastic_Grade_B']
grading_indices_in_final = [2, 3, 6, 7] # Mapped indices in final_classes

# 2. Smart Directory Finder
test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if 'battery' in dirs and 'glass' in dirs and 'trash' in dirs:
        test_dir = root
        break

print(f"Data directory locked at: {test_dir}")

# 3. Load All 3 Models
print("Loading Model 1 (CNN 10-Class)...")
cnn_model = tf.keras.models.load_model(cnn_10class_path, compile=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

print("Loading Model 2 (VLM 10-Class)...")
s1_path = vlm_10class_dir
if not os.path.exists(os.path.join(s1_path, 'adapter_config.json')):
    s1_checkpoints = glob.glob(os.path.join(s1_path, "checkpoint-*"))
    s1_path = max(s1_checkpoints, key=os.path.getmtime)
base_10 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=10, ignore_mismatched_sizes=True)
vlm_10 = PeftModel.from_pretrained(base_10, s1_path).to(device).eval()

print("Loading Model 3 (VLM 4-Class Grading Specialist)...")
s2_path = vlm_grading_dir
if not os.path.exists(os.path.join(s2_path, 'adapter_config.json')):
    s2_checkpoints = glob.glob(os.path.join(s2_path, "checkpoint-*"))
    s2_path = max(s2_checkpoints, key=os.path.getmtime)
base_4 = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=4, ignore_mismatched_sizes=True)
vlm_4 = PeftModel.from_pretrained(base_4, s2_path).to(device).eval()

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

# 5. System Inference Simulation
print(f"\nStarting Ultimate Ensemble inference pipeline on {total_samples} samples...")
print("-" * 135)
print(f"{'ITEM':<6} | {'ACTUAL':<17} | {'CNN (Top)':<17} | {'VLM10 (Top)':<17} | {'FINAL ENSEMBLE':<17} | {'STATUS'}")
print("-" * 135)

y_true = []
y_pred = []
routing_logs = []
grading_activations = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = final_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # 5.1 Run CNN 10-Class
    img_cnn = preprocess_input(img_raw.astype(np.float32).copy())
    p_cnn = cnn_model.predict(img_cnn, verbose=0)[0] # Shape (10,)

    # 5.2 Run VLM 10-Class
    img_vlm = img_raw[0].astype(np.uint8)
    inputs = processor(images=img_vlm, return_tensors="pt").to(device)
    with torch.no_grad():
        out_vlm10 = vlm_10(**inputs)
        p_vlm10 = F.softmax(out_vlm10.logits, dim=-1).cpu().numpy()[0] # Shape (10,)

    # 5.3 Base Ensemble (Weighted Voting: 35% CNN, 65% VLM10)
    p_base = 0.35 * p_cnn + 0.65 * p_vlm10
    base_decision_idx = np.argmax(p_base)
    base_decision = final_classes[base_decision_idx]

    final_decision = base_decision
    action = "Base Ensemble"

    # 5.4 Conditional VLM Grading Ensemble
    if base_decision in grading_classes:
        grading_activations += 1
        action = "Ensemble + Grading"

        # Run VLM 4-Class
        with torch.no_grad():
            out_vlm4 = vlm_4(**inputs)
            p_vlm4 = F.softmax(out_vlm4.logits, dim=-1).cpu().numpy()[0] # Shape (4,)

        # Extract the base probabilities for the 4 grading classes and normalize
        p_base_subset = np.array([p_base[idx] for idx in grading_indices_in_final])
        p_base_subset_norm = p_base_subset / (np.sum(p_base_subset) + 1e-9)

        # Grading Ensemble (30% Base, 70% VLM4 Specialist)
        p_grade = 0.30 * p_base_subset_norm + 0.70 * p_vlm4
        final_grade_idx = np.argmax(p_grade)
        final_decision = grading_classes[final_grade_idx]

    y_pred.append(final_decision)

    # Status
    final_status = "CORRECT" if final_decision == true_class else "INCORRECT"

    # Logging
    cnn_top = final_classes[np.argmax(p_cnn)]
    vlm10_top = final_classes[np.argmax(p_vlm10)]

    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "CNN_Predict": cnn_top,
        "VLM10_Predict": vlm10_top,
        "Routing_Action": action,
        "Final_Decision": final_decision,
        "Status": final_status
    })

    print(f"{i+1:03d}    | {true_class:<17} | {cnn_top:<17} | {vlm10_top:<17} | {final_decision:<17} | {final_status}")

end_time = time.perf_counter()
print("-" * 135)

# 6. Export Log
df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch8_Inference_Log.csv')
df_log.to_csv(csv_path, index=False)

# 7. Metrics
total_time = end_time - start_time
avg_time_per_image_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_per_image_ms

print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time : {total_time:.4f} seconds")
print(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms")
print(f"Frames Per Second    : {fps:.2f} FPS")
print(f"Grading Activations  : {grading_activations}/{total_samples} samples sent to Specialist")

# 8. Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=final_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Architecture_8_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 8: THE ULTIMATE ENSEMBLE\n")
    f.write("="*60 + "\n")
    f.write(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms\n")
    f.write(f"Frames Per Second    : {fps:.2f} FPS\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 9. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
platinum_colors = ["#ffffff", "#f5f5f5", "#e0e0e0", "#9e9e9e", "#424242"]
custom_platinum_cmap = LinearSegmentedColormap.from_list("custom_platinum", platinum_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_platinum_cmap,
                 xticklabels=final_classes, yticklabels=final_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 8: Ultimate Ensemble - Confusion Matrix',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#424242')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch8_Confusion_Matrix_Platinum.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

f1_scores = f1_score(y_true, y_pred, average=None, labels=final_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(final_classes, f1_scores, color='#bdbdbd', edgecolor='#424242', linewidth=1.5)

plt.title('Architecture 8: F1-Score per Class', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch8_F1_Scores_Chart.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Process Complete, outputs saved to: {output_dir}")
