# ==============================================================================
# ARCHITECTURE 7: SEQUENTIAL HYBRID VERIFICATION (SOFT FUSION)
# TEST 1: OPTIMAL ASYMMETRIC FUSION (20% CNN / 80% VLM)
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from matplotlib.colors import LinearSegmentedColormap

# 1. Path Definitions & Configurations
extract_dir = '/content/Base_Test_Data'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
vlm_10class_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
output_dir = '/content/drive/MyDrive/Thesis_Project/Arch7_Test1_Results'

os.makedirs(output_dir, exist_ok=True)

# OPTIMAL WEIGHTS (Derived from Sensitivity Analysis)
WEIGHT_CNN = 0.20
WEIGHT_VLM = 0.80

final_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
                 'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
                 'plastic_Grade_B', 'textiles', 'trash']

# 2. Smart Directory Discovery
def find_data_path(base_path):
    if not os.path.exists(base_path):
        return None
    for root, dirs, files in os.walk(base_path):
        if 'battery' in dirs and 'glass' in dirs:
            return root
    return base_path

test_dir = find_data_path(extract_dir)

if not test_dir or not os.path.exists(test_dir):
    raise ValueError(f"CRITICAL ERROR: Data folder not found at {extract_dir}.")

print(f"Data directory locked at: {test_dir}")

# 3. Load Models
print("Loading Model 1 (CNN 10-Class)...")
cnn_model = tf.keras.models.load_model(model_stage1_path, compile=False)

print("Loading Model 2 (TrashVLM 10-Class)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)

# =========================================================================
# CHECKPOINT CHO VLM 10-CLASS (7975)
# =========================================================================
best_checkpoint_arch7_test1 = "checkpoint-7975"
model_stage2_path = os.path.join(vlm_10class_dir, best_checkpoint_arch7_test1)

if not os.path.exists(os.path.join(model_stage2_path, 'adapter_config.json')):
    raise FileNotFoundError(f"CRITICAL ERROR: Không tìm thấy checkpoint {model_stage2_path}.")
print(f"Force loading Arch 7 VLM checkpoint: {model_stage2_path}")

base_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k", num_labels=10, ignore_mismatched_sizes=True
)
vlm_model = PeftModel.from_pretrained(base_model, model_stage2_path).to(device).eval()

# 4. Data Pipeline
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(224, 224), batch_size=1,
    classes=final_classes, class_mode='categorical', shuffle=False
)

total_samples = test_generator.samples
filenames = test_generator.filenames

# 5. Parallel Inference Pipeline
print(f"\nStarting Asymmetric Soft Fusion pipeline on {total_samples} samples...")
print(f"Strategy: {WEIGHT_CNN*100}% CNN + {WEIGHT_VLM*100}% VLM")
print("-" * 140)
print(f"{'INDEX':<7} | {'ACTUAL':<17} | {'CNN PRED':<17} | {'VLM PRED':<17} | {'FINAL FUSED':<17} | {'STATUS'}")
print("-" * 140)

y_true = []
y_pred = []
fusion_logs = []

fusion_rescues = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_idx = np.argmax(label_batch[0])
    true_class = final_classes[true_idx]
    y_true.append(true_class)
    filename = filenames[i]

    # Model 1: CNN Predict
    img_cnn = preprocess_input(img_raw.astype(np.float32).copy())
    p_cnn = cnn_model.predict(img_cnn, verbose=0)[0]
    cnn_pred = final_classes[np.argmax(p_cnn)]

    # Model 2: VLM Predict
    img_vlm = img_raw[0].astype(np.uint8)
    inputs = processor(images=img_vlm, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = vlm_model(**inputs)
        p_vlm = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        vlm_pred = final_classes[np.argmax(p_vlm)]

    # MEGA SOFT FUSION
    p_fused = (WEIGHT_CNN * p_cnn) + (WEIGHT_VLM * p_vlm)
    final_idx = np.argmax(p_fused)
    final_decision = final_classes[final_idx]

    # VÁ LỖI Ở ĐÂY: Thêm final_decision vào mảng y_pred
    y_pred.append(final_decision)

    # Track Fusion Rescues
    if final_decision == true_class and vlm_pred != true_class:
        fusion_rescues += 1
        status_tag = "FUSION RESCUED VLM"
    else:
        status_tag = "STANDARD"

    status = "CORRECT" if final_decision == true_class else "INCORRECT"

    fusion_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "CNN_Predict": cnn_pred,
        "VLM_Predict": vlm_pred,
        "Final_Decision": final_decision,
        "Status_Tag": status_tag,
        "Is_Correct": status == "CORRECT"
    })

    print(f"{i+1:05d}   | {true_class:<17} | {cnn_pred:<17} | {vlm_pred:<17} | {final_decision:<17} | {status}")

end_time = time.perf_counter()
print("-" * 140)

# 6. Metrics Calculation
total_time = end_time - start_time
avg_time_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_ms

df_log = pd.DataFrame(fusion_logs)
csv_path = os.path.join(output_dir, 'Arch7_Test1_Fusion_Log.csv')
df_log.to_csv(csv_path, index=False)

# 7. Reporting
print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time    : {total_time:.4f} seconds")
print(f"Average Time / Image    : {avg_time_ms:.2f} ms")
print(f"Frames Per Second       : {fps:.2f} FPS")
print(f"System Utilization      : 100% (Every input processed by both models)")
print(f"Fusion Rescues          : {fusion_rescues} cases where CNN regularized VLM successfully")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=final_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Arch7_Test1_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 7: MEGA SOFT FUSION\n")
    f.write(f"TEST 1: ASYMMETRIC FUSION (CNN: {WEIGHT_CNN} | VLM: {WEIGHT_VLM})\n")
    f.write("="*60 + "\n")
    f.write(f"FPS                     : {fps:.2f} FPS\n")
    f.write(f"Fusion Rescues          : {fusion_rescues}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 8. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
emerald_colors = ["#ffffff", "#e8f5e9", "#a5d6a7", "#66bb6a", "#2e7d32", "#1b5e20"]
custom_emerald_cmap = LinearSegmentedColormap.from_list("custom_emerald", emerald_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_emerald_cmap,
                 xticklabels=final_classes, yticklabels=final_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "sans-serif"},
                 linewidths=0, cbar=True)

plt.title(f'Architecture 7: Asymmetric Fusion {int(WEIGHT_CNN*100)}/{int(WEIGHT_VLM*100)} (Confusion Matrix)',
          fontsize=18, fontweight='bold', pad=25, color='#1b5e20')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch7_Test1_Confusion_Matrix.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

f1_scores = f1_score(y_true, y_pred, average=None, labels=final_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(final_classes, f1_scores, color='#66bb6a', edgecolor='#1b5e20', linewidth=1.5)

plt.title('Architecture 7: F1-Score per Class', fontsize=16, fontweight='bold', pad=20, color='#1b5e20')
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch7_Test1_F1_Scores.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# EXTRA EVALUATION ARCH 7 TEST 1: PURE WASTE CLASSIFICATION (8 CLASSES - IGNORING GRADES)
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

purple_colors = ["#ffffff", "#f3e5f5", "#ce93d8", "#ab47bc", "#7b1fa2"]
custom_purple_cmap = LinearSegmentedColormap.from_list("custom_purple", purple_colors, N=256)

plt.figure(figsize=(12, 9))
sns.set_theme(style="white")

ax = sns.heatmap(cm_base, annot=True, fmt='d', cmap=custom_purple_cmap,
                 xticklabels=base_classes, yticklabels=base_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 7 (Test 1): Pure Classification (8 Classes)',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#4a148c')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_base_save_path = os.path.join(output_dir, 'Arch7_Test1_BaseClassification_CM_Purple.png')
plt.savefig(cm_base_save_path, dpi=300, bbox_inches='tight')
plt.close()

# 8-Class F1-Score Chart
print("Generating 8-Class F1-Score Chart for Arch 7 Test 1...")
f1_scores_base = f1_score(y_true_base, y_pred_base, average=None, labels=base_classes)

plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")
bars = plt.bar(base_classes, f1_scores_base, color='#ab47bc', edgecolor='#7b1fa2', linewidth=1.5)

plt.title('Architecture 7 (Test 1): Pure Classification F1-Score', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_base_save_path = os.path.join(output_dir, 'Arch7_Test1_BaseClassification_F1_Chart.png')
plt.savefig(f1_base_save_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Process Complete. All analytical outputs saved to: {output_dir}"
