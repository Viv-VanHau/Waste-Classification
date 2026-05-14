# ==========================================
# ARCHITECTURE 5: END-TO-END VISION TRANSFORMER (ViT)
# COMPONENT: MODEL 5 (TrashVLM 10-Class)
# ==========================================
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

# 1. Define Paths & Hardcoded Classes
extract_dir = '/content/Base_Test_Data'
vlm_10class_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
output_dir = '/content/drive/MyDrive/Thesis_Project/Base_Test_Results_Final'

os.makedirs(output_dir, exist_ok=True)

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

# 3. Load Model (Pure VLM)
print("Loading Model 5 (TrashVLM 10-Class Monolithic)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

# =========================================================================
best_checkpoint_name = "checkpoint-7975"
# =========================================================================

model_path = os.path.join(vlm_10class_dir, best_checkpoint_name)
if not os.path.exists(os.path.join(model_path, 'adapter_config.json')):
    raise FileNotFoundError(f"CRITICAL ERROR: Không tìm thấy config trong {model_path}. Sếp nhớ đổi tên checkpoint ở dòng 43 nha!")

print(f"Force loading specific checkpoint: {model_path}")

base_model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=10,
    ignore_mismatched_sizes=True
)
vlm_model = PeftModel.from_pretrained(base_model, model_path)
vlm_model.to(device)
vlm_model.eval()

# 4. Prepare Data Pipeline (Raw Images for Processor)
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
print(f"\nStarting pure VLM inference pipeline on {total_samples} samples...")
print("-" * 100)
print(f"{'ITEM':<6} | {'ACTUAL':<20} | {'VLM PREDICT':<20} | {'STATUS'}")
print("-" * 100)

y_true = []
y_pred = []
routing_logs = []

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = final_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # Preprocess for VLM
    img_vlm = img_raw[0].astype(np.uint8)
    inputs = processor(images=img_vlm, return_tensors="pt").to(device)

    # VLM Prediction
    with torch.no_grad():
        outputs = vlm_model(**inputs)
        pred_idx = outputs.logits.argmax(-1).item()
        final_decision = final_classes[pred_idx]

    y_pred.append(final_decision)

    # Determine status
    if final_decision == true_class:
        final_status = "CORRECT"
    else:
        final_status = "INCORRECT"

    # Logging
    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "VLM_Predict": final_decision,
        "Status": final_status
    })

    # Terminal trace
    print(f"{i+1:03d}    | {true_class:<20} | {final_decision:<20} | {final_status}")

end_time = time.perf_counter()
print("-" * 100)

# 6. Export Log
df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch5_Inference_Log.csv')
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

# 8. Classification Report
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=final_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Architecture_5_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 5: END-TO-END VISION TRANSFORMER\n")
    f.write("="*60 + "\n")
    f.write(f"Average Time / Image : {avg_time_per_image_ms:.2f} ms\n")
    f.write(f"Frames Per Second    : {fps:.2f} FPS\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 9. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
orange_colors = ["#ffffff", "#fff3e0", "#ffb74d", "#f57c00", "#e65100"]
custom_orange_cmap = LinearSegmentedColormap.from_list("custom_orange", orange_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_orange_cmap,
                 xticklabels=final_classes, yticklabels=final_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 5: End-to-End VLM - Confusion Matrix',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#e65100')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch5_Confusion_Matrix_Orange.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

# 9.2 F1-Score Bar Chart
print("Generating F1-Score Bar Chart...")
f1_scores = f1_score(y_true, y_pred, average=None, labels=final_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(final_classes, f1_scores, color='#ffa726', edgecolor='#e65100', linewidth=1.5)

plt.title('Architecture 5: F1-Score per Class', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch5_F1_Scores_Chart.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

# ==============================================================================
# EXTRA EVALUATION ARCH 5: PURE WASTE CLASSIFICATION (8 CLASSES - IGNORING GRADES)
# ==============================================================================
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

# 2. y_true and y_pred are already Strings in Arch 5, map directly!
y_true_base = [category_mapping[label] for label in y_true]
y_pred_base = [category_mapping[label] for label in y_pred]

# 3. Get unique base classes for the new report (8 classes)
base_classes = sorted(list(set(category_mapping.values())))

# 4. Calculate new metrics
base_accuracy = accuracy_score(y_true_base, y_pred_base)
print(f"Accuracy (Classification Only - No Grading): {base_accuracy * 100:.2f}%\n")

base_report = classification_report(y_true_base, y_pred_base, target_names=base_classes, digits=4)
print(base_report)

# 5. Append this finding to the existing Arch 5 text report
with open(report_path, 'a') as f:
    f.write("\n\n" + "*"*60 + "\n")
    f.write("EXTRA: PURE WASTE CLASSIFICATION (8 CLASSES - NO GRADING)\n")
    f.write("*"*60 + "\n")
    f.write(f"Pure Classification Accuracy: {base_accuracy * 100:.2f}%\n\n")
    f.write(base_report)

# 6. Generate Visualizations for 8-Class Evaluation
print("\nGenerating 8-Class Confusion Matrix for Arch 5...")
cm_base = confusion_matrix(y_true_base, y_pred_base, labels=base_classes)

teal_colors = ["#ffffff", "#e0f2f1", "#80cbc4", "#26a69a", "#00695c"]
custom_teal_cmap = LinearSegmentedColormap.from_list("custom_teal", teal_colors, N=256)

plt.figure(figsize=(12, 9))
sns.set_theme(style="white")

ax = sns.heatmap(cm_base, annot=True, fmt='d', cmap=custom_teal_cmap,
                 xticklabels=base_classes, yticklabels=base_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 5: Pure Classification (8 Classes)',
          fontsize=18, fontweight='bold', pad=25, family='serif', color='#004d40')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_base_save_path = os.path.join(output_dir, 'Arch5_BaseClassification_CM_Teal.png')
plt.savefig(cm_base_save_path, dpi=300, bbox_inches='tight')
plt.close()

# 6.2 F1-Score Bar Chart (8 Classes)
print("Generating 8-Class F1-Score Chart for Arch 5...")
f1_scores_base = f1_score(y_true_base, y_pred_base, average=None, labels=base_classes)

plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")
bars = plt.bar(base_classes, f1_scores_base, color='#26a69a', edgecolor='#00695c', linewidth=1.5)

plt.title('Architecture 5: Pure Classification F1-Score', fontsize=16, fontweight='bold', family='serif', pad=20)
plt.ylabel('F1-Score', fontsize=12, fontweight='bold', family='serif')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_base_save_path = os.path.join(output_dir, 'Arch5_BaseClassification_F1_Chart.png')
plt.savefig(f1_base_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Process Complete. All outputs saved to: {output_dir}")
