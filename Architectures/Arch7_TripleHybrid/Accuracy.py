# ==============================================================================
# ARCHITECTURE 9: THE ACADEMIC CEILING (ULTIMATE ENSEMBLE)
# TEST 2: KỊCH TRẦN BAY PHẤP PHỚI (OPTIMAL WEIGHTS + EXTREME THRESHOLDING)
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
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from matplotlib.colors import LinearSegmentedColormap

# 1. Define Paths
extract_dir = '/content/Base_Test_Data'
cnn_10class_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
vlm_10class_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
vlm_grading_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_Grading_Model'
output_dir = '/content/drive/MyDrive/Thesis_Project/Arch9_Test2_Results'

os.makedirs(output_dir, exist_ok=True)

# --- THE ULTIMATE PARAMETERS (SQUEEZING EVERY PERCENTAGE) ---
WEIGHT_CNN = 0.20
WEIGHT_VLM10 = 0.80
METAL_A_MIN_REQ = 0.15   # Hạ chuẩn Metal A xuống 15% để tăng Recall
PLASTIC_A_MIN_REQ = 0.15 # Hạ chuẩn Plastic A xuống 15% để tăng Recall

final_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
                 'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
                 'plastic_Grade_B', 'textiles', 'trash']
grading_classes = ['metal_Grade_A', 'metal_Grade_B', 'plastic_Grade_A', 'plastic_Grade_B']
grading_indices = [2, 3, 6, 7]

# 2. Smart Directory Finder
def find_data_path(base_path):
    if not os.path.exists(base_path): return None
    for root, dirs, files in os.walk(base_path):
        if 'battery' in dirs and 'glass' in dirs: return root
    return base_path

test_dir = find_data_path(extract_dir)
if not test_dir: raise ValueError("CRITICAL ERROR: Data folder not found.")

# 3. Load Models
print("Loading Model 1 (CNN 10-Class)...")
cnn_model = tf.keras.models.load_model(cnn_10class_path, compile=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)

# SỬA CHỖ NÀY: Khóa checkpoint
def load_vlm_fixed(model_dir, checkpoint_name, num_labels):
    path = os.path.join(model_dir, checkpoint_name)
    if not os.path.exists(os.path.join(path, 'adapter_config.json')):
        print(f"Warning: Checkpoint {checkpoint_name} not found. Falling back to latest...")
        ckpts = glob.glob(os.path.join(model_dir, "checkpoint-*"))
        path = max(ckpts, key=os.path.getmtime)

    print(f"Force loading: {path}")
    base = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=num_labels, ignore_mismatched_sizes=True)
    return PeftModel.from_pretrained(base, path).to(device).eval()

print("Loading Model 2 (VLM 10-Class - FIXED 7975)...")
vlm_10 = load_vlm_fixed(vlm_10class_dir, "checkpoint-7975", 10)
print("Loading Model 3 (VLM 4-Class Specialist)...")
vlm_4 = load_vlm_fixed(vlm_grading_dir, "checkpoint-XXX", 4) # Thay XXX bằng checkpoint xịn nhất

# 4. Data Pipeline
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(224, 224), batch_size=1, classes=final_classes, class_mode='categorical', shuffle=False)
total_samples = test_generator.samples
filenames = test_generator.filenames

# 5. System Inference Simulation
print(f"\nStarting 'Kịch Trần' Pipeline on {total_samples} samples...")
print("-" * 140)
print(f"{'INDEX':<7} | {'ACTUAL':<17} | {'BASE PRED':<17} | {'ACTION':<28} | {'FINAL PREDICT':<17} | {'STATUS'}")
print("-" * 140)

y_true = []
y_pred = []
logs = []
shifted_rescues = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = final_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # --- BASE ENSEMBLE (OPTIMAL 20/80) ---
    img_cnn = preprocess_input(img_raw.astype(np.float32).copy())
    p_cnn = cnn_model.predict(img_cnn, verbose=0)[0]

    img_vlm = img_raw[0].astype(np.uint8)
    inputs = processor(images=img_vlm, return_tensors="pt").to(device)
    with torch.no_grad():
        p_vlm10 = F.softmax(vlm_10(**inputs).logits, dim=-1).cpu().numpy()[0]

    p_base = (WEIGHT_CNN * p_cnn) + (WEIGHT_VLM10 * p_vlm10)
    base_idx = np.argmax(p_base)
    base_pred = final_classes[base_idx]

    final_decision = base_pred
    action = "Base Optimal (20/80)"

    # --- EXTREME SPECIALIST GRADING ---
    if base_idx in grading_indices:
        with torch.no_grad():
            p_s2 = F.softmax(vlm_4(**inputs).logits, dim=-1).cpu().numpy()[0]

        raw_c2 = grading_classes[np.argmax(p_s2)]

        # THRESHOLD MOVING LOGIC
        if p_s2[0] + p_s2[1] > p_s2[2] + p_s2[3]:
            # Nhóm Metal
            if p_s2[0] >= METAL_A_MIN_REQ:
                final_decision = 'metal_Grade_A'
            else:
                final_decision = 'metal_Grade_B'
        else:
            # Nhóm Plastic
            if p_s2[2] >= PLASTIC_A_MIN_REQ:
                final_decision = 'plastic_Grade_A'
            else:
                final_decision = 'plastic_Grade_B'

        if final_decision != raw_c2:
            action = "VLM4 + Extreme Shift (15%)"
            if final_decision == true_class:
                shifted_rescues += 1
        else:
            action = "VLM4 Standard"

    y_pred.append(final_decision)
    status = "CORRECT" if final_decision == true_class else "INCORRECT"

    logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "Final_Decision": final_decision,
        "Is_Correct": status == "CORRECT"
    })

    print(f"{i+1:05d}   | {true_class:<17} | {base_pred:<17} | {action:<28} | {final_decision:<17} | {status}")

end_time = time.perf_counter()
print("-" * 140)

# 6. Export Log
df_log = pd.DataFrame(logs)
df_log.to_csv(os.path.join(output_dir, 'Arch9_Test2_Log.csv'), index=False)

# 7. Metrics
fps = 1000 / (((end_time - start_time) / total_samples) * 1000)

print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Frames Per Second    : {fps:.2f} FPS")
print(f"Shifted Rescues      : {shifted_rescues} Grade A cases recovered by Extreme Gating!")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=final_classes, digits=4)
print(report_str)

report_txt_path = os.path.join(output_dir, 'Arch9_Test2_Report.txt')
with open(report_txt_path, 'w') as f:
    f.write("ARCHITECTURE 9: TEST 2 (THE ABSOLUTE PEAK - PROJECT SUPERNOVA)\n")
    f.write(f"PARAM: Base(20/80) + Extreme VLM4 Threshold (15%)\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 8. Visualizations
cm = confusion_matrix(y_true, y_pred, labels=final_classes)
galactic_colors = ["#ffffff", "#f3e5f5", "#ce93d8", "#ab47bc", "#7b1fa2", "#4a148c"]
cmap = LinearSegmentedColormap.from_list("custom", galactic_colors, N=256)

plt.figure(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=final_classes, yticklabels=final_classes, annot_kws={"size": 12, "weight": "bold"}, cbar=True)
plt.title('Architecture 9: Galactic Supernova (Confusion Matrix)', fontsize=18, fontweight='bold', pad=25, color='#4a148c')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Arch9_Test2_CM.png'), dpi=300, bbox_inches='tight')
plt.close()

f1_scores = f1_score(y_true, y_pred, average=None, labels=final_classes)
plt.figure(figsize=(12, 6))
bars = plt.bar(final_classes, f1_scores, color='#ab47bc', edgecolor='#4a148c', linewidth=1.5)
plt.title('Architecture 9: F1-Score per Class (Peak Limit)', fontsize=16, fontweight='bold', pad=20, color='#4a148c')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}', ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Arch9_Test2_F1.png'), dpi=300, bbox_inches='tight')
plt.close()


# ==============================================================================
# EXTRA EVALUATION ARCH 9 TEST 2: PURE WASTE CLASSIFICATION (8 CLASSES)
# ==============================================================================
print("\n" + "="*80)
print("EXTRA EVALUATION: PURE WASTE CLASSIFICATION (NO GRADING)")
print("="*80)

category_mapping = {
    'battery': 'battery', 'glass': 'glass', 'metal_Grade_A': 'metal', 'metal_Grade_B': 'metal',
    'organic_waste': 'organic_waste', 'paper_cardboard': 'paper_cardboard', 'plastic_Grade_A': 'plastic',
    'plastic_Grade_B': 'plastic', 'textiles': 'textiles', 'trash': 'trash'
}

y_true_8 = [category_mapping[label] for label in y_true]
y_pred_8 = [category_mapping[label] for label in y_pred]
base_classes = sorted(list(set(category_mapping.values())))

acc_8 = accuracy_score(y_true_8, y_pred_8)
print(f"Accuracy (8-Class Peak): {acc_8 * 100:.2f}%\n")

report_8 = classification_report(y_true_8, y_pred_8, target_names=base_classes, digits=4)
print(report_8)

with open(report_txt_path, 'a') as f:
    f.write("\n\n" + "*"*60 + "\n")
    f.write("EXTRA: PURE WASTE CLASSIFICATION (8 CLASSES - PEAK PERFORMANCE)\n")
    f.write("*"*60 + "\n")
    f.write(f"Pure Classification Accuracy: {acc_8 * 100:.2f}%\n\n")
    f.write(report_8)

cm_base = confusion_matrix(y_true_8, y_pred_8, labels=base_classes)
teal_colors = ["#ffffff", "#e0f2f1", "#80cbc4", "#26a69a", "#00695c"]
custom_teal_cmap = LinearSegmentedColormap.from_list("custom_teal", teal_colors, N=256)

plt.figure(figsize=(12, 9))
sns.set_theme(style="white")
ax = sns.heatmap(cm_base, annot=True, fmt='d', cmap=custom_teal_cmap,
                 xticklabels=base_classes, yticklabels=base_classes,
                 annot_kws={"size": 12, "weight": "bold"}, linewidths=0, cbar=True)
plt.title('Architecture 9 (Supernova): Pure Classification (8 Classes)', fontsize=18, fontweight='bold', pad=25, color='#004d40')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'Arch9_Test2_Base_CM.png'), dpi=300, bbox_inches='tight')
plt.close()
