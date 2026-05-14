# ==============================================================================
# ARCHITECTURE 7: MEGA SOFT FUSION PIPELINE
# SENSITIVITY ANALYSIS: OPTIMAL WEIGHT GRID SEARCH
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
from sklearn.metrics import accuracy_score, f1_score

print("Initializing Sensitivity Analysis for Architecture 7 (Weight Sweep & 8-Class Combo)...")

# 1. Path Definitions & Configurations
extract_dir = '/content/Base_Test_Data'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
vlm_10class_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_10_Classes_Model'
output_dir = '/content/drive/MyDrive/Thesis_Project/Arch7_Sensitivity_Results'

os.makedirs(output_dir, exist_ok=True)

final_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
                 'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
                 'plastic_Grade_B', 'textiles', 'trash']

# Weights to test for CNN (VLM weight will be 1.0 - CNN_weight)
cnn_weights_to_test = np.linspace(0.0, 1.0, 21) # 0.0, 0.05, 0.10, ..., 1.0

category_mapping = {
    'battery': 'battery', 'glass': 'glass', 'metal_Grade_A': 'metal', 'metal_Grade_B': 'metal',
    'organic_waste': 'organic_waste', 'paper_cardboard': 'paper_cardboard', 'plastic_Grade_A': 'plastic',
    'plastic_Grade_B': 'plastic', 'textiles': 'textiles', 'trash': 'trash'
}

# 2. Automated Directory Discovery
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
print("Loading Stage 1 Model (CNN 10-Class)...")
cnn_model = tf.keras.models.load_model(model_stage1_path, compile=False)

print("Loading Stage 2 Model (TrashVLM 10-Class)...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k", use_fast=True)

# =========================================================================
# GẮN CỨNG CHECKPOINT CHO VLM 10-CLASS LÀ 7975
# =========================================================================
best_checkpoint_arch7_sens = "checkpoint-7975"
model_stage2_path = os.path.join(vlm_10class_dir, best_checkpoint_arch7_sens)

if not os.path.exists(os.path.join(model_stage2_path, 'adapter_config.json')):
    raise FileNotFoundError(f"CRITICAL ERROR: Không tìm thấy checkpoint {model_stage2_path}.")
print(f"Force loading Arch 7 Sensitivity VLM checkpoint: {model_stage2_path}")

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

# 5. Data Caching Phase (Run models once to extract RAW PROBABILITIES)
print(f"\n[PHASE 1] Pre-computing raw probabilities for {total_samples} samples...")
y_true_cache = []
p_cnn_cache = []
p_vlm_cache = []

start_cache_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    y_true_cache.append(np.argmax(label_batch[0]))

    # CNN Probabilities
    img_cnn = preprocess_input(img_raw.astype(np.float32).copy())
    p_cnn = cnn_model.predict(img_cnn, verbose=0)[0]
    p_cnn_cache.append(p_cnn)

    # VLM Probabilities
    img_vlm = img_raw[0].astype(np.uint8)
    inputs = processor(images=img_vlm, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = vlm_model(**inputs)
        p_vlm = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
        p_vlm_cache.append(p_vlm)

cache_time = time.perf_counter() - start_cache_time
print(f"Caching complete in {cache_time:.2f} seconds.")

# Convert to numpy arrays for fast vectorization
y_true_cache = np.array(y_true_cache)
p_cnn_cache = np.array(p_cnn_cache)
p_vlm_cache = np.array(p_vlm_cache)

# Pre-compute 8-class true labels
y_true_8_cache = np.array([category_mapping[final_classes[idx]] for idx in y_true_cache])

# 6. Sensitivity Sweep Phase
print("\n[PHASE 2] Executing Soft Fusion Weight Sweep & Plotting F1...")
results_summary = []

for w_cnn in cnn_weights_to_test:
    w_vlm = 1.0 - w_cnn

    # Vectorized Fusion Calculation
    p_fused = (w_cnn * p_cnn_cache) + (w_vlm * p_vlm_cache)
    y_pred = np.argmax(p_fused, axis=1)

    # 10-Class Metrics
    acc_10 = accuracy_score(y_true_cache, y_pred)
    macro_f1_10 = f1_score(y_true_cache, y_pred, average='macro')

    # 8-Class Metrics
    y_pred_names = [final_classes[idx] for idx in y_pred]
    y_pred_8 = [category_mapping[name] for name in y_pred_names]
    acc_8 = accuracy_score(y_true_8_cache, y_pred_8)

    results_summary.append({
        "Weight_CNN": float(w_cnn),
        "Weight_VLM": float(w_vlm),
        "Accuracy_10Class": acc_10,
        "Accuracy_8Class": acc_8,
        "Macro_F1_10Class": macro_f1_10
    })

    f1_scores_10 = f1_score(y_true_cache, y_pred, average=None)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    bars = plt.bar(final_classes, f1_scores_10, color='#66bb6a', edgecolor='#1b5e20', linewidth=1.5)
    plt.title(f'Arch 7: F1-Score (CNN={w_cnn:.2f} | VLM={w_vlm:.2f})', fontsize=16, fontweight='bold', pad=20, color='#1b5e20')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Arch7_F1_CNN_{w_cnn:.2f}_VLM_{w_vlm:.2f}.png'), dpi=300, bbox_inches='tight')
    plt.close()

df_results = pd.DataFrame(results_summary)
csv_path = os.path.join(output_dir, 'Arch7_Sensitivity_Summary.csv')
df_results.to_csv(csv_path, index=False)

# Find optimal weight based on 10-Class Accuracy
best_idx = df_results['Accuracy_10Class'].idxmax()
best_w_cnn = df_results.loc[best_idx, 'Weight_CNN']
best_acc_10 = df_results.loc[best_idx, 'Accuracy_10Class']
best_acc_8 = df_results.loc[best_idx, 'Accuracy_8Class']

print("\n" + "="*85)
print("SENSITIVITY ANALYSIS SUMMARY TABLE")
print("="*85)
print(df_results.to_string(index=False))
print("="*85)
print(f"🌟 OPTIMAL POINT: CNN = {best_w_cnn:.2f} | VLM = {1.0-best_w_cnn:.2f}")
print(f"   => 10-Class Acc = {best_acc_10:.4f} | 8-Class Acc = {best_acc_8:.4f}")
print("="*85)

# 7. Visualization

plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

# Plot 10-Class Accuracy
plt.plot(df_results['Weight_CNN'], df_results['Accuracy_10Class'], marker='o', markersize=8,
         color='#1b5e20', linewidth=2.5, label='10-Class Accuracy (Full)')

# Plot 8-Class Accuracy (Amber/Orange for contrast)
plt.plot(df_results['Weight_CNN'], df_results['Accuracy_8Class'], marker='^', markersize=8,
         color='#ff8f00', linewidth=2.5, linestyle='-', label='8-Class Accuracy (No Grading)')

# Highlight the best point for 10-Class
plt.plot(best_w_cnn, best_acc_10, marker='*', markersize=15, color='#d32f2f',
         label=f'Optimal 10-Class (Acc: {best_acc_10:.4f})')

plt.title('Architecture 7: Fusion Weight Sensitivity (10-Class vs 8-Class)', fontsize=16, fontweight='bold', pad=20, color='#424242')
plt.xlabel('CNN Weight (\u03B1)', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy', fontsize=13, fontweight='bold')
plt.xticks(np.arange(0, 1.05, 0.1))
plt.legend(loc='lower right', fontsize=12)

# Adding secondary X-axis for VLM Weight
secax = plt.gca().secondary_xaxis('top', functions=(lambda x: 1 - x, lambda x: 1 - x))
secax.set_xlabel('VLM Weight (1 - \u03B1)', fontsize=13, fontweight='bold')
secax.set_xticks(np.arange(0, 1.05, 0.1))

plt.tight_layout()
plot_path = os.path.join(output_dir, 'Arch7_Sensitivity_Curve_Combo.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close('all')

print(f"Process Complete. Combo plot and 21 F1 charts saved to: {output_dir}")
