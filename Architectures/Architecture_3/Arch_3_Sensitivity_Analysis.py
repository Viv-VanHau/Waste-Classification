# ==============================================================================
# ARCHITECTURE 3: CONFIDENCE-ROUTED HYBRID SYSTEM
# SENSITIVITY ANALYSIS: GATING THRESHOLD OPTIMIZATION
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
from sklearn.metrics import classification_report, accuracy_score, f1_score


# 1. Path Definitions & Configurations
extract_dir = '/content/Base_Test_Data'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
vlm_base_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_LoRA_Output'
output_dir = '/content/drive/MyDrive/Thesis_Project/Base_Test_Results/Arch3_Sensitivity'

os.makedirs(output_dir, exist_ok=True)

s2_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
              'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
              'plastic_Grade_B', 'textiles', 'trash']

thresholds_to_test = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

category_mapping = {
    'battery': 'battery', 'glass': 'glass', 'metal_Grade_A': 'metal', 'metal_Grade_B': 'metal',
    'organic_waste': 'organic_waste', 'paper_cardboard': 'paper_cardboard', 'plastic_Grade_A': 'plastic',
    'plastic_Grade_B': 'plastic', 'textiles': 'textiles', 'trash': 'trash'
}
base_classes = sorted(list(set(category_mapping.values())))

# 2. Automated Directory Discovery
test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if 'battery' in dirs and 'glass' in dirs and 'trash' in dirs:
        test_dir = root
        break

print(f"Data directory located at: {test_dir}")

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
        print(f"Auto-found best LoRA Checkpoint: {model_stage2_path}")
    else:
        raise FileNotFoundError(f"CRITICAL ERROR: No checkpoints found in {vlm_base_dir}.")
else:
    print(f"Loading LoRA from base dir: {vlm_base_dir}")

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

# 5. Data Caching Phase
print(f"\n[PHASE 1] Pre-computing predictions for {total_samples} samples...")
y_true_cache = []
cnn_preds_cache = []
cnn_confs_cache = []
vlm_preds_cache = []

start_cache_time = time.perf_counter()

for i in range(total_samples):
    img_raw, label_batch = test_generator[i]
    true_class = s2_classes[np.argmax(label_batch[0])]
    y_true_cache.append(true_class)

    # CNN Inference
    img_cnn = preprocess_input(img_raw.astype(np.float32).copy())
    p_cnn = cnn_model.predict(img_cnn, verbose=0)[0]
    cnn_preds_cache.append(s2_classes[np.argmax(p_cnn)])
    cnn_confs_cache.append(np.max(p_cnn))

    # VLM Inference
    img_vlm = img_raw[0].astype(np.uint8)
    inputs = processor(images=img_vlm, return_tensors="pt").to(device)
    with torch.no_grad():
        out_vlm = vlm_model(**inputs)
        vlm_preds_cache.append(s2_classes[out_vlm.logits.argmax(-1).item()])

cache_time = time.perf_counter() - start_cache_time
print(f"Caching complete in {cache_time:.2f} seconds.")

# 6. Sensitivity Sweep Phase
print("\n[PHASE 2] Executing Threshold Sweep & Generating F1 Charts...")
results_summary = []

for tau in thresholds_to_test:
    y_pred_current = []
    vlm_calls = 0
    rescues = 0

    for i in range(total_samples):
        c1 = cnn_preds_cache[i]
        c2 = vlm_preds_cache[i]
        conf = cnn_confs_cache[i]
        actual = y_true_cache[i]

        if conf < tau:
            vlm_calls += 1
            final_decision = c2
            if c2 == actual and c1 != actual:
                rescues += 1
        else:
            final_decision = c1

        y_pred_current.append(final_decision)

    # 10-Class Metrics
    acc_10 = accuracy_score(y_true_cache, y_pred_current)
    macro_f1_10 = f1_score(y_true_cache, y_pred_current, average='macro')

    # 8-Class Mapping & Metrics
    y_true_8 = [category_mapping[lbl] for lbl in y_true_cache]
    y_pred_8 = [category_mapping[lbl] for lbl in y_pred_current]
    acc_8 = accuracy_score(y_true_8, y_pred_8)

    vlm_util_pct = (vlm_calls / total_samples) * 100

    results_summary.append({
        "Threshold": tau,
        "Accuracy_10Class": acc_10,
        "Accuracy_8Class": acc_8,
        "Macro_F1_10Class": macro_f1_10,
        "VLM_Calls": vlm_calls,
        "VLM_Util_%": vlm_util_pct,
        "Rescues": rescues
    })

    # Save individual classification reports (10-Class)
    report_str = classification_report(y_true_cache, y_pred_current, target_names=s2_classes, digits=4)
    with open(os.path.join(output_dir, f'Report_Threshold_{tau:.2f}.txt'), 'w') as f:
        f.write(f"ARCHITECTURE 3 - THRESHOLD: {tau:.2f}\n")
        f.write(f"10-Class Acc: {acc_10*100:.2f}% | 8-Class Acc: {acc_8*100:.2f}%\n")
        f.write("="*60 + "\n")
        f.write(report_str)

    f1_scores_10 = f1_score(y_true_cache, y_pred_current, average=None, labels=s2_classes)
    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")
    bars = plt.bar(s2_classes, f1_scores_10, color='#ab47bc', edgecolor='#4a148c', linewidth=1.5)
    plt.title(f'Arch 3: F1-Score per Class (Threshold $\\tau$ = {tau:.2f})', fontsize=16, fontweight='bold', pad=20, color='#4a148c')
    plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
    plt.ylim(0, 1.1)
    plt.xticks(rotation=45, ha='right', fontsize=11)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'Arch3_F1_Threshold_{tau:.2f}.png'), dpi=300, bbox_inches='tight')
    plt.close()

df_results = pd.DataFrame(results_summary)
csv_path = os.path.join(output_dir, 'Sensitivity_Analysis_Summary.csv')
df_results.to_csv(csv_path, index=False)

print("\n" + "="*95)
print("SENSITIVITY ANALYSIS SUMMARY TABLE")
print("="*95)
print(df_results.to_string(index=False))
print("="*95)

# 7. Visualization: Dual-Axis Sensitivity Plot
print("\nGenerating Combo Sensitivity Curve Plot...")

plt.figure(figsize=(12, 7))
sns.set_theme(style="whitegrid")

fig, ax1 = plt.subplots(figsize=(12, 7))

# Primary Axis: Accuracy (10-Class & 8-Class)
color_10class = '#7b1fa2'
color_8class = '#d81b60'

ax1.set_xlabel('Confidence Threshold ($\\tau$)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy', color='#424242', fontsize=13, fontweight='bold')

line1, = ax1.plot(df_results['Threshold'], df_results['Accuracy_10Class'], marker='o', markersize=8,
                  color=color_10class, linewidth=2.5, label='10-Class Accuracy (Full)')
line2, = ax1.plot(df_results['Threshold'], df_results['Accuracy_8Class'], marker='^', markersize=8,
                  color=color_8class, linewidth=2.5, linestyle='-', label='8-Class Accuracy (No Grading)')

ax1.tick_params(axis='y', labelcolor='#424242')
ax1.set_ylim(min(df_results['Accuracy_10Class']) - 0.02, max(df_results['Accuracy_8Class']) + 0.02)
ax1.legend(loc='upper left', fontsize=11)

# Secondary Axis: VLM Utilization
ax2 = ax1.twinx()
color_util = '#9e9e9e'
ax2.set_ylabel('VLM Utilization (%)', color=color_util, fontsize=13, fontweight='bold')
line3, = ax2.plot(df_results['Threshold'], df_results['VLM_Util_%'], marker='s', markersize=8,
                  color=color_util, linewidth=2.5, linestyle='--', label='VLM Utilization (%)')
ax2.tick_params(axis='y', labelcolor=color_util)
ax2.set_ylim(0, 100)

# Annotate Rescues on the Utilization line
for i, txt in enumerate(df_results['Rescues']):
    ax2.annotate(f"{txt} rescues", (df_results['Threshold'][i], df_results['VLM_Util_%'][i]),
                 textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, color='#616161')

plt.title('Architecture 3: Sensitivity Analysis (10-Class vs 8-Class & Utilization)', fontsize=16, fontweight='bold', pad=20)
fig.tight_layout()

plot_path = os.path.join(output_dir, 'Sensitivity_Analysis_Curve_Combo.png')
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close('all')
