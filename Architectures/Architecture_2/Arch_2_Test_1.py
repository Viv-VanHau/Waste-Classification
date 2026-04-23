# ==============================================================================
# ARCHITECTURE 2: DUAL-STAGE SEQUENTIAL CNN
# TEST 1: CONFIDENCE-BASED FALLBACK & MATERIAL RESCUE MECHANISM
# ==============================================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from matplotlib.colors import LinearSegmentedColormap

# 1. Path Definitions & Class Configurations
extract_dir = '/content/Base_Test_Data'
model_stage1_path = '/content/drive/MyDrive/Thesis_Project/Backbone_Stage1_Final_9429.h5'
model_stage2_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
output_dir = '/content/drive/MyDrive/Thesis_Project/Base_Test_Results'

os.makedirs(output_dir, exist_ok=True)

s1_classes = ['battery', 'glass', 'metal', 'organic_waste', 'paper_cardboard', 'plastic', 'textiles', 'trash']
s2_classes = ['battery', 'glass', 'metal_Grade_A', 'metal_Grade_B',
              'organic_waste', 'paper_cardboard', 'plastic_Grade_A',
              'plastic_Grade_B', 'textiles', 'trash']

# Helper function to map S2 class to S1 base material for accurate tracking
def get_base_material(class_name):
    if class_name in ['metal_Grade_A', 'metal_Grade_B']: return 'metal'
    if class_name in ['plastic_Grade_A', 'plastic_Grade_B']: return 'plastic'
    return class_name

# 2. Automated Directory Discovery
test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if 'battery' in dirs and 'glass' in dirs and 'trash' in dirs:
        test_dir = root
        break

print(f"Data directory located at: {test_dir}")

# 3. Model Instantiation
print("Loading Stage 1 Model (CNN 8-Class)...")
stage1_model = tf.keras.models.load_model(model_stage1_path, compile=False)

print("Loading Stage 2 Model (CNN 10-Class)...")
stage2_model = tf.keras.models.load_model(model_stage2_path, compile=False)

# 4. Data Pipeline Initialization
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    classes=s2_classes,
    class_mode='categorical',
    shuffle=False
)

total_samples = test_generator.samples
filenames = test_generator.filenames

# 5. Inference Pipeline
print(f"\nStarting inference pipeline on {total_samples} samples...")
print("-" * 120)
print(f"{'INDEX':<7} | {'ACTUAL':<17} | {'S1 PREDICT':<15} | {'CONF':<6} | {'S2 PREDICT':<17} | {'FINAL':<17} | {'STATUS'}")
print("-" * 120)

y_true = []
y_pred = []
routing_logs = []

# System metrics
stage2_calls = 0
grading_calls = 0
fallback_calls = 0
successful_rescues = 0

CONFIDENCE_THRESHOLD = 0.90

start_time = time.perf_counter()

for i in range(total_samples):
    img, label_batch = test_generator[i]
    true_class = s2_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # Stage 1 Inference
    p_s1 = stage1_model.predict(img, verbose=0)[0]
    s1_idx = np.argmax(p_s1)
    s1_conf = np.max(p_s1)
    c1 = s1_classes[s1_idx]

    c2 = "N/A"
    route_action = "Fast-Path (Bypass)"
    final_decision = c1

    # Stage 2 Routing Logic
    requires_grading = c1 in ['metal', 'plastic']
    requires_fallback = (not requires_grading) and (s1_conf < CONFIDENCE_THRESHOLD)

    if requires_grading or requires_fallback:
        stage2_calls += 1
        if requires_grading:
            grading_calls += 1
            route_action = "Grading Sub-routine"
        else:
            fallback_calls += 1
            route_action = "Confidence Fallback"

        # Execute Stage 2
        p_s2 = stage2_model.predict(img, verbose=0)[0]
        c2 = s2_classes[np.argmax(p_s2)]
        final_decision = c2

        # Determine if a successful material rescue occurred
        # A rescue is successful if S1 base prediction was wrong, but S2 final prediction is perfectly correct
        base_true = get_base_material(true_class)
        if requires_fallback and (c1 != base_true) and (final_decision == true_class):
            successful_rescues += 1
            route_action = "Successful Rescue"

    y_pred.append(final_decision)

    # Status Formatting
    status = "CORRECT" if final_decision == true_class else "INCORRECT"

    # Logging
    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "S1_Predict": c1,
        "S1_Confidence": float(s1_conf),
        "S2_Predict": c2,
        "Routing_Action": route_action,
        "Final_Decision": final_decision,
        "Is_Correct": status == "CORRECT"
    })

    print(f"{i+1:05d}   | {true_class:<17} | {c1:<15} | {s1_conf:.2f} | {c2:<17} | {final_decision:<17} | {status}")

end_time = time.perf_counter()
print("-" * 120)

# 6. Metrics & Performance Calculation
total_time = end_time - start_time
avg_time_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_ms

# Export CSV Log
df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch2_Test1_Routing_Log.csv')
df_log.to_csv(csv_path, index=False)

# 7. Reporting
print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time    : {total_time:.4f} seconds")
print(f"Average Time / Image    : {avg_time_ms:.2f} ms")
print(f"Frames Per Second       : {fps:.2f} FPS")
print(f"Stage 2 Total Calls     : {stage2_calls}/{total_samples} ({(stage2_calls/total_samples)*100:.1f}%)")
print(f"  -> Grading Calls      : {grading_calls}")
print(f"  -> Fallback Calls     : {fallback_calls}")
print(f"Successful Rescues      : {successful_rescues} errors prevented by Stage 2 Fallback")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=s2_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Arch2_Test1_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 2: DUAL-STAGE SEQUENTIAL CNN\n")
    f.write("TEST 1: CONFIDENCE-BASED FALLBACK\n")
    f.write("="*60 + "\n")
    f.write(f"FPS                     : {fps:.2f} FPS\n")
    f.write(f"Stage 2 Utilization     : {stage2_calls}/{total_samples}\n")
    f.write(f"Successful Rescues      : {successful_rescues}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 8. Visualizations

# 8.1 Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=s2_classes)
blue_colors = ["#ffffff", "#e3f2fd", "#90caf9", "#42a5f5", "#1e88e5", "#1565c0"]
custom_blue_cmap = LinearSegmentedColormap.from_list("custom_blue", blue_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_blue_cmap,
                 xticklabels=s2_classes, yticklabels=s2_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "sans-serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 2: Confidence-Based Routing (Confusion Matrix)',
          fontsize=18, fontweight='bold', pad=25, color='#1565c0')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch2_Test1_Confusion_Matrix.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

# 8.2 F1-Score Bar Chart
f1_scores = f1_score(y_true, y_pred, average=None, labels=s2_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(s2_classes, f1_scores, color='#42a5f5', edgecolor='#1565c0', linewidth=1.5)

plt.title('Architecture 2: F1-Score per Class', fontsize=16, fontweight='bold', pad=20, color='#1565c0')
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch2_Test1_F1_Scores.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Evaluation Process Complete, outputs saved to: {output_dir}")
