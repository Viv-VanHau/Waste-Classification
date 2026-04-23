# ==============================================================================
# ARCHITECTURE 2: DUAL-STAGE SEQUENTIAL CNN
# TEST 3: CALIBRATED CONSTRAINED ROUTING (THRESHOLD SHIFTING)
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

# Define masking indices and calibration thresholds
# We lower the threshold for Grade A because S2 is heavily biased towards Grade B
routing_config = {
    'metal': {
        'indices': [2, 3], # [Grade A, Grade B]
        'threshold_a': 0.30
    },
    'plastic': {
        'indices': [6, 7], # [Grade A, Grade B]
        'threshold_a': 0.40
    }
}

# 2. Automated Directory Discovery
test_dir = extract_dir
for root, dirs, files in os.walk(extract_dir):
    if 'battery' in dirs and 'glass' in dirs and 'trash' in dirs:
        test_dir = root
        break

print(f"Data directory located at: {test_dir}")

# 3. Model Instantiation
print("Loading Models...")
stage1_model = tf.keras.models.load_model(model_stage1_path, compile=False)
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
print("-" * 125)
print(f"{'INDEX':<7} | {'ACTUAL':<17} | {'S1 PREDICT':<15} | {'P_REL(A)':<8} | {'FINAL DECISION':<17} | {'STATUS'}")
print("-" * 125)

y_true = []
y_pred = []
routing_logs = []
stage2_calls = 0
calibration_shifts = 0

start_time = time.perf_counter()

for i in range(total_samples):
    img, label_batch = test_generator[i]
    true_class = s2_classes[np.argmax(label_batch[0])]
    y_true.append(true_class)
    filename = filenames[i]

    # Stage 1 Inference
    p_s1 = stage1_model.predict(img, verbose=0)[0]
    c1 = s1_classes[np.argmax(p_s1)]

    final_decision = c1
    p_rel_a_str = "N/A"

    # Stage 2 Routing with Calibration
    if c1 in ['metal', 'plastic']:
        stage2_calls += 1
        p_s2 = stage2_model.predict(img, verbose=0)[0]

        config = routing_config[c1]
        idx_a, idx_b = config['indices']
        threshold_a = config['threshold_a']

        # Extract probabilities for Grade A and Grade B
        prob_a = p_s2[idx_a]
        prob_b = p_s2[idx_b]

        # Calculate relative probability of Grade A within the masked domain
        epsilon = 1e-9
        relative_prob_a = prob_a / (prob_a + prob_b + epsilon)
        p_rel_a_str = f"{relative_prob_a:.3f}"

        # Normal argmax would require relative_prob_a > 0.5. We shift it based on threshold_a.
        if relative_prob_a >= threshold_a:
            final_decision = s2_classes[idx_a]
        else:
            final_decision = s2_classes[idx_b]

        # Track if the calibration shift altered the standard argmax decision
        if relative_prob_a >= threshold_a and relative_prob_a < 0.5:
            calibration_shifts += 1

    y_pred.append(final_decision)

    status = "CORRECT" if final_decision == true_class else "INCORRECT"

    routing_logs.append({
        "Filename": filename,
        "Actual_Class": true_class,
        "S1_Predict": c1,
        "Relative_Prob_A": p_rel_a_str,
        "Final_Decision": final_decision,
        "Is_Correct": status == "CORRECT"
    })

    print(f"{i+1:05d}   | {true_class:<17} | {c1:<15} | {p_rel_a_str:<8} | {final_decision:<17} | {status}")

end_time = time.perf_counter()
print("-" * 125)

# 6. Metrics & Performance Calculation
total_time = end_time - start_time
avg_time_ms = (total_time / total_samples) * 1000
fps = 1000 / avg_time_ms

df_log = pd.DataFrame(routing_logs)
csv_path = os.path.join(output_dir, 'Arch2_Test3_Routing_Log.csv')
df_log.to_csv(csv_path, index=False)

# 7. Reporting
print("\n" + "="*60)
print("SYSTEM PERFORMANCE METRICS")
print("="*60)
print(f"Total Inference Time    : {total_time:.4f} seconds")
print(f"Average Time / Image    : {avg_time_ms:.2f} ms")
print(f"Frames Per Second       : {fps:.2f} FPS")
print(f"Stage 2 Utilization     : {stage2_calls}/{total_samples} samples")
print(f"Calibration Shifts      : {calibration_shifts} predictions corrected by threshold shifting")

print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
report_str = classification_report(y_true, y_pred, target_names=s2_classes, digits=4)
print(report_str)

report_path = os.path.join(output_dir, 'Arch2_Test3_Classification_Report.txt')
with open(report_path, 'w') as f:
    f.write("ARCHITECTURE 2: DUAL-STAGE SEQUENTIAL CNN\n")
    f.write("TEST 3: CALIBRATED CONSTRAINED ROUTING\n")
    f.write("="*60 + "\n")
    f.write(f"FPS                     : {fps:.2f} FPS\n")
    f.write(f"Calibration Shifts      : {calibration_shifts}\n")
    f.write("="*60 + "\n")
    f.write(report_str)

# 8. Visualizations

cm = confusion_matrix(y_true, y_pred, labels=s2_classes)
blue_colors = ["#ffffff", "#e3f2fd", "#bbdefb", "#64b5f6", "#1e88e5", "#0d47a1"]
custom_blue_cmap = LinearSegmentedColormap.from_list("custom_blue", blue_colors, N=256)

plt.figure(figsize=(14, 11))
sns.set_theme(style="white")
ax = sns.heatmap(cm, annot=True, fmt='d', cmap=custom_blue_cmap,
                 xticklabels=s2_classes, yticklabels=s2_classes,
                 annot_kws={"size": 12, "weight": "bold", "family": "sans-serif"},
                 linewidths=0, cbar=True)

plt.title('Architecture 2: Calibrated Routing (Confusion Matrix)',
          fontsize=18, fontweight='bold', pad=25, color='#0d47a1')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.yticks(rotation=0, fontsize=11)
plt.tight_layout()

cm_save_path = os.path.join(output_dir, 'Arch2_Test3_Confusion_Matrix.png')
plt.savefig(cm_save_path, dpi=300, bbox_inches='tight')
plt.close()

f1_scores = f1_score(y_true, y_pred, average=None, labels=s2_classes)

plt.figure(figsize=(12, 6))
sns.set_theme(style="whitegrid")
bars = plt.bar(s2_classes, f1_scores, color='#42a5f5', edgecolor='#0d47a1', linewidth=1.5)

plt.title('Architecture 2: F1-Score per Class', fontsize=16, fontweight='bold', pad=20, color='#0d47a1')
plt.ylabel('F1-Score', fontsize=12, fontweight='bold')
plt.ylim(0, 1.1)
plt.xticks(rotation=45, ha='right', fontsize=11)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.015, f'{yval:.4f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
f1_save_path = os.path.join(output_dir, 'Arch2_Test3_F1_Scores.png')
plt.savefig(f1_save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Evaluation Process Complete, outputs saved to: {output_dir}")
