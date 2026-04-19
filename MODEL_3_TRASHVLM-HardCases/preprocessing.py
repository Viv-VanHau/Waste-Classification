# =============================================
# 1. ERROR MINING FROM MODEL 2 - CNN 10 CLASSES
# =============================================
import os
import zipfile
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

output_dir = '/content/drive/MyDrive/Thesis_Project/Outputs_Stage3'
os.makedirs(output_dir, exist_ok=True)

zip_path = '/content/drive/MyDrive/Thesis_Project/UWCD_10Classes_Final.zip'
extract_dir = '/content/Dataset_Final_10Classes'

if not os.path.exists(extract_dir):
    print(f"Extracting Data...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

data_dir = extract_dir
subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
if len(subdirs) == 1:
    data_dir = os.path.join(data_dir, subdirs[0])

datagen = ImageDataGenerator(preprocessing_function=preprocess_input, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation',
    shuffle=False,
    seed=42
)

model_path = '/content/drive/MyDrive/Thesis_Project/Stage2_10Classes_Best.h5'
print(f"\nInputing model from: {model_path}")
model = load_model(model_path)

print("Running predictions...")
val_preds = model.predict(val_generator)
val_pred_classes = np.argmax(val_preds, axis=1)
val_true_classes = val_generator.classes
class_labels = list(val_generator.class_indices.keys())

target_classes = ['plastic_Grade_A', 'plastic_grade_B', 'metal_Grade_A', 'metal_Grade_B']
target_indices = [class_labels.index(c) for c in target_classes if c in class_labels]

misclassified_plastic = []
misclassified_metal = []
error_report = []

for i in range(len(val_true_classes)):
    true_idx = val_true_classes[i]
    pred_idx = val_pred_classes[i]

    if true_idx in target_indices:
        if true_idx != pred_idx:
            error_data = {
                'path': val_generator.filepaths[i],
                'true_name': class_labels[true_idx],
                'pred_name': class_labels[pred_idx],
                'confidence': val_preds[i][pred_idx] * 100
            }
            error_report.append(error_data)
            if 'plastic' in class_labels[true_idx].lower():
                misclassified_plastic.append(error_data)
            elif 'metal' in class_labels[true_idx].lower():
                misclassified_metal.append(error_data)

# ==============================================
# 2. XAI DIAGNOSIS - GRAD-CAM FOR PLASTIC ERRORS
# ==============================================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.cm as cm 
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("Running Grad-CAM for Plastic...")

target_tensor = None
for layer in model.layers:
    if 'global_average_pooling' in layer.name.lower() or 'global_max_pooling' in layer.name.lower():
        target_tensor = layer.input
        break

grad_model = Model(inputs=model.input, outputs=[target_tensor, model.output])

def make_gradcam_heatmap(img_array, grad_model, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_tensor)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

num_to_plot = min(6, len(misclassified_plastic))
if num_to_plot > 0:
    plastic_cases = sorted(misclassified_plastic, key=lambda x: x['confidence'], reverse=True)[:num_to_plot]

    fig, axes = plt.subplots(num_to_plot, 3, figsize=(15, 5 * num_to_plot))
    if num_to_plot == 1: axes = np.array([axes])
    fig.suptitle('Stage 3: Grad-CAM Analysis on Plastic Misclassifications', fontsize=20, fontweight='bold', y=1.02)

    for i, patient in enumerate(plastic_cases):
        img_orig = load_img(patient['path'], target_size=(224, 224))
        img_orig_array = img_to_array(img_orig)
        img_preprocessed = preprocess_input(np.expand_dims(np.copy(img_orig_array), axis=0))

        heatmap = make_gradcam_heatmap(img_preprocessed, grad_model)
        heatmap_resized = np.uint8(255 * heatmap)

        jet = plt.get_cmap('jet')
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_resized]

        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap).resize((224, 224))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap * 0.4 + img_orig_array)

        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f"Ground Truth: {patient['true_name']}", fontsize=14)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title("Grad-CAM Activation Map", fontsize=14)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(superimposed_img)
        axes[i, 2].set_title(f"Predicted: {patient['pred_name']}\n(Confidence: {patient['confidence']:.2f}%)", fontsize=14, color='darkred')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'GradCAM_Plastic_Errors.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
  
# ================================================
# 3. GRAD-CAM FOR METAL ERRORS & POSITIVE CONTRAST
# ================================================
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

print("Running Grd-CAM for Metal...")

num_metal_plot = min(6, len(misclassified_metal))
if num_metal_plot > 0:
    metal_cases = sorted(misclassified_metal, key=lambda x: x['confidence'], reverse=True)[:num_metal_plot]
    fig, axes = plt.subplots(num_metal_plot, 3, figsize=(15, 5 * num_metal_plot))
    if num_metal_plot == 1: axes = np.array([axes])
    fig.suptitle('Stage 3: Grad-CAM Analysis on Metal Misclassifications', fontsize=20, fontweight='bold', y=1.02)

    for i, patient in enumerate(metal_cases):
        img_orig = load_img(patient['path'], target_size=(224, 224))
        img_orig_array = img_to_array(img_orig)
        img_preprocessed = preprocess_input(np.expand_dims(np.copy(img_orig_array), axis=0))

        heatmap = make_gradcam_heatmap(img_preprocessed, grad_model)
        heatmap_resized = np.uint8(255 * heatmap)

        jet = plt.get_cmap('jet')
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_resized]

        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap).resize((224, 224))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap * 0.4 + img_orig_array)

        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f"Ground Truth: {patient['true_name']}", fontsize=14)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title("Grad-CAM Activation Map", fontsize=14)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(superimposed_img)
        axes[i, 2].set_title(f"Predicted: {patient['pred_name']}\n(Confidence: {patient['confidence']:.2f}%)", fontsize=14, color='darkred')
        axes[i, 2].axis('off')

    plt.tight_layout()
    metal_plot_path = os.path.join(output_dir, 'GradCAM_Metal_Errors.png')
    plt.savefig(metal_plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
# --- POSITIVE CONTRAST ---
print("Creating Positive Contrast...")
healthy_cases = []
for i in range(len(val_true_classes)):
    true_idx, pred_idx = val_true_classes[i], val_pred_classes[i]
    if true_idx == pred_idx and true_idx in target_indices:
        if val_preds[i][pred_idx] > 0.95: # Confidence > 95%
            healthy_cases.append({'path': val_generator.filepaths[i], 'name': class_labels[true_idx], 'confidence': val_preds[i][pred_idx] * 100})
            if len(healthy_cases) >= 3: break

if healthy_cases:
    fig, axes = plt.subplots(len(healthy_cases), 3, figsize=(15, 5 * len(healthy_cases)))
    if len(healthy_cases) == 1: axes = np.array([axes])
    fig.suptitle('Stage 3: Positive Contrast (Highly Confident Correct Predictions)', fontsize=20, fontweight='bold', y=1.05)

    for i, patient in enumerate(healthy_cases):
        img_orig = load_img(patient['path'], target_size=(224, 224))
        img_orig_array = img_to_array(img_orig)
        img_preprocessed = preprocess_input(np.expand_dims(np.copy(img_orig_array), axis=0))

        heatmap = make_gradcam_heatmap(img_preprocessed, grad_model)
        heatmap_resized = np.uint8(255 * heatmap)

        jet = plt.get_cmap('jet')
        jet_colors = jet(np.arange(256))[:, :3]
        jet_heatmap = jet_colors[heatmap_resized]

        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap).resize((224, 224))
        jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
        superimposed_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap * 0.4 + img_orig_array)

        axes[i, 0].imshow(img_orig)
        axes[i, 0].set_title(f"Ground Truth: {patient['name']}", fontsize=14)
        axes[i, 0].axis('off')

        axes[i, 1].imshow(heatmap, cmap='jet')
        axes[i, 1].set_title("Grad-CAM Activation Map", fontsize=14)
        axes[i, 1].axis('off')

        axes[i, 2].imshow(superimposed_img)
        axes[i, 2].set_title(f"Predicted: CORRECT\n(Confidence: {patient['confidence']:.2f}%)", fontsize=14, color='green')
        axes[i, 2].axis('off')

    plt.tight_layout()
    contrast_path = os.path.join(output_dir, 'GradCAM_Positive_Contrast.png')
    plt.savefig(contrast_path, dpi=300, bbox_inches='tight')
    plt.show()
  
# =====================================
# 3. AUTOMATED HEURISTIC ERROR ANALYSIS
# =====================================
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import warnings
warnings.filterwarnings('ignore')

# --- HEATMAP CREATING ---
target_tensor = None
for layer in model.layers:
    if 'global_average_pooling' in layer.name.lower() or 'global_max_pooling' in layer.name.lower():
        target_tensor = layer.input
        break
grad_model = Model(inputs=model.input, outputs=[target_tensor, model.output])

def get_heatmap(img_array, g_model, pred_idx):
    img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
    with tf.GradientTape() as tape:
        last_conv_output, preds = g_model(img_tensor)
        class_channel = preds[:, pred_idx]
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    max_val = tf.math.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros(heatmap.shape)

    heatmap = tf.maximum(heatmap, 0) / max_val
    return heatmap.numpy()

# --- AUTOMATED HEURISTIC ERROR ANALYSIS ---
def diagnose_error_cause(img_path, g_model, pred_idx):
    try:
        img_orig = load_img(img_path, target_size=(224, 224))
        img_orig_array = img_to_array(img_orig)
        img_preprocessed = preprocess_input(np.expand_dims(np.copy(img_orig_array), axis=0))

        heatmap = get_heatmap(img_preprocessed, g_model, pred_idx)

        if heatmap is None or np.isnan(heatmap).any() or np.max(heatmap) == 0:
            return 'Boundary Ambiguity' 
        heatmap = np.float32(heatmap)
        heatmap_resized = cv2.resize(heatmap, (224, 224))

        # Binarize heatmap
        hot_mask = (heatmap_resized > 0.6).astype(np.uint8)

        if np.sum(hot_mask) == 0:
            return 'Boundary Ambiguity'

        # Background Bias
        M = cv2.moments(hot_mask)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            dist_from_center = np.sqrt((cX - 112)**2 + (cY - 112)**2)
            if dist_from_center > 70:
                return 'Background Bias'

        masked_img = cv2.bitwise_and(img_orig_array.astype(np.uint8), img_orig_array.astype(np.uint8), mask=hot_mask)
        gray_masked = cv2.cvtColor(masked_img, cv2.COLOR_RGB2GRAY)

        # Specular Reflection
        mean_brightness = np.sum(gray_masked) / (np.sum(hot_mask) + 1e-5)
        if mean_brightness > 180:
            return 'Specular Reflection'

        # Label Interference
        edges = cv2.Canny(gray_masked, 100, 200)
        edge_density = np.sum(edges > 0) / (np.sum(hot_mask) + 1e-5)
        if edge_density > 0.15:
            return 'Label Interference'

        return 'Boundary Ambiguity'

    except Exception as e:
        return 'Boundary Ambiguity'

print("\n314 errors extracting...")
plastic_causes = {'Label Interference': 0, 'Specular Reflection': 0, 'Background Bias': 0, 'Boundary Ambiguity': 0}
metal_causes = {'Label Interference': 0, 'Specular Reflection': 0, 'Background Bias': 0, 'Boundary Ambiguity': 0}

for item in tqdm(misclassified_plastic, desc="Plastic Analyzing"):
    pred_idx = class_labels.index(item['pred_name'])
    cause = diagnose_error_cause(item['path'], grad_model, pred_idx)
    plastic_causes[cause] += 1

for item in tqdm(misclassified_metal, desc="Metal Analyzing"):
    pred_idx = class_labels.index(item['pred_name'])
    cause = diagnose_error_cause(item['path'], grad_model, pred_idx)
    metal_causes[cause] += 1

total_plastic = len(misclassified_plastic) or 1
total_metal = len(misclassified_metal) or 1

plastic_pct = [round(plastic_causes[k] / total_plastic * 100, 1) for k in plastic_causes.keys()]
metal_pct = [round(metal_causes[k] / total_metal * 100, 1) for k in metal_causes.keys()]

labels = df_taxonomy['Error_Root_Cause']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, df_taxonomy['Plastic_Errors_Pct'], width, label='Plastic Errors (%)', color='#2ca02c', edgecolor='black')
rects2 = ax.bar(x + width/2, df_taxonomy['Metal_Errors_Pct'], width, label='Metal Errors (%)', color='#7f7f7f', edgecolor='black')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontweight='bold')

autolabel(rects1)
autolabel(rects2)

ax.set_ylabel('Percentage within Material Category (%)', fontsize=12, fontweight='bold')
ax.set_title('Stage 3: Automated Error Root Cause Analysis via OpenCV & Grad-CAM', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
ax.legend(fontsize=12)
ax.set_ylim(0, max(max(plastic_pct), max(metal_pct)) + 15)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
chart_path = os.path.join(output_dir, 'Stage3_Real_Material_Error_Chart.png')
plt.savefig(chart_path, dpi=300)
plt.show()

# ================================
# 4. AUTO OVERSAMPLING FOR CLASSES
# ================================
import os
import shutil
import random

dataset_dir = '/content/drive/MyDrive/Thesis_Project/Outputs_Stage3/Stage6_VLM_Training_Data'
target_count = 50 

print("Deleating previous oversampling (if have any)...")
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if "Oversampled" in file:
            os.remove(os.path.join(root, file))

class_files = {}
for root, dirs, files in os.walk(dataset_dir):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
"True-metal_Grade_B___..."
                true_label_str = file.split('___')[0].replace('True-', '')
                if true_label_str not in class_files:
                    class_files[true_label_str] = []
                class_files[true_label_str].append(os.path.join(root, file))
            except:
                continue

for class_name, file_paths in class_files.items():
    num_files = len(file_paths)
    if 0 < num_files < target_count:
        num_to_add = target_count - num_files
        print(f"  🔹 Class {class_name}: {num_files} images -> Buffing {num_to_add} images")
        for i in range(num_to_add):
            src_path = random.choice(file_paths)
            src_dir, src_file = os.path.split(src_path)

            parts = src_file.split("___")
            if len(parts) > 1:
                new_name = f"{parts[0]}___Oversampled_{i}_{parts[1]}"
            else:
                new_name = f"Oversampled_{i}_{src_file}"

            dst_path = os.path.join(src_dir, new_name)
            shutil.copy(src_path, dst_path)

print("Done buffing")
