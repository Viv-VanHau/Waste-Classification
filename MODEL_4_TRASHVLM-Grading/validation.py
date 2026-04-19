import os
import shutil
import zipfile
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import AutoImageProcessor, ViTForImageClassification, Trainer, TrainingArguments
from peft import PeftModel
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

zip_path = '/content/drive/MyDrive/Thesis_Project/UWCD_10Classes_Final.zip'
extract_dir = '/content/UWCD_10_Classes_Temp'
subset_dir = '/content/UWCD_4_Classes_Subset'

if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)

os.makedirs(subset_dir, exist_ok=True)
target_classes = ['metal_Grade_A', 'metal_Grade_B', 'plastic_Grade_A', 'plastic_grade_B']

for root, dirs, files in os.walk(extract_dir):
    if len(dirs) == 10:
        for cls in target_classes:
            src = os.path.join(root, cls)
            dst = os.path.join(subset_dir, cls)
            if not os.path.exists(dst):
                shutil.copytree(src, dst)
        break

dataset = load_dataset("imagefolder", data_dir=subset_dir)
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42)
val_dataset = dataset['test']
vlm_classes = val_dataset.features["label"].names

model_dir = '/content/drive/MyDrive/Thesis_Project/TrashVLM_Grading_Model'
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")

def transforms(examples):
    pixel_values = [processor(img.convert("RGB"), return_tensors="pt")["pixel_values"][0] for img in examples["image"]]
    return {"pixel_values": pixel_values, "labels": examples["label"]}

val_dataset = val_dataset.with_transform(transforms)

base_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k", num_labels=4, ignore_mismatched_sizes=True)
vlm_model = PeftModel.from_pretrained(base_model, model_dir)

eval_args = TrainingArguments(
    output_dir="/tmp/eval", 
    per_device_eval_batch_size=32, 
    report_to="none",
    remove_unused_columns=False
)
trainer = Trainer(model=vlm_model, args=eval_args)

predictions, y_true_vlm, _ = trainer.predict(val_dataset)
y_pred_vlm = np.argmax(predictions, axis=1)

print(classification_report(y_true_vlm, y_pred_vlm, target_names=vlm_classes, digits=4))

output_dir = '/content/drive/MyDrive/Thesis_Project/Outputs_Stage7'
os.makedirs(output_dir, exist_ok=True)
cm = confusion_matrix(y_true_vlm, y_pred_vlm)

blue_colors = ["#ffffff", "#e3f2fd", "#90caf9", "#2196f3", "#0d47a1"]
custom_blue_cmap = LinearSegmentedColormap.from_list("white_blue", blue_colors, N=256)

plt.figure(figsize=(10, 8))
sns.set_theme(style="white")
sns.heatmap(cm, annot=True, fmt='d', cmap=custom_blue_cmap, 
            xticklabels=vlm_classes, yticklabels=vlm_classes, 
            annot_kws={"size": 13, "weight": "bold", "family": "serif"}, 
            linewidths=0, cbar=True)

plt.title('TrashVLM (4-Class Grading)', fontsize=18, fontweight='bold', pad=20, family='serif', color='#0d47a1')
plt.ylabel('Actual Category', fontsize=14, fontweight='bold', family='serif')
plt.xlabel('AI Predicted Category', fontsize=14, fontweight='bold', family='serif')
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()

cm_path = os.path.join(output_dir, 'TrashVLM_Validation_Matrix_Model4.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
plt.show()
